# train.py
import os
os.environ["XFORMERS_DISABLED"] = "1"

from pathlib import Path

import torch
from torch import GradScaler
from torch.utils.data import DataLoader
from torch.nn import MSELoss
# from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau ,OneCycleLR
from tqdm.auto import tqdm
import datetime
import time
import random
import draccus
from dlofeel_state.configs.train import TrainConfig
from dlofeel_state.data.dataset import DLOH5SampleDataset
from dlofeel_state.models.dlo_state_model import DLOStateModel
from dlofeel_state.losses.chamfer import chamfer_loss, mean_point_distance_per_sample, chamfer_distance_per_sample
from dlofeel_state.engine.metrics import extract_session, reduce_metrics
from dlofeel_state.data.transforms import check_no_session_overlap

from dlofeel_state.losses.weighted import WeightedLoss
#
import wandb

import numpy as np
# load config from yaml
import yaml
import h5py
import json
import shutil

def _ensure_cublas_workspace_config_for_determinism() -> None:
    """Set CuBLAS workspace config required by deterministic PyTorch on CUDA>=10.2.

    PyTorch raises at runtime for some CuBLAS ops under deterministic mode unless
    CUBLAS_WORKSPACE_CONFIG is set before those ops execute.
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        return
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA operations.")

class BestRunsManager:
    """Maintains a leaderboard of the top-K best training runs.

    Persists a ``leaderboard.json`` inside *best_runs_dir* that tracks the
    best validation loss achieved by each experiment.  Each experiment
    occupies at most **one** slot (its personal-best epoch).  When a new
    run beats the worst entry in a full leaderboard, the worst entry's
    checkpoint directory is deleted and replaced.
    """

    def __init__(self, best_runs_dir: Path, max_entries: int = 5) -> None:
        self.best_runs_dir = Path(best_runs_dir)
        self.max_entries = max_entries
        self.best_runs_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard_path = self.best_runs_dir / "leaderboard.json"

    # ---- persistence helpers ------------------------------------------------

    def _load_leaderboard(self) -> list[dict]:
        """Load the leaderboard from disk (empty list if file missing)."""
        if self.leaderboard_path.exists():
            with open(self.leaderboard_path, "r") as f:
                return json.load(f)
        return []

    def _save_leaderboard(self, entries: list[dict]) -> None:
        """Sort ascending by val_loss and write to disk."""
        entries.sort(key=lambda e: e["val_loss"])
        with open(self.leaderboard_path, "w") as f:
            json.dump(entries, f, indent=2)

    # ---- public API ---------------------------------------------------------

    def try_update(
        self,
        exp_name: str,
        val_loss: float,
        epoch: int,
        ckpt_src_dir: Path,
    ) -> None:
        """Try to insert / update this run in the leaderboard.

        Args:
            exp_name: Unique experiment name (directory name).
            val_loss: Current best validation loss for this run.
            epoch: Epoch at which the best val_loss was achieved.
            ckpt_src_dir: Path to the experiment's checkpoint directory
                          (must contain ``best.pt``, ``config.yaml``, etc.).
        """
        entries = self._load_leaderboard()
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")

        # Check if this run already has a slot
        existing_idx: int | None = None
        for idx, entry in enumerate(entries):
            if entry["exp_name"] == exp_name:
                existing_idx = idx
                break

        if existing_idx is not None:
            # Run already tracked -- update in-place (caller only invokes us
            # when the run improves, so val_loss is guaranteed to be better).
            entries[existing_idx]["val_loss"] = val_loss
            entries[existing_idx]["epoch"] = epoch
            entries[existing_idx]["timestamp"] = timestamp
            self._copy_checkpoint(exp_name, ckpt_src_dir)
            self._save_leaderboard(entries)
            print(f"[BestRuns] Updated {exp_name} -> val_loss={val_loss:.6f} (epoch {epoch})")
            return

        # New run -- check if there is room or if we can evict the worst
        if len(entries) < self.max_entries:
            # Room available
            entries.append({
                "exp_name": exp_name,
                "val_loss": val_loss,
                "epoch": epoch,
                "timestamp": timestamp,
            })
            self._copy_checkpoint(exp_name, ckpt_src_dir)
            self._save_leaderboard(entries)
            print(f"[BestRuns] Added {exp_name} -> val_loss={val_loss:.6f} (epoch {epoch})  "
                  f"[{len(entries)}/{self.max_entries}]")
            return

        # Leaderboard full -- compare against worst (last after sort)
        entries.sort(key=lambda e: e["val_loss"])
        worst = entries[-1]
        if val_loss < worst["val_loss"]:
            # Evict worst
            evicted_dir = self.best_runs_dir / worst["exp_name"]
            if evicted_dir.exists():
                shutil.rmtree(evicted_dir)
            print(f"[BestRuns] Evicted {worst['exp_name']} (val_loss={worst['val_loss']:.6f})")
            entries.pop(-1)

            entries.append({
                "exp_name": exp_name,
                "val_loss": val_loss,
                "epoch": epoch,
                "timestamp": timestamp,
            })
            self._copy_checkpoint(exp_name, ckpt_src_dir)
            self._save_leaderboard(entries)
            print(f"[BestRuns] Added {exp_name} -> val_loss={val_loss:.6f} (epoch {epoch})  "
                  f"[{len(entries)}/{self.max_entries}]")
        # else: not good enough -- nothing to do

    # ---- file helpers -------------------------------------------------------

    def _copy_checkpoint(self, exp_name: str, src_dir: Path) -> None:
        """Copy checkpoint files into the best-runs directory."""
        dst_dir = self.best_runs_dir / exp_name
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def log_model_info(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    use_wandb: bool = False,
    log_file: Path | None = None,
    train_sessions: int | None = None,
    val_sessions: int | None = None,
) -> None:
    """Log model size, trainable parameters, per-component breakdown, and inference time.

    Args:
        model: The PyTorch model to analyze
        config: Configuration dict with model parameters
        device: Device to run inference timing on
        use_wandb: Whether to log metrics to WandB
        log_file: Optional path to training log file to append summary
        train_sessions: Number of training sessions (folders)
        val_sessions: Number of validation/test sessions (folders)
    """
    def count_params(module: torch.nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    total, trainable = count_params(model)
    frozen = total - trainable

    # Size in MB (assuming float32 = 4 bytes)
    size_mb = total * 4 / (1024 ** 2)

    # Measure inference time with dummy inputs
    batch_size = config.get("batch_size", 8)
    img_h = config.get("img_size_H", 320)
    img_w = config.get("img_size_W", 240)
    only_pose = config.get("only_pose", False)

    # Create dummy inputs: poses [B, 7], imgs [B, 4, 6, H, W] (4 digits, 6 channels for temporal pair)
    dummy_poses = torch.randn(batch_size, 7, device=device)
    if not only_pose:
        dummy_imgs = torch.randn(batch_size, 4, 6, img_h, img_w, device=device)
    else:
        dummy_imgs = None

    # Warm-up runs
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_poses, dummy_imgs)

    # Timed runs
    if device.type == "cuda":
        torch.cuda.synchronize()

    n_runs = 10
    start_time = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_poses, dummy_imgs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    mean_inference_ms = (time.time() - start_time) / n_runs * 1000
    model.train()

    # Build summary lines
    lines = [
        "",
        "=" * 60,
        "MODEL SUMMARY",
        "=" * 60,
        f"Total parameters:     {total:>12,}",
        f"Trainable parameters: {trainable:>12,} ({100*trainable/total:.1f}%)",
        f"Frozen parameters:    {frozen:>12,} ({100*frozen/total:.1f}%)",
        f"Model size:           {size_mb:>12.2f} MB",
        f"Mean inference time:  {mean_inference_ms:>12.2f} ms (batch_size={batch_size})",
    ]

    # Add dataset session counts if provided
    if train_sessions is not None or val_sessions is not None:
        lines.append("-" * 60)
        lines.append("Dataset sessions:")
        if train_sessions is not None:
            lines.append(f"  Train sessions:    {train_sessions:>12}")
        if val_sessions is not None:
            lines.append(f"  Test sessions:     {val_sessions:>12}")

    lines.extend([
        "-" * 60,
        "Per-component breakdown:",
    ])

    component_stats = {}
    for name, child in model.named_children():
        child_total, child_trainable = count_params(child)
        if child_total > 0:
            lines.append(f"  {name:25s} {child_total:>10,} params ({child_trainable:,} trainable)")
            component_stats[name] = {"total": child_total, "trainable": child_trainable}
    lines.append("=" * 60)
    lines.append("")

    # Print to console
    for line in lines:
        print(line)

    # Write to log file
    if log_file is not None:
        with open(log_file, "a") as f:
            for line in lines:
                f.write(line + "\n")

    # Log to wandb
    if use_wandb:
        log_dict = {
            "model/total_params": total,
            "model/trainable_params": trainable,
            "model/frozen_params": frozen,
            "model/size_mb": size_mb,
            "model/mean_inference_ms": mean_inference_ms,
            **{f"model/component/{k}/total": v["total"] for k, v in component_stats.items()},
            **{f"model/component/{k}/trainable": v["trainable"] for k, v in component_stats.items()},
        }
        # Add session counts if provided
        if train_sessions is not None:
            log_dict["dataset/train_sessions"] = train_sessions
        if val_sessions is not None:
            log_dict["dataset/val_sessions"] = val_sessions
        wandb.log(log_dict)


def apply_loss_weight_override(config: dict, use_weighted_loss: bool) -> dict:
    """Optionally zero the physics loss terms (edges/length/curvature) in *config*.

    Implements the "no L_weight" ablation arm by weight-zeroing only: when
    *use_weighted_loss* is False, ``w_edges/w_lengths/w_kb`` are set to 0 so the
    WeightedLoss reduces to pure position MSE (``w_nodes`` retained), while
    ``w_nodes/mse_weight/chamfer_weight`` are left untouched. The terms are still
    computed by the loss class (no speed-skip); only their gradients vanish.

    Args:
        config: The merged run config (mutated in place).
        use_weighted_loss: Whether to keep the physics weights. False zeros them.

    Returns:
        The same *config* dict, with ``use_weighted_loss`` recorded for provenance.
    """
    config["use_weighted_loss"] = use_weighted_loss
    if not use_weighted_loss:
        config["w_edges"] = 0
        config["w_lengths"] = 0
        config["w_kb"] = 0
    return config


def apply_active_sensors_mask(imgs: torch.Tensor, active_sensors: list[int] | None, n_digits: int = 4) -> torch.Tensor:
    """Zero out image channels for sensors not in active_sensors. Keeps shape [B, n_digits, C, H, W]."""
    if active_sensors is None or len(active_sensors) == n_digits:
        return imgs
    imgs = imgs.clone()
    for i in range(n_digits):
        if i not in active_sensors:
            imgs[:, i] = 0.0
    return imgs

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_reproducibility(seed: int, deterministic: bool = False, deterministic_warn_only: bool = False) -> None:
    """Seed all RNGs and optionally enable deterministic PyTorch behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        _ensure_cublas_workspace_config_for_determinism()
        # Deterministic kernels improve reproducibility at the cost of speed.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)

# ------------------------------ Train ------------------------------
def train(
    train_data_root: str | None = None,
    val_data_root: str | None = None,
    blank_ref_root: str | None = None,
    config_path: str = "configs/train/default.yaml",
    normalize: bool | None = None,
    normalize_inputs: bool | None = None,
    norm_stats_path: str | None = None,
    use_diff_images: bool | None = None,
    use_blank_ref: bool | None = None,
    only_pose: bool | None = None,
    single_encoder: bool | None = None,
    image_backbone_type: str | None = None,
    use_spline: bool | None = None,
    n_control_points: int | None = None,
    num_resample_points: int | None = None,
    out_dir: str = "results",
    img_size_H: int | None = None,
    img_size_W: int | None = None,
    n_digits: int = 4,
    n_markers: int = 14,
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    dropout: float | None = None,
    weight_decay: float | None = None,
    val_split: float = 0.1,
    num_workers: int | None = None,
    seed: int | None = None,
    with_att: bool | None = None,
    max_batches_per_epoch: int = 0,  # 0 = no limit, >0 = limit batches per epoch
    max_eval_batches: int = 0,  # 0 = full val set, >0 = limit eval batches (for fast testing)
    wandb_project: str = "dlo-state-estimation",
    wandb_entity: str | None = None,
    use_wandb: bool | None = None,
    active_sensors: list[int] | None = None,  # Which digit indices [0,1,2,3] = fu,fd,ru,rd are active; None = all
    exp_name_override: str | None = None,  # If set, use this as exp_name (for ablation scripts)
    deterministic: bool = False,  # If True, enforce deterministic PyTorch ops where possible
    deterministic_warn_only: bool = False,  # If True, warn (instead of error) for non-deterministic ops
    model_class: type | None = None,  # Override model class (default: DLOStateModel)
    drop_augment: bool = False,  # If True, wrap train_ds with the 15-mask sensor-drop augmentation
    use_weighted_loss: bool | None = None,  # None -> config.get('use_weighted_loss', True); False zeros physics terms
    init_from: str | None = None,  # If set, load state_dict (strict=False) for weights-only fine-tune init
):
    config = load_config(config_path)
    # Override config only for values explicitly provided via args/CLI.
    if num_resample_points is not None:
        config["num_output_pts"] = num_resample_points
    if use_diff_images is not None:
        config["use_diff_images"] = use_diff_images
    if use_blank_ref is not None:
        config["use_blank_ref"] = use_blank_ref
    if normalize is not None:
        config["normalize"] = normalize
    if normalize_inputs is not None:
        config["normalize_inputs"] = normalize_inputs
    if only_pose is not None:
        config["only_pose"] = only_pose
    if with_att is not None:
        config["with_att"] = with_att
    if image_backbone_type is not None:
        config["model_params"]["image_backbone"]["type"] = image_backbone_type
    if single_encoder is not None:
        config["model_params"]["image_backbone"]["single_encoder"] = single_encoder
    if use_spline is not None:
        config["model_params"]["use_spline"] = use_spline
    if n_control_points is not None:
        config["n_control_points"] = n_control_points
    if img_size_H is not None:
        config["img_size_H"] = img_size_H
    if img_size_W is not None:
        config["img_size_W"] = img_size_W
    if batch_size is not None:
        config["batch_size"] = batch_size
    if lr is not None:
        config["lr"] = lr
    if weight_decay is not None:
        config["weight_decay"] = weight_decay
    if dropout is not None:
        config["dropout"] = dropout
    if epochs is not None:
        config["epochs"] = epochs
    if train_data_root is not None:
        config["train_data_root"] = train_data_root
    if val_data_root is not None:
        config["val_data_root"] = val_data_root
    if blank_ref_root is not None:
        config["blank_ref_root"] = blank_ref_root
    if norm_stats_path is not None:
        config["norm_stats_path"] = norm_stats_path
    if seed is not None:
        config["seed"] = seed
    if num_workers is not None:
        config["num_workers"] = num_workers
    if active_sensors is not None:
        config["active_sensors"] = active_sensors
    if deterministic is not None:
        config["deterministic"] = deterministic
    if deterministic_warn_only is not None:
        config["deterministic_warn_only"] = deterministic_warn_only
    # Provenance: record drop augmentation in the saved config.yaml (set before
    # the config dump below). The eval ignores unknown keys.
    config["drop_augment"] = drop_augment

    # Resolve final runtime values from merged config.
    train_data_root_cfg = config.get("train_data_root")
    val_data_root_cfg = config.get("val_data_root")
    blank_ref_root_cfg = config.get("blank_ref_root")
    norm_stats_path_cfg = config.get("norm_stats_path", "")
    if train_data_root_cfg is None or val_data_root_cfg is None or blank_ref_root_cfg is None:
        raise ValueError(
            "Missing dataset roots in config/args. "
            "Expected train_data_root, val_data_root, and blank_ref_root."
        )
    train_data_root = str(train_data_root_cfg)
    val_data_root = str(val_data_root_cfg)
    blank_ref_root = str(blank_ref_root_cfg)
    norm_stats_path = str(norm_stats_path_cfg)
    num_resample_points = int(config.get("num_output_pts", 0))
    normalize = bool(config.get("normalize", False))
    normalize_inputs = bool(config.get("normalize_inputs", False))
    use_diff_images = bool(config.get("use_diff_images", False))
    use_blank_ref = bool(config.get("use_blank_ref", False))
    only_pose = bool(config.get("only_pose", False))
    with_att = bool(config.get("with_att", False))
    single_encoder = bool(config["model_params"]["image_backbone"].get("single_encoder", False))
    img_size_H = int(config.get("img_size_H", 320))
    img_size_W = int(config.get("img_size_W", 240))
    batch_size = int(config.get("batch_size", 16))
    epochs = int(config.get("epochs", 10))
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    dropout = float(config.get("dropout", 0.1))
    num_workers = int(config.get("num_workers", 4))
    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", False))
    deterministic_warn_only = bool(config.get("deterministic_warn_only", False))
    use_wandb = True if use_wandb is None else use_wandb

    def _build_exp_name() -> str:
        parameter_str = (
            f"np{config['num_output_pts']}_sinenc{config['model_params']['image_backbone']['single_encoder']}"
            f"_att{config['with_att']}_norm{config['normalize']}_norminp{config['normalize_inputs']}"
            f"_pose{config['only_pose']}_diff{config['use_diff_images']}_blankref{config['use_blank_ref']}"
            f"_img{config['img_size_H']}x{config['img_size_W']}_bs{config['batch_size']}_lr{config['lr']}"
            f"_wd{config['weight_decay']}_dp{config['dropout']}_ep{config['epochs']}"
        )
        D_M_h_m = datetime.datetime.now().strftime("%d_%m_%H_%M")
        return exp_name_override if exp_name_override else f"{D_M_h_m}_{parameter_str}"

    exp_name = _build_exp_name()


    # --- WANDB INIT ---
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=exp_name,
            config=config
        )

        # --- SWEEP OVERRIDE: Apply sweep parameters to config ---
        # This allows WandB sweeps to override config values using dot notation
        # e.g., "image_backbone.fusion_method" -> config['model_params']['image_backbone']['fusion_method']
        sweep_config = dict(wandb.config)
        if sweep_config:
            print(f"Applying sweep overrides: {sweep_config}")
            for key, value in sweep_config.items():
                if key.startswith('_'):  # Skip wandb internal keys
                    continue

                # Handle nested keys with dot notation
                if '.' in key:
                    parts = key.split('.')
                    # Special handling: image_backbone.* -> model_params.image_backbone.*
                    if parts[0] == 'image_backbone':
                        d = config['model_params']['image_backbone']
                        d[parts[-1]] = value
                    elif parts[0] == 'pose_backbone':
                        d = config['model_params']['pose_backbone']
                        d[parts[-1]] = value
                    else:
                        # Generic nested navigation
                        d = config
                        for part in parts[:-1]:
                            d = d.setdefault(part, {})
                        d[parts[-1]] = value
                    print(f"  {key} = {value}")
                else:
                    # Top-level key (lr, dropout, use_lr_scheduler, etc.)
                    config[key] = value
                    print(f"  {key} = {value}")

        # CRITICAL: Re-read values that were passed as function args but may have been overridden
        # The sweep may override lr, dropout, weight_decay etc. which were passed as function args
        # Sweep/UI sources may provide numeric values as strings; normalize types.
        lr = float(config['lr'])
        dropout = float(config['dropout'])
        weight_decay = float(config['weight_decay'])
        epochs = int(config['epochs'])
        config['lr'] = lr
        config['dropout'] = dropout
        config['weight_decay'] = weight_decay
        config['epochs'] = epochs
        print(f"Using sweep values: lr={lr:.2e}, dropout={dropout}, weight_decay={weight_decay:.2e}, epochs={epochs}")

    # Resolve the L_weight ablation arm and bake it into config BEFORE the
    # config.yaml dump and WeightedLoss construction below: an explicit CLI flag
    # wins; else an existing config key; else True (current behavior).
    resolved_use_weighted_loss = (
        use_weighted_loss
        if use_weighted_loss is not None
        else config.get("use_weighted_loss", True)
    )
    apply_loss_weight_override(config, resolved_use_weighted_loss)
    if not resolved_use_weighted_loss:
        print("[use_weighted_loss] disabled -> w_edges=w_lengths=w_kb=0 (position MSE only)")

    # Recompute final experiment name after any sweep overrides.
    exp_name = _build_exp_name()
    print(f"Model save folder: {exp_name}")
    if use_wandb and wandb.run is not None:
        wandb.run.name = exp_name

    # build experiment folder by parameters and time

    setup_reproducibility(
        seed=seed,
        deterministic=deterministic,
        deterministic_warn_only=deterministic_warn_only,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check for session overlap between train and val
    check_no_session_overlap(train_data_root, val_data_root)

    print(f"Using device: {device}")
    out_path = Path(out_dir)
    (out_path / "checkpoints"/ exp_name).mkdir(parents=True, exist_ok=True)
    best_runs_mgr = BestRunsManager(out_path / "best_runs", max_entries=5)
    # save config as yaml for loading in evaluate.py
    config_save_path = out_path / "checkpoints" / exp_name / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to: {config_save_path}")
    # Dataset: section → frame view
    print("Loading dataset...")

    # if the the train_data_root is packed or not
    if "packed" in train_data_root:
        is_packed = True
    else:
        is_packed = False
    train_ds = DLOH5SampleDataset(  root_dir=train_data_root,
                                    norm_stats_path=norm_stats_path,
                                    blank_ref_root=blank_ref_root,
                                    normalize=normalize,
                                    normalize_inputs=normalize_inputs,
                                    use_diff_images=use_diff_images,
                                    use_blank_ref=use_blank_ref,
                                    img_size_H=config["img_size_H"],
                                    img_size_W=config["img_size_W"],
                                    transform=None,
                                    num_resample_points=num_resample_points,
                                    is_packed=is_packed,
                                    past_window_steps=config["past_window_steps"],
                                    max_time_delta=config["max_time_delta"],
                                    device=device
                                    )

    if "packed" in val_data_root:
        is_packed = True
    else:
        is_packed = False
    val_ds = DLOH5SampleDataset(root_dir=val_data_root,
                                    norm_stats_path=norm_stats_path,
                                    blank_ref_root=blank_ref_root,
                                    normalize=normalize,
                                    use_diff_images=use_diff_images,
                                    use_blank_ref=use_blank_ref,
                                    img_size_H=config["img_size_H"],
                                    img_size_W=config["img_size_W"],
                                    normalize_inputs=normalize_inputs,
                                    transform=None,
                                    num_resample_points=num_resample_points,
                                    is_packed=is_packed,
                                    past_window_steps=config["past_window_steps"],
                                    max_time_delta=config["max_time_delta"],
                                    device=device
                                        )


    # Count unique sessions in each dataset
    def count_sessions(dataset: DLOH5SampleDataset) -> int:
        """Count unique sessions (parent directories) in the dataset."""
        if dataset.is_packed:
            with h5py.File(dataset.h5_path, 'r') as f:
                ds = f['paths']
                paths = [p.decode('utf-8') for p in np.asarray(ds)]
        else:
            paths = [str(p) for p in dataset.files]
        sessions = set(extract_session(p) for p in paths)
        return len(sessions)

    train_sessions = count_sessions(train_ds)
    val_sessions = count_sessions(val_ds)

    # Sensor-drop augmentation: wrap ONLY train (val tracks the clean k=0 set).
    # Must come AFTER count_sessions, which reads train_ds.is_packed/.files.
    if drop_augment:
        from experiments.table2_sensor_ablation.drop_augment import DropAugmentDataset
        train_ds = DropAugmentDataset(train_ds)
        config["drop_augment"] = True
        print(f"[drop_augment] expanded train set to {len(train_ds)} samples (15x enumeration).")

    # Loaders
    print(f"Train samples: {len(train_ds)} ({train_sessions} sessions), Val samples: {len(val_ds)} ({val_sessions} sessions)")
    print("Creating data loaders...")
    # collate = frame_collate_pad_markers(n_markers)
    data_loader_time = time.time()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True,
                            persistent_workers=True ,
                            pin_memory=True,
                            prefetch_factor=2)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False,
                            persistent_workers=True ,
                            pin_memory=True,
                            prefetch_factor=2)

    print(f"Dataset loaded in {time.time() - data_loader_time:.2f} seconds.")
    # Model
    print("Creating model...")
    _model_cls = model_class if model_class is not None else DLOStateModel
    model = _model_cls(config).to(device)
    # model = torch.compile(model)

    # Weights-only fine-tune init: load a state_dict only (no optimizer/epoch
    # resume) so fine-tuning gets a fresh LR schedule. strict=False tolerates
    # head/shape mismatches between the source ckpt and this config.
    config["init_from"] = init_from
    if init_from is not None:
        state = torch.load(init_from, map_location=device)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[init_from] loaded {init_from} (strict=False): "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys"
        )
    log_file = out_path / "checkpoints" / exp_name / "training_log.txt"
    log_model_info(model, config=config, device=device, use_wandb=use_wandb, log_file=log_file,
                   train_sessions=train_sessions, val_sessions=val_sessions)

    # Optimizer with differential learning rates
    # Backbone (if unfrozen) gets lower LR, decoder/heads get full LR
    backbone_lr_factor = config['model_params']['image_backbone']['backbone_lr_factor']  # Default: backbone gets 10x lower LR
    backbone_params = []  # Initialize outside the if block for scheduler use

    if not only_pose and hasattr(model, 'img_enc'):
        # Separate backbone params from other params
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'img_enc.backbone' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        if backbone_params:
            print(f"Differential LR: backbone={lr * backbone_lr_factor:.2e}, decoder/heads={lr:.2e}")
            opt = torch.optim.AdamW([
                {'params': backbone_params, 'lr': lr * backbone_lr_factor},
                {'params': other_params, 'lr': lr}
            ], weight_decay=weight_decay)
        else:
            print("No backbone params to train, using standard optimizer")
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fc = WeightedLoss(w_nodes=config['w_nodes'],
        w_edges=config['w_edges'],
        w_lengths=config['w_lengths'],
        w_kb=config['w_kb'])
    # mse_loss = MSELoss()
    scaler = GradScaler(device="cuda" ,enabled=(device.type == "cuda"))
    steps_per_epoch = len(train_loader)

    # LR Scheduler (configurable)
    use_lr_scheduler = config['use_lr_scheduler']
    if use_lr_scheduler:
        sched = OneCycleLR(
            opt,
            max_lr=[lr * backbone_lr_factor, lr] if backbone_params else lr,  # Match param groups
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config['lr_warmup_pct'],  # 10% warmup default
            div_factor=config['lr_div_factor'],  # start lr = lr/10
            final_div_factor=config['lr_final_div_factor']  # end lr = lr/100
        )
        print(f"Using OneCycleLR scheduler: {config['lr_warmup_pct']*100:.0f}% warmup, div_factor={config['lr_div_factor']}")
    else:
        sched = None
        print("No LR scheduler (constant learning rate)")

    best_val = float("inf")

    # Per-epoch x per-session metric history for the detailed run_metrics.json.
    run_metrics_epochs: list[dict] = []
    run_metrics_best: dict = {}
    run_metrics_path = out_path / "checkpoints" / exp_name / "run_metrics.json"

    print("Starting training...")
    step_count = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False, dynamic_ncols=True)
        loss_count = 0

        for batch_idx, batch in enumerate(pbar):
            if max_batches_per_epoch > 0 and batch_idx >= max_batches_per_epoch:
                break

            step_count += 1
            if not only_pose:
                imgs = batch["digit_images"].to(device)          # [B,S,C,H,W]
                imgs = apply_active_sensors_mask(imgs, active_sensors, n_digits=4)
            else:
                imgs = None
            poses = batch["ee_pose"].to(device)                 # [B,7]
            tgt = batch["marker_positions"].to(device)      # [B,n_markers,3]

            # opt.zero_grad(set_to_none=True)
            opt.zero_grad()


            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                pred = model(poses, imgs)
                loss = config['chamfer_weight'] * chamfer_loss(pred,tgt) + config['mse_weight'] * loss_fc(pred,tgt)
                # loss = mse_loss(pred,tgt)

            # --- NaN DETECTION: Terminate run if loss becomes NaN ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[WARNING] NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}!")
                if use_wandb:
                    wandb.log({
                        "nan_detected": True,
                        "nan_epoch": epoch,
                        "nan_batch": batch_idx,
                    })
                    if wandb.run is not None:
                        wandb.run.summary["nan_detected"] = True
                        wandb.run.summary["termination_reason"] = "nan_loss"
                    wandb.finish(exit_code=1)
                raise ValueError(f"NaN loss detected at epoch {epoch}, batch {batch_idx}. Terminating run.")

            scaler.scale(loss).backward()

            # --- GRADIENT CLIPPING: Prevent exploding gradients ---
            max_grad_norm = config['max_grad_norm']  # 0 = disabled
            if max_grad_norm > 0:
                scaler.unscale_(opt)  # Unscale gradients for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            old_scale = scaler.get_scale() if device.type == "cuda" else None
            scaler.step(opt)
            scaler.update()

            if sched is not None:
                # With AMP, scheduler should step only if optimizer.step actually ran.
                if device.type != "cuda":
                    sched.step()
                else:
                    new_scale = scaler.get_scale()
                    optimizer_step_ran = (old_scale is not None) and (new_scale >= old_scale)
                    if optimizer_step_ran:
                        sched.step()

            running += loss.item() * tgt.size(0)
            loss_count += tgt.size(0)
            pbar.set_postfix({"loss": f"{(running / loss_count):.4f}"})

        train_loss = running / loss_count
        # --- WANDB LOG TRAIN ---
        current_loss = loss.item()
        current_lr = sched.get_last_lr()[0] if sched else lr
        if use_wandb:
            wandb.log({
                "train/loss_step": current_loss,
                "train/lr": current_lr,
                "epoch": epoch
            }, step=step_count)





        # ---- Val ----
        model.eval()
        # Accumulate denormalized (mm) per-sample points + paths across the val
        # loop, then reduce once via the shared eval_metrics.reduce_metrics so
        # train and the eval harnesses share one metric definition.
        val_pred_pts: list[np.ndarray] = []
        val_tgt_pts: list[np.ndarray] = []
        val_paths: list[str] = []
        all_l2: list[float] = []  # display-only running buffer for the live pbar
        all_chamfer: list[float] = []  # display-only running buffer for the live pbar

        if normalize:
            m_mean = train_ds.marker_mean.to(device) if hasattr(train_ds.marker_mean, "to") else torch.tensor(train_ds.marker_mean, device=device)
            m_std = train_ds.marker_std.to(device) if hasattr(train_ds.marker_std, "to") else torch.tensor(train_ds.marker_std, device=device)
        else:
            m_mean = m_std = None

        total_val = max_eval_batches if max_eval_batches > 0 else len(val_loader)
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            pbar_val = tqdm(val_loader, desc=f"Val {epoch}/{epochs}", total=total_val, leave=False, dynamic_ncols=True)
            for batch_idx, batch in enumerate(pbar_val):
                if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                    break
                poses = batch["ee_pose"].to(device)        # [B,7]
                tgt = batch["marker_positions"].to(device)        # [B,n_markers,3]
                paths = batch["path"]  # list of paths

                if not only_pose:
                    imgs = batch["digit_images"].to(device)          # [B,S,C,H,W]
                    imgs = apply_active_sensors_mask(imgs, active_sensors, n_digits=4)
                else:
                    imgs = None
                pred = model(poses, imgs)

                if normalize:
                    pred_unnorm = train_ds.denormalize_sample(pred.detach(), m_mean, m_std)
                    tgt_unnorm = train_ds.denormalize_sample(tgt.detach(), m_mean, m_std)
                else:
                    pred_unnorm = pred.detach()
                    tgt_unnorm = tgt.detach()

                val_pred_pts.append(pred_unnorm.float().cpu().numpy())
                val_tgt_pts.append(tgt_unnorm.float().cpu().numpy())
                if paths is not None:
                    val_paths.extend(list(paths))

                l2 = mean_point_distance_per_sample(pred_unnorm, tgt_unnorm).cpu().numpy()  # [B]
                chamfer = chamfer_distance_per_sample(pred_unnorm, tgt_unnorm).cpu().numpy()  # [B]
                all_l2.extend(l2.tolist())
                all_chamfer.extend(chamfer.tolist())
                pbar_val.set_postfix({"L2": f"{np.mean(all_l2):.4f}", "Chamfer": f"{np.mean(all_chamfer):.6f}"})

        if val_pred_pts:
            pred_all = np.concatenate(val_pred_pts, axis=0)
            tgt_all = np.concatenate(val_tgt_pts, axis=0)
        else:
            pred_all = np.zeros((0, 0, 3), dtype=np.float32)
            tgt_all = np.zeros((0, 0, 3), dtype=np.float32)
        epoch_metrics = reduce_metrics(pred_all, tgt_all, val_paths)

        # Re-derive the existing scalars from the reducer aggregate so the
        # best.pt selection criterion is byte-for-byte unchanged.
        agg = epoch_metrics["aggregate"]
        val_loss = agg["l2_mean"]
        val_loss_std = agg["l2_std"]
        val_chamfer_mean = agg["chamfer_mean"]
        val_chamfer_std = agg["chamfer_std"]

        # Map the reducer's per-session output back to the existing session_stats
        # shape (rmse_* labels) so console/log/wandb output stays unchanged.
        session_stats = {}
        for session in sorted(epoch_metrics["per_session"].keys()):
            s = epoch_metrics["per_session"][session]
            session_stats[session] = {
                'rmse_mean': s['l2_mean'],
                'rmse_std': s['l2_std'],
                'chamfer_mean': s['chamfer_mean'],
                'chamfer_std': s['chamfer_std'],
                'n_samples': s['n'],
            }

        # --- WANDB LOG VAL ---
        if use_wandb:
            log_dict = {
                "val/loss": val_loss,
                "val/loss_std": val_loss_std,
                "epoch": epoch
            }
            # Add per-session metrics
            for session, stats in session_stats.items():
                log_dict[f"val/session/{session}/rmse"] = stats['rmse_mean']
                log_dict[f"val/session/{session}/chamfer"] = stats['chamfer_mean']
            wandb.log(log_dict, step=step_count)

        print(f"epoch {epoch:03d}  train {train_loss:.4f}  val {val_loss:.4f} ± {val_loss_std:.4f}")

        # Print per-session breakdown
        if session_stats:
            print("  Per-session validation (RMSE / Chamfer):")
            for session, stats in session_stats.items():
                print(f"    {session}: RMSE={stats['rmse_mean']:.4f}±{stats['rmse_std']:.4f}, "
                    f"Chamfer={stats['chamfer_mean']:.6f}±{stats['chamfer_std']:.6f} (n={stats['n_samples']})")

        with open(out_path / "checkpoints" / exp_name / "training_log.txt", "a") as f:
            f.write(f"epoch {epoch:03d}  train {train_loss:.4f}  val {val_loss:.4f} ± {val_loss_std:.4f}\n")
            f.write(
                f"  Full-val stats: RMSE={val_loss:.4f}±{val_loss_std:.4f}, "
                f"Chamfer={val_chamfer_mean:.6f}±{val_chamfer_std:.6f}\n"
            )
            if session_stats:
                f.write("  Per-session validation (RMSE / Chamfer):\n")
                for session, stats in session_stats.items():
                    f.write(
                        f"    {session}: RMSE={stats['rmse_mean']:.4f}±{stats['rmse_std']:.4f}, "
                        f"Chamfer={stats['chamfer_mean']:.6f}±{stats['chamfer_std']:.6f} "
                        f"(n={stats['n_samples']})\n"
                    )

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            if use_wandb and wandb.run is not None:
                wandb.run.summary["best_val"] = best_val
            ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "img_size_H": img_size_H,
                    "img_size_W": img_size_W,
                    "n_digits": n_digits,
                    "n_markers": n_markers,
                    "with_att": with_att,
                },
            }

            torch.save(ckpt, out_path / "checkpoints" / exp_name / "best.pt")
            best_metrics = {
                "val_l2_mean": val_loss,
                "val_l2_std": val_loss_std,
                "val_chamfer_mean": val_chamfer_mean,
                "val_chamfer_std": val_chamfer_std,
            }
            with open(out_path / "checkpoints" / exp_name / "best_metrics.json", "w") as f:
                json.dump(best_metrics, f, indent=2)

            # Detailed-JSON "best" pointer tracks the same best-val-L2 epoch.
            run_metrics_best = {
                "epoch": epoch,
                "aggregate": epoch_metrics["aggregate"],
                "per_session": epoch_metrics["per_session"],
            }

            try:
                # Update global best-runs leaderboard
                best_runs_mgr.try_update(
                    exp_name=exp_name,
                    val_loss=float(val_loss),
                    epoch=epoch,
                    ckpt_src_dir=out_path / "checkpoints" / exp_name,
                )
            except Exception as e:
                print(f"Error updating best-runs leaderboard: {e}")
                pass

        # Detailed per-run metrics: append this epoch's full per-session reduction
        # and (re)write run_metrics.json at every epoch end so a crashed run still
        # leaves the learning curve up to the last completed epoch.
        run_metrics_epochs.append({"epoch": epoch, **epoch_metrics})
        run_metrics_doc = {
            "run": exp_name,
            "config": config,
            "epochs": run_metrics_epochs,
            "best": run_metrics_best,
        }
        with open(run_metrics_path, "w") as f:
            json.dump(run_metrics_doc, f, indent=2)

    # Final save
    torch.save(model.state_dict(), out_path / "checkpoints" / exp_name / "last_weights.pt")

    if use_wandb:
        wandb.finish()


@draccus.wrap()
def main(cfg: TrainConfig):
    """draccus entrypoint for the ``dlofeel-train`` console script.

    Translates the typed ``TrainConfig`` into the keyword arguments the ported
    ``train(...)`` loop expects: it resolves the ``active_sensors`` string and the
    ``model_type`` string exactly as the original argparse ``__main__`` block did,
    then hands off. The training logic in ``train()`` is untouched; this layer is
    the CLI shim only.
    """
    # Parse active_sensors: "0,1,2,3" or "fu,fd,ru,rd" -> list[int] (fu=0, fd=1, ru=2, rd=3; serials 22,18,23,24)
    _sensor_name_to_idx = {"fu": 0, "fd": 1, "ru": 2, "rd": 3}
    active_sensors = None
    if cfg.active_sensors:
        tokens = [t.strip().lower() for t in cfg.active_sensors.split(",") if t.strip()]
        active_sensors = []
        for t in tokens:
            if t in _sensor_name_to_idx:
                active_sensors.append(_sensor_name_to_idx[t])
            else:
                active_sensors.append(int(t))
        active_sensors = sorted(set(active_sensors))

    # Resolve model class from model_type. The direction model import is lazy so a
    # default dlo_state run never imports it (forward dep on Task 13).
    _MODEL_REGISTRY: dict[str, type] = {"dlo_state": DLOStateModel}
    if cfg.model_type == "direction":
        from dlofeel_state.models.direction_model import DLODirectionModel
        _MODEL_REGISTRY["direction"] = DLODirectionModel
    resolved_model_class = _MODEL_REGISTRY[cfg.model_type]

    train(
        train_data_root=cfg.train_data_root,
        val_data_root=cfg.val_data_root,
        blank_ref_root=cfg.blank_ref_root,
        config_path=cfg.train_config_path,
        norm_stats_path=cfg.norm_stats_path,
        num_resample_points=cfg.num_resample_points,
        normalize=cfg.normalize,
        normalize_inputs=cfg.normalize_inputs,
        with_att=cfg.with_att,
        use_diff_images=cfg.use_diff_images,
        use_blank_ref=cfg.use_blank_ref,
        only_pose=cfg.only_pose,
        single_encoder=cfg.single_encoder,
        image_backbone_type=cfg.image_backbone_type,
        use_spline=cfg.use_spline,
        n_control_points=cfg.n_control_points,
        out_dir=cfg.out_dir,
        img_size_H=cfg.img_size_H,
        img_size_W=cfg.img_size_W,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        dropout=cfg.dropout,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        use_wandb=cfg.use_wandb,
        max_batches_per_epoch=cfg.max_batches,
        max_eval_batches=cfg.max_eval_batches,
        active_sensors=active_sensors,
        exp_name_override=cfg.exp_name,
        deterministic=cfg.deterministic,
        deterministic_warn_only=cfg.deterministic_warn_only,
        model_class=resolved_model_class,
        drop_augment=cfg.drop_augment,
        use_weighted_loss=cfg.use_weighted_loss,
        init_from=cfg.init_from,
    )


if __name__ == "__main__":
    main()
