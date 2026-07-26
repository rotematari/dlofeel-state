# train_session_scaling.py
"""
Session-scaling experiment: measures how model performance scales with the
number of training sessions.

For each session count n (1 .. N_sessions), k random subsets of n sessions
are drawn, a fresh model is trained for a fixed number of epochs, and the
L2 + Chamfer loss on the *full* test set is recorded.  Results (per-trial
and aggregated mean/std per n) are saved incrementally (JSONL + CSV), with
checkpoint support for crash recovery via --resume.
"""

import os
os.environ["XFORMERS_DISABLED"] = "1"

import argparse
import csv
import gc
import datetime
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from dlofeel_state.data.dataset import DLOH5SampleDataset
from dlofeel_state.losses.chamfer import chamfer_loss, mean_point_distance_per_sample, chamfer_distance_per_sample
from dlofeel_state.losses.weighted import WeightedLoss
from dlofeel_state.models.dlo_state_model import DLOStateModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_sessions(root: Path) -> list[str]:
    """Return sorted list of session folder names (dirs that contain .h5 files)."""
    sessions: set[str] = set()
    for h5 in root.rglob("*.h5"):
        sessions.add(h5.parent.name)
    return sorted(sessions)


def group_indices_by_session(dataset: DLOH5SampleDataset) -> dict[str, list[int]]:
    """Map session name -> list of dataset indices for non-packed datasets."""
    session_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, f in enumerate(dataset.files):
        session_to_idx[f.parent.name].append(i)
    return dict(session_to_idx)


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    test_ds: DLOH5SampleDataset,
    device: torch.device,
    only_pose: bool,
    normalize: bool,
    max_eval_batches: int,
) -> dict[str, float]:
    """Evaluate current model on test loader and return L2/Chamfer stats."""
    model.eval()
    all_l2: list[float] = []
    all_chamfer: list[float] = []

    if normalize:
        m_mean = test_ds.marker_mean.to(device) if hasattr(test_ds.marker_mean, 'to') else torch.tensor(test_ds.marker_mean, device=device)
        m_std = test_ds.marker_std.to(device) if hasattr(test_ds.marker_std, 'to') else torch.tensor(test_ds.marker_std, device=device)

    total = max_eval_batches if max_eval_batches > 0 else len(test_loader)
    pbar = tqdm(test_loader, desc="  Eval", total=total, leave=False, dynamic_ncols=True)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
        for batch_idx, batch in enumerate(pbar):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
            poses = batch["ee_pose"].to(device)
            tgt = batch["marker_positions"].to(device)
            imgs = None if only_pose else batch["digit_images"].to(device)

            pred = model(poses, imgs)

            if normalize:
                pred_u = test_ds.denormalize_sample(pred, m_mean, m_std)
                tgt_u = test_ds.denormalize_sample(tgt, m_mean, m_std)
            else:
                pred_u = pred
                tgt_u = tgt

            l2 = mean_point_distance_per_sample(pred_u, tgt_u).cpu().numpy()
            chamfer = chamfer_distance_per_sample(pred_u, tgt_u).cpu().numpy()

            all_l2.extend(l2.tolist())
            all_chamfer.extend(chamfer.tolist())
            pbar.set_postfix({"L2": f"{np.mean(all_l2):.4f}", "Chamfer": f"{np.mean(all_chamfer):.6f}"})

    return {
        "l2_mean": float(np.mean(all_l2)),
        "l2_std": float(np.std(all_l2)),
        "chamfer_mean": float(np.mean(all_chamfer)),
        "chamfer_std": float(np.std(all_chamfer)),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def compute_summary(all_results: list[dict], max_n: int) -> dict[str, dict]:
    """Compute per-n_sessions aggregated statistics from trial results."""
    summary: dict[str, dict] = {}
    for n in range(1, max_n + 1):
        trials_n = [r for r in all_results if r["n_sessions"] == n]
        if not trials_n:
            continue
        final_l2_vals = [r["final_l2_mean"] for r in trials_n]
        final_chamfer_vals = [r["final_chamfer_mean"] for r in trials_n]
        best_l2_vals = [r["best_l2_mean"] for r in trials_n]
        best_chamfer_vals = [r["best_chamfer_mean"] for r in trials_n]
        summary[str(n)] = {
            "final_l2_mean_of_means": float(np.mean(final_l2_vals)),
            "final_l2_std_of_means": float(np.std(final_l2_vals)),
            "final_chamfer_mean_of_means": float(np.mean(final_chamfer_vals)),
            "final_chamfer_std_of_means": float(np.std(final_chamfer_vals)),
            "best_l2_mean_of_means": float(np.mean(best_l2_vals)),
            "best_l2_std_of_means": float(np.std(best_l2_vals)),
            "best_chamfer_mean_of_means": float(np.mean(best_chamfer_vals)),
            "best_chamfer_std_of_means": float(np.std(best_chamfer_vals)),
            "k_trials": len(trials_n),
        }
    return summary


def compute_grand_summary(all_results: list[dict]) -> dict[str, float]:
    """Aggregate statistics across *all* trials (all n_sessions values)."""
    if not all_results:
        return {}
    final_l2 = [r["final_l2_mean"] for r in all_results]
    final_chamfer = [r["final_chamfer_mean"] for r in all_results]
    best_l2 = [r["best_l2_mean"] for r in all_results]
    best_chamfer = [r["best_chamfer_mean"] for r in all_results]
    final_l2_stds = [r["final_l2_std"] for r in all_results]
    final_chamfer_stds = [r["final_chamfer_std"] for r in all_results]
    return {
        "n_total_trials": len(all_results),
        "final_l2_mean_of_means": float(np.mean(final_l2)),
        "final_l2_std_of_means": float(np.std(final_l2)),
        "final_l2_mean_of_stds": float(np.mean(final_l2_stds)),
        "final_chamfer_mean_of_means": float(np.mean(final_chamfer)),
        "final_chamfer_std_of_means": float(np.std(final_chamfer)),
        "final_chamfer_mean_of_stds": float(np.mean(final_chamfer_stds)),
        "best_l2_mean_of_means": float(np.mean(best_l2)),
        "best_l2_std_of_means": float(np.std(best_l2)),
        "best_chamfer_mean_of_means": float(np.mean(best_chamfer)),
        "best_chamfer_std_of_means": float(np.std(best_chamfer)),
    }


# ---------------------------------------------------------------------------
# CSV / JSONL writers
# ---------------------------------------------------------------------------

def write_trials_csv(csv_path: Path, trials: list[dict]) -> None:
    """Write per-trial results CSV."""
    fieldnames = [
        "n_sessions",
        "trial",
        "n_train_samples",
        "sessions",
        "final_l2_mean",
        "final_l2_std",
        "final_chamfer_mean",
        "final_chamfer_std",
        "best_epoch",
        "best_l2_mean",
        "best_l2_std",
        "best_chamfer_mean",
        "best_chamfer_std",
        "final_train_loss",
        "trial_time_s",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trials:
            writer.writerow({
                "n_sessions": row["n_sessions"],
                "trial": row["trial"],
                "n_train_samples": row["n_train_samples"],
                "sessions": ";".join(row["sessions"]),
                "final_l2_mean": row["final_l2_mean"],
                "final_l2_std": row["final_l2_std"],
                "final_chamfer_mean": row["final_chamfer_mean"],
                "final_chamfer_std": row["final_chamfer_std"],
                "best_epoch": row["best_epoch"],
                "best_l2_mean": row["best_l2_mean"],
                "best_l2_std": row["best_l2_std"],
                "best_chamfer_mean": row["best_chamfer_mean"],
                "best_chamfer_std": row["best_chamfer_std"],
                "final_train_loss": row["final_train_loss"],
                "trial_time_s": row["trial_time_s"],
            })


def write_summary_csv(csv_path: Path, summary: dict[str, dict]) -> None:
    """Write per-n aggregated summary CSV."""
    fieldnames = [
        "n_sessions",
        "k_trials",
        "final_l2_mean_of_means",
        "final_l2_std_of_means",
        "final_chamfer_mean_of_means",
        "final_chamfer_std_of_means",
        "best_l2_mean_of_means",
        "best_l2_std_of_means",
        "best_chamfer_mean_of_means",
        "best_chamfer_std_of_means",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for n_str, row in summary.items():
            writer.writerow({
                "n_sessions": int(n_str),
                "k_trials": row["k_trials"],
                "final_l2_mean_of_means": row["final_l2_mean_of_means"],
                "final_l2_std_of_means": row["final_l2_std_of_means"],
                "final_chamfer_mean_of_means": row["final_chamfer_mean_of_means"],
                "final_chamfer_std_of_means": row["final_chamfer_std_of_means"],
                "best_l2_mean_of_means": row["best_l2_mean_of_means"],
                "best_l2_std_of_means": row["best_l2_std_of_means"],
                "best_chamfer_mean_of_means": row["best_chamfer_mean_of_means"],
                "best_chamfer_std_of_means": row["best_chamfer_std_of_means"],
            })


def write_jsonl_line(jsonl_path: Path, record: dict) -> None:
    """Append a single JSON line to the JSONL log."""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def rewrite_jsonl(jsonl_path: Path, records: list[dict]) -> None:
    """Overwrite JSONL file with given records (used on resume to sync)."""
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_params(args: argparse.Namespace,
                       n_sessions: int, k_repeats: int,
                       epochs_per_trial: int) -> dict:
    """Extract key experiment parameters for checkpoint compatibility check."""
    return {
        "seed": args.seed,
        "only_pose": args.only_pose,
        "normalize": args.normalize,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "n_sessions": n_sessions,
        "k_repeats": k_repeats,
        "epochs_per_trial": epochs_per_trial,
        "num_resample_points": args.num_resample_points,
        "with_att": args.with_att,
        "image_backbone_type": args.image_backbone_type,
    }


def save_checkpoint(path: Path, trial_plan: list[dict], all_results: list[dict],
                    experiment_id: str, params: dict,
                    status: str = "in_progress") -> None:
    """Atomically save experiment checkpoint (write tmp then rename)."""
    payload = {
        "status": status,
        "experiment_id": experiment_id,
        "params": params,
        "trial_plan": trial_plan,
        "completed": all_results,
        "last_updated": datetime.datetime.now().isoformat(),
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.rename(path)


def load_checkpoint(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Single-trial training
# ---------------------------------------------------------------------------

_NAN_RESULT: dict[str, float] = {
    "final_l2_mean": float("nan"), "final_l2_std": float("nan"),
    "final_chamfer_mean": float("nan"), "final_chamfer_std": float("nan"),
    "best_epoch": -1,
    "best_l2_mean": float("nan"), "best_l2_std": float("nan"),
    "best_chamfer_mean": float("nan"), "best_chamfer_std": float("nan"),
    "final_train_loss": float("nan"),
}


def _cleanup_trial(device: torch.device, *objects) -> None:
    """Delete trial-scoped objects and free GPU memory."""
    for obj in objects:
        del obj
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_and_evaluate(
    config: dict,
    train_subset: Subset,
    test_loader: DataLoader,
    test_ds: DLOH5SampleDataset,
    device: torch.device,
    epochs: int,
    only_pose: bool,
    normalize: bool,
    max_batches_per_epoch: int,
    max_eval_batches: int,
) -> dict[str, float]:
    """Train a fresh model on *train_subset* for *epochs* epochs and evaluate
    on *test_loader*. Returns final-epoch and best-epoch metrics."""

    batch_size = config["batch_size"]
    lr: float = config["lr"]
    weight_decay: float = config["weight_decay"]

    # --- DataLoader for this trial's subset ---
    train_loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": config["num_workers"],
        "drop_last": True,
        "persistent_workers": False,
        "pin_memory": True,
    }
    if config["num_workers"] > 0:
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_subset, **train_loader_kwargs)

    # --- Fresh model ---
    model = DLOStateModel(config).to(device)

    # --- Optimizer (differential LR for backbone) ---
    backbone_lr_factor = config["model_params"]["image_backbone"]["backbone_lr_factor"]
    backbone_params: list[torch.nn.Parameter] = []

    if not only_pose and hasattr(model, "img_enc"):
        other_params: list[torch.nn.Parameter] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "img_enc.backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        if backbone_params:
            opt = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr * backbone_lr_factor},
                    {"params": other_params, "lr": lr},
                ],
                weight_decay=weight_decay,
            )
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fc = WeightedLoss(
        w_nodes=config["w_nodes"],
        w_edges=config["w_edges"],
        w_lengths=config["w_lengths"],
        w_kb=config["w_kb"],
    )
    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        _cleanup_trial(device, model, opt, loss_fc, scaler, train_loader)
        return dict(_NAN_RESULT)

    use_lr_scheduler = config["use_lr_scheduler"]
    sched = None
    if use_lr_scheduler:
        sched = OneCycleLR(
            opt,
            max_lr=[lr * backbone_lr_factor, lr] if backbone_params else lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=config["lr_warmup_pct"],
            div_factor=config["lr_div_factor"],
            final_div_factor=config["lr_final_div_factor"],
        )

    max_grad_norm: float = config["max_grad_norm"]
    best_epoch = -1
    best_metrics = {
        "l2_mean": float("inf"),
        "l2_std": float("nan"),
        "chamfer_mean": float("nan"),
        "chamfer_std": float("nan"),
    }
    final_eval_metrics = {
        "l2_mean": float("nan"),
        "l2_std": float("nan"),
        "chamfer_mean": float("nan"),
        "chamfer_std": float("nan"),
    }

    # ---- Training loop ----
    running: float = 0.0
    loss_count: int = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        loss_count = 0
        pbar = tqdm(train_loader, desc=f"  Train {epoch}/{epochs}", leave=False, dynamic_ncols=True)
        for batch_idx, batch in enumerate(pbar):
            if max_batches_per_epoch > 0 and batch_idx >= max_batches_per_epoch:
                break
            if not only_pose:
                imgs = batch["digit_images"].to(device)
            else:
                imgs = None
            poses = batch["ee_pose"].to(device)
            tgt = batch["marker_positions"].to(device)

            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                pred = model(poses, imgs)
                loss = (config["chamfer_weight"] * chamfer_loss(pred, tgt)
                        + config["mse_weight"] * loss_fc(pred, tgt))

            if torch.isnan(loss) or torch.isinf(loss):
                _cleanup_trial(device, model, opt, loss_fc, scaler, sched, train_loader)
                return dict(_NAN_RESULT)

            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()

            if sched is not None:
                sched.step()

            running += loss.item() * tgt.size(0)
            loss_count += tgt.size(0)
            pbar.set_postfix({"loss": f"{running / loss_count:.4f}"})

        final_eval_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            test_ds=test_ds,
            device=device,
            only_pose=only_pose,
            normalize=normalize,
            max_eval_batches=max_eval_batches,
        )
        if final_eval_metrics["l2_mean"] < best_metrics["l2_mean"]:
            best_metrics = dict(final_eval_metrics)
            best_epoch = epoch

    final_train_loss = running / loss_count if loss_count > 0 else float("nan")

    _cleanup_trial(device, model, opt, loss_fc, scaler, sched, train_loader)

    return {
        "final_l2_mean": final_eval_metrics["l2_mean"],
        "final_l2_std": final_eval_metrics["l2_std"],
        "final_chamfer_mean": final_eval_metrics["chamfer_mean"],
        "final_chamfer_std": final_eval_metrics["chamfer_std"],
        "best_epoch": best_epoch,
        "best_l2_mean": best_metrics["l2_mean"],
        "best_l2_std": best_metrics["l2_std"],
        "best_chamfer_mean": best_metrics["chamfer_mean"],
        "best_chamfer_std": best_metrics["chamfer_std"],
        "final_train_loss": float(final_train_loss),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    config = load_config(args.config_path)

    # Override config with CLI args (same pattern as train.py)
    config["num_output_pts"] = args.num_resample_points
    config["use_diff_images"] = args.use_diff_images
    config["use_blank_ref"] = args.use_blank_ref
    config["normalize"] = args.normalize
    config["normalize_inputs"] = args.normalize_inputs
    config["only_pose"] = args.only_pose
    config["with_att"] = args.with_att
    config["model_params"]["image_backbone"]["type"] = args.image_backbone_type
    config["model_params"]["image_backbone"]["single_encoder"] = args.single_encoder
    config["img_size_H"] = args.img_size_H
    config["img_size_W"] = args.img_size_W
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["weight_decay"] = args.weight_decay
    config["dropout"] = args.dropout
    effective_num_workers = args.num_workers
    effective_k_repeats = args.k_repeats
    effective_epochs_per_trial = args.epochs_per_trial
    effective_max_batches = args.max_batches
    effective_max_eval_batches = args.max_eval_batches
    run_mode = "full"
    if args.fast_test:
        run_mode = "fast_test"
        effective_epochs_per_trial = 1
        effective_k_repeats = 1
        effective_num_workers = min(args.num_workers, 2)
        if args.max_batches == 0:
            effective_max_batches = 20
        if args.max_eval_batches == 0:
            effective_max_eval_batches = 20
        if args.trials_per_n_override is not None:
            effective_k_repeats = max(1, args.trials_per_n_override)

    config["epochs"] = effective_epochs_per_trial
    config["norm_stats_path"] = args.norm_stats_path
    config["seed"] = args.seed
    config["num_workers"] = effective_num_workers

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_root = Path(args.train_data_root)
    val_root = Path(args.val_data_root)

    # ----------------------------------------------------------------
    # 1. Discover sessions
    # ----------------------------------------------------------------
    sessions = discover_sessions(train_root)
    n_sessions_total = len(sessions)
    default_max_n = min(3, n_sessions_total) if args.fast_test else n_sessions_total
    if args.max_sessions_n is None:
        max_sessions_n = default_max_n
    else:
        max_sessions_n = args.max_sessions_n if args.max_sessions_n > 0 else default_max_n
    n_sessions = min(n_sessions_total, max_sessions_n)
    if n_sessions < 1:
        raise ValueError("No training sessions found to run session-scaling experiment.")
    print(f"Found {n_sessions_total} training sessions: {sessions}")
    print(f"Evaluating n_sessions range: 1..{n_sessions}")
    print(f"Run mode: {run_mode} (epochs_per_trial={effective_epochs_per_trial}, "
          f"k_repeats={effective_k_repeats}, num_workers={effective_num_workers}, "
          f"max_batches={effective_max_batches}, max_eval_batches={effective_max_eval_batches})")

    # ----------------------------------------------------------------
    # 2. Build full train dataset (non-packed) + group indices
    # ----------------------------------------------------------------
    print("Loading full training dataset (non-packed) ...")
    full_train_ds = DLOH5SampleDataset(
        root_dir=str(train_root),
        blank_ref_root=args.blank_ref_root,
        is_packed=False,
        norm_stats_path=args.norm_stats_path,
        normalize=args.normalize,
        normalize_inputs=args.normalize_inputs,
        use_diff_images=args.use_diff_images,
        use_blank_ref=args.use_blank_ref,
        img_size_H=args.img_size_H,
        img_size_W=args.img_size_W,
        num_resample_points=args.num_resample_points,
        past_window_steps=config["past_window_steps"],
        max_time_delta=config["max_time_delta"],
        device=device,
    )
    session_to_indices = group_indices_by_session(full_train_ds)
    for s in sessions:
        print(f"  {s}: {len(session_to_indices.get(s, []))} samples")

    # ----------------------------------------------------------------
    # 3. Build test dataset + loader (fixed across all trials)
    # ----------------------------------------------------------------
    print("Loading test dataset ...")
    test_ds = DLOH5SampleDataset(
        root_dir=str(val_root),
        blank_ref_root=args.blank_ref_root,
        is_packed=False,
        norm_stats_path=args.norm_stats_path,
        normalize=args.normalize,
        normalize_inputs=args.normalize_inputs,
        use_diff_images=args.use_diff_images,
        use_blank_ref=args.use_blank_ref,
        img_size_H=args.img_size_H,
        img_size_W=args.img_size_W,
        num_resample_points=args.num_resample_points,
        past_window_steps=config["past_window_steps"],
        max_time_delta=config["max_time_delta"],
        device=device,
    )
    test_loader_kwargs: dict = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": effective_num_workers,
        "drop_last": False,
        "persistent_workers": effective_num_workers > 0,
        "pin_memory": True,
    }
    if effective_num_workers > 0:
        test_loader_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_ds, **test_loader_kwargs)
    print(f"Test samples: {len(test_ds)}")

    # ----------------------------------------------------------------
    # 4. Checkpoint resume or fresh trial plan
    # ----------------------------------------------------------------
    k_repeats = effective_k_repeats
    epochs_per_trial = effective_epochs_per_trial
    only_pose = args.only_pose
    normalize = args.normalize

    ckpt_params = _checkpoint_params(args, n_sessions, k_repeats, epochs_per_trial)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mode_tag = "pose_only" if only_pose else "full"
    file_tag = f"{mode_tag}_{'fasttest' if args.fast_test else 'full'}"

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume)
        if ckpt["status"] == "complete":
            print("Checkpoint already marked complete — nothing to do.")
            return
        saved_params = ckpt["params"]
        for key in ckpt_params:
            if key in saved_params and saved_params[key] != ckpt_params[key]:
                raise ValueError(
                    f"Checkpoint param mismatch: {key}={saved_params[key]} "
                    f"(checkpoint) vs {ckpt_params[key]} (current). "
                    f"Cannot resume with different experiment settings."
                )
        trial_plan: list[dict] = ckpt["trial_plan"]
        all_results: list[dict] = list(ckpt["completed"])
        start_idx = len(all_results)
        experiment_id: str = ckpt["experiment_id"]
        print(f"  Resuming from trial {start_idx}/{len(trial_plan)}")
    else:
        trial_plan = []
        for n in range(1, n_sessions + 1):
            for trial in range(k_repeats):
                chosen = random.sample(sessions, n)
                trial_plan.append({"n": n, "trial": trial, "sessions": chosen})
        all_results = []
        start_idx = 0
        experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    total_trials = len(trial_plan)
    checkpoint_file = out_path / f"checkpoint_{file_tag}_{experiment_id}.json"
    jsonl_file = out_path / f"session_scaling_log_{file_tag}_{experiment_id}.jsonl"
    trial_csv_file = out_path / f"session_scaling_trials_{file_tag}_{experiment_id}.csv"
    summary_csv_file = out_path / f"session_scaling_summary_{file_tag}_{experiment_id}.csv"
    out_file = out_path / f"session_scaling_{file_tag}_{experiment_id}.json"

    rewrite_jsonl(jsonl_file, all_results)
    save_checkpoint(checkpoint_file, trial_plan, all_results, experiment_id, ckpt_params)

    print(f"\n{'='*70}")
    print("SESSION SCALING EXPERIMENT")
    print(f"  Sessions:         {n_sessions}")
    print(f"  k_repeats:        {k_repeats}")
    print(f"  epochs_per_trial: {epochs_per_trial}")
    print(f"  only_pose:        {only_pose}")
    print(f"  Total trials:     {total_trials}")
    if start_idx > 0:
        print(f"  Resuming from:    trial {start_idx}")
    print(f"  Checkpoint:       {checkpoint_file}")
    print(f"  Log (JSONL):      {jsonl_file}")
    print(f"{'='*70}\n")

    experiment_start = time.time()
    prev_n = -1

    for plan_idx in range(start_idx, total_trials):
        entry = trial_plan[plan_idx]
        n = entry["n"]
        trial = entry["trial"]
        chosen = entry["sessions"]

        if n != prev_n:
            tag = " (resumed)" if plan_idx == start_idx and start_idx > 0 else ""
            print(f"\n--- n_sessions = {n}/{n_sessions}{tag} ---")
            prev_n = n

        indices = [i for s in chosen for i in session_to_indices.get(s, [])]
        n_samples = len(indices)
        train_subset = Subset(full_train_ds, indices)

        print(f"  Trial {trial+1}/{k_repeats}: sessions={chosen}  "
              f"({n_samples} samples)")

        trial_start = time.time()
        metrics = train_and_evaluate(
            config=config,
            train_subset=train_subset,
            test_loader=test_loader,
            test_ds=test_ds,
            device=device,
            epochs=epochs_per_trial,
            only_pose=only_pose,
            normalize=normalize,
            max_batches_per_epoch=effective_max_batches,
            max_eval_batches=effective_max_eval_batches,
        )
        trial_elapsed = time.time() - trial_start

        result = {
            "n_sessions": n,
            "trial": trial,
            "sessions": chosen,
            "n_train_samples": n_samples,
            "final_l2_mean": metrics["final_l2_mean"],
            "final_l2_std": metrics["final_l2_std"],
            "final_chamfer_mean": metrics["final_chamfer_mean"],
            "final_chamfer_std": metrics["final_chamfer_std"],
            "best_epoch": metrics["best_epoch"],
            "best_l2_mean": metrics["best_l2_mean"],
            "best_l2_std": metrics["best_l2_std"],
            "best_chamfer_mean": metrics["best_chamfer_mean"],
            "best_chamfer_std": metrics["best_chamfer_std"],
            "final_train_loss": metrics["final_train_loss"],
            "trial_time_s": round(trial_elapsed, 1),
        }
        all_results.append(result)

        # --- Incremental saves after every trial ---
        write_jsonl_line(jsonl_file, result)
        save_checkpoint(
            checkpoint_file, trial_plan, all_results,
            experiment_id, ckpt_params,
        )

        is_last_trial_for_n = (trial == k_repeats - 1)
        is_last_trial = (plan_idx == total_trials - 1)
        if is_last_trial_for_n or is_last_trial:
            write_trials_csv(trial_csv_file, all_results)
            partial_summary = compute_summary(all_results, n)
            write_summary_csv(summary_csv_file, partial_summary)

        print(f"    Final L2={metrics['final_l2_mean']:.4f}+-{metrics['final_l2_std']:.4f}  "
              f"Final Chamfer={metrics['final_chamfer_mean']:.6f}+-{metrics['final_chamfer_std']:.6f}  "
              f"Best@epoch {metrics['best_epoch']}: "
              f"L2={metrics['best_l2_mean']:.4f}, Chamfer={metrics['best_chamfer_mean']:.6f}  "
              f"({trial_elapsed:.0f}s)")

    total_elapsed = time.time() - experiment_start

    # ----------------------------------------------------------------
    # 5. Final aggregation
    # ----------------------------------------------------------------
    summary = compute_summary(all_results, n_sessions)
    grand = compute_grand_summary(all_results)

    # ----------------------------------------------------------------
    # 6. Save final results
    # ----------------------------------------------------------------
    payload = {
        "experiment": "session_scaling",
        "mode": mode_tag,
        "run_mode": run_mode,
        "n_sessions_total": n_sessions_total,
        "n_sessions_evaluated": n_sessions,
        "k_repeats": k_repeats,
        "epochs_per_trial": epochs_per_trial,
        "total_time_s": round(total_elapsed, 1),
        "config": {
            "lr": config["lr"],
            "batch_size": config["batch_size"],
            "weight_decay": config["weight_decay"],
            "dropout": config["dropout"],
            "num_resample_points": args.num_resample_points,
            "with_att": args.with_att,
            "only_pose": only_pose,
            "normalize": normalize,
            "normalize_inputs": args.normalize_inputs,
            "num_workers": config["num_workers"],
            "fast_test": args.fast_test,
            "max_batches": effective_max_batches,
            "max_eval_batches": effective_max_eval_batches,
        },
        "grand_summary": grand,
        "summary": summary,
        "trials": all_results,
    }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    write_trials_csv(trial_csv_file, all_results)
    write_summary_csv(summary_csv_file, summary)

    save_checkpoint(
        checkpoint_file, trial_plan, all_results,
        experiment_id, ckpt_params, status="complete",
    )

    print(f"\nResults saved to: {out_file}")
    print(f"Trials CSV saved to: {trial_csv_file}")
    print(f"Summary CSV saved to: {summary_csv_file}")
    print(f"Log (JSONL) saved to: {jsonl_file}")
    print(f"Checkpoint: {checkpoint_file} (status=complete)")

    # ----------------------------------------------------------------
    # 7. Print summary table (with ±std for paper reporting)
    # ----------------------------------------------------------------
    col = f"{'n':>4s}  {'Final L2':>10s} {'±std':>8s}  {'Final Cham.':>12s} {'±std':>10s}  {'Best L2':>10s} {'±std':>8s}  {'Best Cham.':>12s} {'±std':>10s}"
    sep = "-" * len(col)
    print(f"\n{sep}")
    print(f"SUMMARY  (mode={mode_tag}, {total_elapsed:.0f}s total)")
    print(sep)
    print(col)
    print(sep)
    for n_val in range(1, n_sessions + 1):
        s = summary.get(str(n_val))
        if s is None:
            continue
        print(f"{n_val:4d}  {s['final_l2_mean_of_means']:10.4f} {s['final_l2_std_of_means']:8.4f}  "
              f"{s['final_chamfer_mean_of_means']:12.6f} {s['final_chamfer_std_of_means']:10.6f}  "
              f"{s['best_l2_mean_of_means']:10.4f} {s['best_l2_std_of_means']:8.4f}  "
              f"{s['best_chamfer_mean_of_means']:12.6f} {s['best_chamfer_std_of_means']:10.6f}")
    print(sep)
    if grand:
        print(f"{'ALL':>4s}  {grand['final_l2_mean_of_means']:10.4f} {grand['final_l2_std_of_means']:8.4f}  "
              f"{grand['final_chamfer_mean_of_means']:12.6f} {grand['final_chamfer_std_of_means']:10.6f}  "
              f"{grand['best_l2_mean_of_means']:10.4f} {grand['best_l2_std_of_means']:8.4f}  "
              f"{grand['best_chamfer_mean_of_means']:12.6f} {grand['best_chamfer_std_of_means']:10.6f}")
        print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Session-scaling experiment: test how model performance "
                    "scales with the number of training sessions."
    )
    # Data paths
    p.add_argument("--train_data_root", type=str,
                   default="src/datasets/c1/ds/train",
                   help="Directory with session sub-dirs containing .h5 files")
    p.add_argument("--val_data_root", type=str,
                   default="src/datasets/c1/ds/test",
                   help="Directory with test session sub-dirs containing .h5 files")
    p.add_argument("--blank_ref_root", type=str,
                   default="src/datasets/ds/digit_blank_closed_refs",
                   help="Directory containing blank reference images")
    p.add_argument("--config_path", type=str,
                   default="src/dlofeel_state/configs/default.yaml")
    p.add_argument("--norm_stats_path", type=str,
                   default="src/dlofeel_state/configs/norm_stats_k16.yaml")
    p.add_argument("--out_dir", type=str, default="results/session_scaling",
                   help="Directory to save result JSON files")

    # Checkpoint resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint JSON file to resume an interrupted run")

    # Experiment parameters
    p.add_argument("--k_repeats", type=int, default=3,
                   help="Number of random trials per session count n")
    p.add_argument("--epochs_per_trial", type=int, default=10,
                   help="Training epochs per trial")
    p.add_argument("--fast_test", action="store_true",
                   help="Run quick sanity ablation with lightweight defaults")
    p.add_argument("--max_sessions_n", type=int, default=None,
                   help="Maximum n (session count) to evaluate; defaults to all, or 3 in --fast_test mode")
    p.add_argument("--trials_per_n_override", type=int, default=None,
                   help="Override k_repeats when --fast_test is enabled")

    # Model / data flags (mirror train.py)
    p.add_argument("--with_att", action="store_true")
    p.add_argument("--use_diff_images", action="store_true")
    p.add_argument("--use_blank_ref", action="store_true")
    p.add_argument("--only_pose", action="store_true",
                   help="Pose-only ablation (no tactile images)")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--normalize_inputs", action="store_true")
    p.add_argument("--single_encoder", action="store_true")
    p.add_argument("--image_backbone_type", type=str, default="sparsh_vit",
                   choices=["sparsh_vit", "resnet18"],
                   help="Image backbone type for ablation runs")
    p.add_argument("--num_resample_points", type=int, default=16,
                   help="Output marker count K. Must equal the per_marker_stats row "
                        "count of --norm_stats_path (the [K,3] GT is normalized against "
                        "it elementwise); the shipped stats file is K=16.")

    # Training hyper-parameters
    p.add_argument("--img_size_H", type=int, default=320)
    p.add_argument("--img_size_W", type=int, default=240)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_batches", type=int, default=0,
                   help="Max training batches per epoch (0 = all)")
    p.add_argument("--max_eval_batches", type=int, default=0,
                   help="Max evaluation batches per epoch (0 = all)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    print("=" * 60)
    print("SESSION SCALING EXPERIMENT ARGS")
    print("=" * 60)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 60)

    main(args)
