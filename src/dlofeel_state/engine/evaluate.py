# evaluate.py
import json
from pathlib import Path
import draccus
import h5py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import plotly.graph_objects as go

from dlofeel_state.configs.eval import EvalConfig
from dlofeel_state.data.dataset import DLOH5SampleDataset
from dlofeel_state.models.dlo_state_model import DLOStateModel
from dlofeel_state.losses.chamfer import mean_point_distance_per_sample, chamfer_distance_per_sample
from dlofeel_state.engine.metrics import reduce_metrics

import numpy as np


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# save preds and gts as hdf5 file
def append_preds_gts_hdf5(h5_path, preds, gts, poses,sample_paths ,loss):
    with h5py.File(h5_path, "a") as f:
        # poses
        if "poses" not in f:
            maxshape = (None,) + poses.shape[1:]
            dset = f.create_dataset("poses", data=poses,
                                    maxshape=maxshape,
                                    chunks=True)
        else:
            dset = f["poses"]
            old = dset.shape[0]
            dset.resize(old + poses.shape[0], axis=0)
            dset[old:] = poses
        
        # preds
        if "preds" not in f:
            maxshape = (None,) + preds.shape[1:]
            dset = f.create_dataset("preds", data=preds,
                                    maxshape=maxshape,
                                    chunks=True)
        else:
            dset = f["preds"]
            old = dset.shape[0]
            dset.resize(old + preds.shape[0], axis=0)
            dset[old:] = preds

        # gts
        if "gts" not in f:
            maxshape = (None,) + gts.shape[1:]
            dset = f.create_dataset("gts", data=gts,
                                    maxshape=maxshape,
                                    chunks=True)
        else:
            dset = f["gts"]
            old = dset.shape[0]
            dset.resize(old + gts.shape[0], axis=0)
            dset[old:] = gts
        # sample paths
        if "sample_paths" not in f:
            dt = h5py.string_dtype(encoding='utf-8')
            maxshape = (None,)
            dset = f.create_dataset("sample_paths", data=sample_paths,
                                    maxshape=maxshape,
                                    chunks=True, dtype=dt)
        else:
            dset = f["sample_paths"]
            old = len(dset)
            dset.resize(old + len(sample_paths), axis=0)
            dset[old:] = sample_paths
        # loss
        if "losses" not in f:
            maxshape = (None,)
            dset = f.create_dataset("losses", data=loss,
                                    maxshape=maxshape,
                                    chunks=True)
        else:
            dset = f["losses"]
            old = len(dset)
            dset.resize(old + len(loss), axis=0)
            dset[old:] = loss
# ---------- Evaluate ----------
@torch.no_grad()
def evaluate(data_root, config:dict, ckpt_path, out_dir="results_eval",
             batch_size=None, low_rmse_threshold=0.055):

    torch.manual_seed(config.get("seed", 42) )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    exp_name = Path(ckpt_path).parent.name
    out_dir = out_dir / exp_name
    print(f"out_dir: {out_dir}")
    (out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print("Loading dataset...")

    # is packed dataset
    if "packed" in data_root:
        is_packed = True
    else:
        is_packed = False
    val_ds = DLOH5SampleDataset(  root_dir=data_root, 
                                    norm_stats_path=config["norm_stats_path"],
                                    normalize_inputs=config.get("normalize_inputs", True),
                                    normalize=config["normalize"],
                                    blank_ref_root=config["blank_ref_root"],
                                    use_diff_images=config["use_diff_images"],
                                    use_blank_ref=config["use_blank_ref"], 
                                    num_resample_points=config["num_output_pts"],
                                    img_size_H=config["img_size_H"],
                                    img_size_W=config["img_size_W"],
                                    transform=None,
                                    is_packed=is_packed,
                                    past_window_steps=config.get("past_window_steps", 1),
                                    max_time_delta=config.get("max_time_delta", 0.13),
                                    device=device
                                    )

    if batch_size is None:
        batch_size = config["batch_size"]
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=config.get("num_workers", 4), drop_last=True,
                            persistent_workers=True ,pin_memory=True, prefetch_factor=1)

    print(f"loaded {len(val_ds)} validation samples.")
    
    model = DLOStateModel(config=config).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    model.load_state_dict(ckpt["state_dict"] , strict=True)
    model.eval()
    
    normalize = config["normalize"]
    normalize_inputs = config["normalize_inputs"]
    print("Starting evaluation...")
    
    val_running = 0.0
    count = 0
    val_running_list = []
    all_chamfer = []
    # Accumulate denormalized (mm) per-sample points + paths for the shared
    # reduce_metrics pass (aggregate + per-session run_metrics.json).
    eval_pred_pts = []
    eval_tgt_pts = []
    eval_paths: list[str] = []
    low_rmse_samples = []

    only_pose = config.get("only_pose", False)

    for b_idx, batch in enumerate(loader):
        poses = batch["ee_pose"].to(device)        # [B,7]
        tgt = batch["marker_positions"].to(device)        # [B,n_markers,3]
        paths = batch["path"]  # list of paths
        
        if not only_pose:
            imgs = batch["digit_images"].to(device)          # [B,S,C,H,W]
        else:
            imgs = None
        
        pred = model(poses, imgs)
        # denormalize
        if normalize:
            pred_unnorm = val_ds.denormalize_sample(pred.cpu(), val_ds.marker_mean, val_ds.marker_std)
            tgt_unnorm = val_ds.denormalize_sample(tgt.cpu(), val_ds.marker_mean, val_ds.marker_std)
        else:
            pred_unnorm = pred
            tgt_unnorm = tgt
        
        if normalize_inputs:
            poses_unnorm = val_ds.denormalize_sample(poses.cpu(), val_ds.ee_mean, val_ds.ee_std)
            # norm the pose quaternion to unit norm
            poses_unnorm[:, 3:7] = poses_unnorm[:, 3:7] / torch.norm(poses_unnorm[:, 3:7], dim=1, keepdim=True)
        
        else:
            poses_unnorm = poses
        
        per_sample_loss = mean_point_distance_per_sample(pred_unnorm, tgt_unnorm)
        loss = per_sample_loss.mean()

        chamfer_ps = chamfer_distance_per_sample(pred_unnorm, tgt_unnorm)
        all_chamfer.extend(chamfer_ps.cpu().numpy().tolist())

        eval_pred_pts.append(pred_unnorm.float().cpu().numpy())
        eval_tgt_pts.append(tgt_unnorm.float().cpu().numpy())
        eval_paths.extend(list(paths))

        val_running_list.append(loss.item())
        if len(paths) == 1:
            print(f"Batch {b_idx}: RMSE {loss.item():.4f} | sample {paths[0]}")
        else:
            print(f"Batch {b_idx}: RMSE {loss.item():.4f}")

        for s_path, s_rmse in zip(paths, per_sample_loss.cpu().numpy().tolist()):
            if s_rmse < low_rmse_threshold:
                low_rmse_samples.append({"path": str(s_path), "rmse": s_rmse})

        # save preds and gts to hdf5
        append_preds_gts_hdf5(
            out_dir / "preds_gts.h5",
            pred_unnorm.cpu().numpy(),
            tgt_unnorm.cpu().numpy(),
            poses_unnorm.cpu().numpy(),
            paths,
            per_sample_loss.cpu().numpy(),
        )

    val_loss = np.mean(val_running_list)
    val_loss_std = np.std(val_running_list)
    print(f"Validation RMSE: {val_loss:.4f} ± {val_loss_std:.4f}")

    chamfer_mean = float(np.mean(all_chamfer)) if all_chamfer else float("nan")
    chamfer_std = float(np.std(all_chamfer)) if all_chamfer else float("nan")
    print(f"Validation Chamfer: {chamfer_mean:.6f} ± {chamfer_std:.6f}")

    # Detailed per-run metrics (aggregate + per-session). Single-pass eval, so
    # no per-epoch array (unlike train.py's run_metrics.json).
    if eval_pred_pts:
        pred_all = np.concatenate(eval_pred_pts, axis=0)
        tgt_all = np.concatenate(eval_tgt_pts, axis=0)
    else:
        pred_all = np.zeros((0, 0, 3), dtype=np.float32)
        tgt_all = np.zeros((0, 0, 3), dtype=np.float32)
    result = reduce_metrics(pred_all, tgt_all, eval_paths)
    run_metrics_doc = {
        "run": exp_name,
        "ckpt": str(ckpt_path),
        "config": config,
        **result,
    }
    with open(out_dir / "run_metrics.json", "w") as f:
        json.dump(run_metrics_doc, f, indent=2)
    print(f"Wrote per-run metrics to: {out_dir / 'run_metrics.json'}")

    low_rmse_samples.sort(key=lambda r: r["rmse"])
    low_rmse_doc = {
        "threshold": low_rmse_threshold,
        "n_below": len(low_rmse_samples),
        "n_total": len(eval_paths),
        "samples": low_rmse_samples,
    }
    with open(out_dir / "low_rmse_samples.json", "w") as f:
        json.dump(low_rmse_doc, f, indent=2)
    print(f"{len(low_rmse_samples)}/{len(eval_paths)} samples below RMSE "
          f"{low_rmse_threshold}; logged to {out_dir / 'low_rmse_samples.json'}")

@draccus.wrap()
def main(cfg: EvalConfig):
    """draccus entrypoint for the ``dlofeel-eval`` console script.

    Resolves the model-architecture config YAML, applies the run-level overrides
    from ``cfg``, and hands off to the parity-locked ``evaluate()``. The typed
    ``EvalConfig`` carries only run knobs; the model config stays as YAML (loaded
    from ``config_path`` or the checkpoint folder), the standard draccus split.
    """
    # Load the model config: explicit --model_config_path, else <ckpt_dir>/config.yaml,
    # mirroring the original evaluate.py __main__ auto-load + default fallback.
    if cfg.model_config_path is not None:
        config_path = Path(cfg.model_config_path)
    else:
        config_path = Path(cfg.ckpt_path).parent / "config.yaml"

    if config_path.exists():
        print(f"Loading config from: {config_path}")
        config = load_config(str(config_path))
    else:
        # Fallback: default config (for older checkpoints without config.yaml).
        print(f"Config not found at {config_path}, using default config")
        config = load_config("src/dlofeel_state/configs/default.yaml")

    # Override data paths for evaluation (as the original __main__ did).
    config["data_root"] = cfg.data_root
    config["blank_ref_root"] = cfg.blank_ref_root
    if cfg.norm_stats_path is not None:
        config["norm_stats_path"] = cfg.norm_stats_path

    # Eval loads a FULL checkpoint (strict state_dict load), so the Sparsh DINO
    # pretrained file is never needed and is not shipped in this repo.
    config["model_params"]["image_backbone"]["pretrained_weights_path"] = ""

    ckpt_dir = Path(cfg.ckpt_path).parent
    print(f"Evaluating experiment: {ckpt_dir.name}")
    print(f"Config: img_size={config.get('img_size_H')}x{config.get('img_size_W')}, "
          f"num_output_pts={config.get('num_output_pts')}, normalize={config.get('normalize')}")

    evaluate(
        data_root=cfg.data_root,
        config=config,
        ckpt_path=cfg.ckpt_path,
        out_dir=cfg.out_dir,
        batch_size=cfg.batch_size,
        low_rmse_threshold=cfg.low_rmse_threshold,
    )


if __name__ == "__main__":
    main()
