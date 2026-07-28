# pack_dataset_v2.py
"""
Optimized dataset packing script for DLOPredict v2.

Key features:
- Pre-processes data: transforms to fixed frame, resamples spline
- Compresses images with gzip by default (built into HDF5, so packed files are
  readable anywhere); optional LZ4 via --compression lz4 needs the hdf5plugin package
- Two-pass approach: scan for stats, then pack
- Computes and stores normalization stats in dataset config YAML
- Compatible with dataset_seq.py expected format

Output format (matches dataset_seq.py expectations for packed data):
- timestamp: [N] int64
- ee_position: [N, 3] float32 (pre-transformed to fixed frame)
- ee_orientation: [N, 4] float32 (pre-transformed to fixed frame)
- fixed_position: [N, 3] float32 (kept for reference)
- fixed_orientation: [N, 4] float32 (kept for reference)
- resampled_marker_positions: [N, num_points, 3] float32 (pre-transformed and resampled)
- digit_images: [N, 4, 3, H, W] uint8
- paths: [N] strings
- session_ids: [N] int32
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import hdf5plugin  # Registers the LZ4 filter, only needed for --compression lz4
except ImportError:
    hdf5plugin = None

import h5py
import numpy as np
import yaml
from tqdm import tqdm

# Import utilities
try:
    from dlofeel_state.data.spline_utils import fit_spline_unordered
    from dlofeel_state.data.transforms import get_ee_relative_to_fixed, get_markers_relative_to_fixed
except ImportError:
    print("Could not import utilities. Make sure you run from src/ or set PYTHONPATH.")
    exit(1)


def _decode_image_1d(buf: np.ndarray, height: int, width: int, step: int, encoding: str) -> np.ndarray:
    """Decodes raw buffer to (H, W, 3) uint8."""
    enc = encoding.lower()

    if enc == 'mono8':
        arr = buf.reshape((height, step))[:, :width]
        return np.repeat(arr[:, :, np.newaxis], 3, axis=2)

    if enc == 'rgb8':
        arr = buf.reshape((height, step))[:, :width * 3]
        return arr.reshape((height, width, 3))

    if enc == 'bgr8':
        arr = buf.reshape((height, step))[:, :width * 3].reshape((height, width, 3))
        # Convert BGR -> RGB
        return arr[..., ::-1].copy()

    raise ValueError(f"Encoding {encoding} not supported for packing.")


def read_sample(filepath: Path) -> dict:
    """Read a single sample from per-sample H5 file."""
    with h5py.File(filepath, 'r') as f:
        ts = f['timestamp'][()]
        ee_p = f['ee_position'][:].astype(np.float32)
        ee_q = f['ee_orientation'][:].astype(np.float32)
        fx_p = f['fixed_position'][:].astype(np.float32)
        fx_q = f['fixed_orientation'][:].astype(np.float32)
        mpos = f['marker_positions'][:].astype(np.float32)

        # Decode images
        imgs = []
        g = f['digit_images']
        for k in range(4):
            ds = g[f'img_{k}']
            raw = ds[:]
            h = int(ds.attrs['height'])
            w = int(ds.attrs['width'])
            step = int(ds.attrs['step'])
            enc = str(ds.attrs['encoding'])
            decoded = _decode_image_1d(raw, h, w, step, enc)  # [H, W, 3]
            imgs.append(decoded)

        imgs_stacked = np.stack(imgs, axis=0)  # [4, H, W, 3]

        return {
            'timestamp': ts,
            'digit_images': imgs_stacked,  # [4, H, W, 3] uint8
            'ee_position': ee_p,
            'ee_orientation': ee_q,
            'fixed_position': fx_p,
            'fixed_orientation': fx_q,
            'marker_positions': mpos,
            'path': str(filepath)
        }


def collect_files_by_session(source_root: Path) -> dict[str, list[Path]]:
    """Collect all H5 files grouped by session (parent folder)."""
    session_files = defaultdict(list)

    for dirpath, _, filenames in os.walk(source_root):
        for name in filenames:
            if name.endswith('.h5'):
                filepath = Path(dirpath) / name
                session_name = filepath.parent.name
                session_files[session_name].append(filepath)

    # Sort files within each session by sample index
    for session in session_files:
        session_files[session].sort(key=lambda p: int(p.stem.split('_')[1]))

    return dict(session_files)


def _scan_session_timestamps(args: tuple) -> dict:
    """Worker function to scan timestamps for a single session."""
    session_name, files = args

    timestamps = []
    for fp in files:
        with h5py.File(fp, 'r') as f:
            timestamps.append(f['timestamp'][()])

    # Compute inter-frame deltas
    deltas = []
    for i in range(1, len(timestamps)):
        delta_t = abs(timestamps[i] - timestamps[i-1]) / 1e9  # ns to seconds
        deltas.append(delta_t)

    session_stat = None
    if deltas:
        session_stat = {
            'num_samples': len(files),
            'duration_sec': float(timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0,
            'avg_fps': float(len(files) / ((timestamps[-1] - timestamps[0]) / 1e9)) if len(timestamps) > 1 and timestamps[-1] != timestamps[0] else 0,
            'delta_t_mean': float(np.mean(deltas)),
            'delta_t_std': float(np.std(deltas)),
        }

    return {
        'session_name': session_name,
        'deltas': deltas,
        'num_samples': len(files),
        'session_stat': session_stat,
    }


def compute_timestep_stats(session_files: dict[str, list[Path]],
                           num_workers: int = 8) -> dict:
    """
    Pass 1: Compute timestep statistics (parallelized).

    Returns:
        stats: Dictionary with timestep and session statistics
    """
    print(f"\n=== Pass 1: Computing timestep statistics ({num_workers} workers) ===")

    # Prepare args for parallel execution
    tasks = [(session_name, files) for session_name, files in session_files.items()]

    all_deltas = []
    total_samples = 0
    session_stats = {}

    # Parallel scan
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_scan_session_timestamps, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning sessions"):
            result = future.result()

            all_deltas.extend(result['deltas'])
            total_samples += result['num_samples']

            if result['session_stat']:
                session_stats[result['session_name']] = result['session_stat']

    # Global timestep stats
    all_deltas = np.array(all_deltas)
    timestep_stats = {
        'min_sec': float(np.min(all_deltas)) if len(all_deltas) > 0 else 0,
        'max_sec': float(np.max(all_deltas)) if len(all_deltas) > 0 else 0,
        'mean_sec': float(np.mean(all_deltas)) if len(all_deltas) > 0 else 0,
        'std_sec': float(np.std(all_deltas)) if len(all_deltas) > 0 else 0,
        'median_sec': float(np.median(all_deltas)) if len(all_deltas) > 0 else 0,
    }

    # Print summary
    print(f"\n--- Timestep Statistics ---")
    print(f"  Min:    {timestep_stats['min_sec']*1000:.2f} ms")
    print(f"  Max:    {timestep_stats['max_sec']*1000:.2f} ms")
    print(f"  Mean:   {timestep_stats['mean_sec']*1000:.2f} ms")
    print(f"  Median: {timestep_stats['median_sec']*1000:.2f} ms")
    print(f"  Std:    {timestep_stats['std_sec']*1000:.2f} ms")
    print(f"\n--- Sample Statistics ---")
    print(f"  Total samples: {total_samples}")
    print(f"  Sessions:      {len(session_files)}")

    return {
        'timestep_stats': timestep_stats,
        'total_samples': total_samples,
        'session_stats': session_stats,
    }


def pack_dataset(source_root: str,
                 output_path: str,
                 num_resample_points: int = 32,
                 spline_degree: int = 3,
                 num_workers: int = 8,
                 compression: str = "gzip"):
    """
    Pack dataset with pre-processed individual samples (compatible with dataset_seq.py).

    Pre-processing:
    - Transforms ee_position/ee_orientation to fixed frame
    - Transforms and resamples marker_positions using spline

    Output format matches what dataset_seq.py expects for packed data:
    - timestamp: [N] int64
    - ee_position: [N, 3] float32 (transformed to fixed frame)
    - ee_orientation: [N, 4] float32 (transformed to fixed frame)
    - fixed_position: [N, 3] float32 (original, for reference)
    - fixed_orientation: [N, 4] float32 (original, for reference)
    - resampled_marker_positions: [N, num_points, 3] float32 (transformed and resampled)
    - digit_images: [N, 4, 3, H, W] uint8
    - paths: [N] strings
    - session_ids: [N] int32
    """
    source_root = Path(source_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Packing dataset: {source_root}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  Resample points:    {num_resample_points}")
    print(f"  Spline degree:      {spline_degree}")
    print(f"  Workers:            {num_workers}")

    # Collect files by session
    print("\nCollecting files...")
    session_files = collect_files_by_session(source_root)
    total_files = sum(len(f) for f in session_files.values())
    print(f"Found {total_files} samples in {len(session_files)} sessions")

    if total_files == 0:
        print("ERROR: No samples found!")
        return None

    # Detect original image dimensions from first sample
    first_session = list(session_files.keys())[0]
    first_file = session_files[first_session][0]
    first_sample = read_sample(first_file)
    orig_img_H, orig_img_W = first_sample['digit_images'].shape[1:3]  # [4, H, W, 3]
    print(f"  Original img size:  {orig_img_H}x{orig_img_W}")

    # Pass 1: Compute timestep stats
    stats = compute_timestep_stats(session_files, num_workers=num_workers)

    n_samples = stats['total_samples']

    # Pass 2: Pack all samples with pre-processing
    print(f"\n=== Pass 2: Packing {n_samples} samples (with pre-processing) ===")

    # Accumulators for norm stats (computed on transformed data)
    img_sum = np.zeros(3, dtype=np.float64)
    img_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    # ee_pose: [pos(3) + ori(4)] = 7 dims
    ee_pose_sum = np.zeros(7, dtype=np.float64)
    ee_pose_sq_sum = np.zeros(7, dtype=np.float64)

    marker_sum = np.zeros((num_resample_points, 3), dtype=np.float64)
    marker_sq_sum = np.zeros((num_resample_points, 3), dtype=np.float64)

    total_processed = 0

    # Compression. gzip is the default because it ships inside HDF5: a packed file
    # then opens with a plain h5py install. LZ4 decompresses faster but writes the
    # 32004 filter, so reading it REQUIRES the hdf5plugin package; on real DIGIT
    # images it also compresses far worse (measured 1.07x vs gzip's 1.61x).
    if compression == "lz4":
        if hdf5plugin is None:
            raise RuntimeError(
                "--compression lz4 needs the hdf5plugin package (pip install hdf5plugin). "
                "Note that readers of the resulting file need it too."
            )
        compression_kwargs = hdf5plugin.LZ4()
    elif compression == "gzip":
        compression_kwargs = {'compression': 'gzip', 'compression_opts': 4}
    else:
        raise ValueError(f"unknown compression {compression!r}; use 'gzip' or 'lz4'")

    with h5py.File(output_path, "w") as f_out:
        # Create datasets - individual samples [N, ...]
        # digit_images: [N, 4, 3, H, W] uint8
        dset_imgs = f_out.create_dataset(
            "digit_images",
            (n_samples, 4, 3, orig_img_H, orig_img_W),
            dtype='uint8',
            chunks=(1, 4, 3, orig_img_H, orig_img_W),
            **compression_kwargs
        )

        # timestamp: [N] int64
        dset_timestamp = f_out.create_dataset(
            "timestamp",
            (n_samples,),
            dtype='int64'
        )

        # ee_position: [N, 3] float32 (transformed to fixed frame)
        dset_ee_pos = f_out.create_dataset(
            "ee_position",
            (n_samples, 3),
            dtype='float32'
        )

        # ee_orientation: [N, 4] float32 (transformed to fixed frame)
        dset_ee_ori = f_out.create_dataset(
            "ee_orientation",
            (n_samples, 4),
            dtype='float32'
        )

        # fixed_position: [N, 3] float32 (original, for reference)
        dset_fx_pos = f_out.create_dataset(
            "fixed_position",
            (n_samples, 3),
            dtype='float32'
        )

        # fixed_orientation: [N, 4] float32 (original, for reference)
        dset_fx_ori = f_out.create_dataset(
            "fixed_orientation",
            (n_samples, 4),
            dtype='float32'
        )

        # resampled_marker_positions: [N, num_points, 3] float32 (transformed and resampled)
        dset_markers = f_out.create_dataset(
            "resampled_marker_positions",
            (n_samples, num_resample_points, 3),
            dtype='float32'
        )

        # session_ids: [N] int32
        dset_session_ids = f_out.create_dataset(
            "session_ids",
            (n_samples,),
            dtype='int32'
        )

        # paths: [N] strings
        dt_str = h5py.special_dtype(vlen=str)
        dset_paths = f_out.create_dataset(
            "paths",
            (n_samples,),
            dtype=dt_str
        )

        # Map session names to IDs
        session_names = sorted(session_files.keys())
        session_to_id = {name: i for i, name in enumerate(session_names)}

        # Flatten all files in session order (maintains temporal ordering within sessions)
        all_files = []
        for session_name in session_names:
            for fp in session_files[session_name]:
                all_files.append((session_name, fp))

        # Process each sample
        for i, (session_name, filepath) in enumerate(tqdm(all_files, desc="Packing")):
            try:
                raw = read_sample(filepath)

                # Store timestamp (raw, int64)
                dset_timestamp[i] = raw['timestamp']

                # Transform ee_position and ee_orientation to fixed frame
                ee_pos_fixed, ee_ori_fixed = get_ee_relative_to_fixed(
                    raw['ee_position'],
                    raw['ee_orientation'],
                    raw['fixed_position'],
                    raw['fixed_orientation']
                )
                dset_ee_pos[i] = ee_pos_fixed.astype(np.float32)
                dset_ee_ori[i] = ee_ori_fixed.astype(np.float32)

                # Store original fixed frame pose (for reference)
                dset_fx_pos[i] = raw['fixed_position']
                dset_fx_ori[i] = raw['fixed_orientation']

                # Transform markers to fixed frame
                markers_fixed = get_markers_relative_to_fixed(
                    raw['marker_positions'],
                    raw['fixed_position'],
                    raw['fixed_orientation']
                )

                # Resample markers using spline
                resampled = fit_spline_unordered(
                    markers_fixed,
                    num_samples=num_resample_points,
                    k=spline_degree,
                    use_gpu=False
                )[0]
                dset_markers[i] = resampled.astype(np.float32)

                # Store images as [4, 3, H, W] uint8
                imgs_hwc = raw['digit_images']  # [4, H, W, 3]
                imgs_chw = np.transpose(imgs_hwc, (0, 3, 1, 2))  # [4, 3, H, W]
                dset_imgs[i] = imgs_chw

                # Store metadata
                dset_session_ids[i] = session_to_id[session_name]
                dset_paths[i] = raw['path']

                # Accumulate norm stats (on transformed data)
                imgs_float = imgs_chw.astype(np.float64) / 255.0  # [4, 3, H, W]
                img_sum += imgs_float.sum(axis=(0, 2, 3))  # Sum over digits, H, W
                img_sq_sum += (imgs_float ** 2).sum(axis=(0, 2, 3))
                total_pixels += 4 * orig_img_H * orig_img_W

                ee_pose = np.concatenate([ee_pos_fixed, ee_ori_fixed]).astype(np.float64)
                ee_pose_sum += ee_pose
                ee_pose_sq_sum += (ee_pose ** 2)

                marker_sum += resampled.astype(np.float64)
                marker_sq_sum += (resampled.astype(np.float64) ** 2)

                total_processed += 1

            except Exception as e:
                print(f"\nError packing sample {i} ({filepath}): {e}")
                continue

    # Compute final norm stats
    digit_mean = (img_sum / total_pixels).tolist()
    digit_var = (img_sq_sum / total_pixels) - np.array(digit_mean) ** 2
    digit_std = np.sqrt(np.maximum(digit_var, 0)).tolist()

    ee_pose_mean = (ee_pose_sum / total_processed).tolist()
    ee_pose_var = (ee_pose_sq_sum / total_processed) - np.array(ee_pose_mean) ** 2
    ee_pose_std = np.sqrt(np.maximum(ee_pose_var, 0)).tolist()

    marker_mean = (marker_sum / total_processed).tolist()
    marker_var = (marker_sq_sum / total_processed) - np.array(marker_mean) ** 2
    marker_std = np.sqrt(np.maximum(marker_var, 0)).tolist()

    # Build config dict
    config = {
        'version': 2,
        'created': datetime.now().isoformat(),
        'source': str(source_root),

        'packing_params': {
            'img_size_H': orig_img_H,
            'img_size_W': orig_img_W,
            'num_resample_points': num_resample_points,
            'spline_degree': spline_degree,
            'pre_processed': True,  # Indicates data is transformed and resampled
        },

        'timestep_stats': stats['timestep_stats'],
        'sample_stats': {
            'total_samples': total_processed,
            'sessions': len(session_files),
        },

        'norm_stats': {
            'digit_image_stats': {
                'mean': digit_mean,
                'std': digit_std,
            },
            'ee_position_stats': {
                'mean': ee_pose_mean,
                'std': ee_pose_std,
            },
            'per_marker_stats': {
                'mean': marker_mean,
                'std': marker_std,
            },
        },

        'sessions': {
            'names': session_names,
            'id_map': session_to_id,
            'stats': stats['session_stats'],
        },
    }

    # Save config YAML
    config_path = output_path.with_suffix('.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"Packing complete!")
    print(f"  H5 file:       {output_path}")
    print(f"  Config file:   {config_path}")
    print(f"  Total samples: {total_processed}")
    print(f"{'='*60}")

    # Print norm stats summary
    print(f"\n--- Normalization Statistics ---")
    print(f"  Digit images:")
    print(f"    Mean: [{digit_mean[0]:.4f}, {digit_mean[1]:.4f}, {digit_mean[2]:.4f}]")
    print(f"    Std:  [{digit_std[0]:.4f}, {digit_std[1]:.4f}, {digit_std[2]:.4f}]")
    print(f"  EE pose (pos[3] + ori[4]):")
    print(f"    Mean: {[f'{x:.4f}' for x in ee_pose_mean]}")
    print(f"    Std:  {[f'{x:.4f}' for x in ee_pose_std]}")
    print(f"  Markers ({num_resample_points} points):")
    print(f"    Mean shape: ({len(marker_mean)}, 3)")
    print(f"    Std shape:  ({len(marker_std)}, 3)")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack DLO dataset v2 (compatible with dataset_seq.py)")

    parser.add_argument("--src_train", type=str,
                        default="src/datasets/ds_h5/train",
                        help="Source train directory")
    parser.add_argument("--src_test", type=str,
                        default="src/datasets/ds_h5/test",
                        help="Source test directory")
    parser.add_argument("--out_dir", type=str,
                        default="src/datasets/packed_v2",
                        help="Output base directory")

    parser.add_argument("--num_points", type=int, default=32,
                        help="Number of resampled marker points")
    parser.add_argument("--spline_degree", type=int, default=3,
                        help="B-spline degree")
    parser.add_argument("--compression", type=str, default="gzip",
                        choices=["gzip", "lz4"],
                        help="Image compression. gzip (default) is built into HDF5 so the "
                             "packed file opens anywhere; lz4 is faster to read but needs "
                             "the hdf5plugin package at BOTH write and read time.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers for processing")

    args = parser.parse_args()

    # Generate output folder name based on parameters
    folder_name = f"pts{args.num_points}"
    out_folder = Path(args.out_dir) / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)

    out_train = out_folder / "train.h5"
    out_test = out_folder / "test.h5"

    print("\n" + "=" * 60)
    print("DLO Dataset Packer v2")
    print("=" * 60)
    print(f"Output folder: {out_folder}")

    # Pack train
    print("\n--- Packing TRAIN ---")
    pack_dataset(
        args.src_train, str(out_train),
        num_resample_points=args.num_points,
        spline_degree=args.spline_degree,
        num_workers=args.num_workers,
        compression=args.compression,
    )

    # Pack test
    print("\n--- Packing TEST ---")
    pack_dataset(
        args.src_test, str(out_test),
        num_resample_points=args.num_points,
        spline_degree=args.spline_degree,
        num_workers=args.num_workers,
        compression=args.compression,
    )

    print("\nDone!")
