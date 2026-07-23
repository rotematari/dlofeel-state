# dataset.py
"""
PyTorch Dataset for per-sample HDF5 files produced by read_bag_bags_dirs_new.py.
Modified to return Temporal Pairs (Current + Past) for Sparsh.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import yaml
from PIL import Image
import time
from torch import Tensor
# import transforms 
from torchvision import transforms as T
from dlofeel_state.data.transforms import get_ee_relative_to_fixed, get_markers_relative_to_fixed, compute_all_stats
from dlofeel_state.data.spline_utils import fit_spline_unordered
import cv2
# -------------------------
# Image Parsing Helpers
# -------------------------

def _decode_image_1d(buf: np.ndarray, height: int, width: int, step: int, encoding: str) -> np.ndarray:
    """Convert flat uint8 buffer -> (H,W,C) or (H,W)."""
    enc = encoding.lower()
    if enc == 'mono8':
        arr = buf.reshape((height, step))[:, :width]
        return arr
    if enc == 'rgb8':
        arr = buf.reshape((height, step))[:, :width * 3]
        arr = arr.reshape((height, width, 3))
        return arr
    if enc == 'bgr8':
        arr = buf.reshape((height, step))[:, :width * 3].reshape((height, width, 3))
        if arr.shape != (320, 240, 3):
            raise ValueError(f"Unexpected BGR image shape: {arr.shape}")
        # Convert BGR -> RGB for consistency with pack_dataset_v2.py
        return arr[..., ::-1].copy()
    return buf

def _to_chw_float(img: np.ndarray) -> np.ndarray:
    """Convert image to CHW float32 in [0,1]."""
    if img.ndim == 1:
        c = 1; h = int(np.sqrt(img.size)); w = img.size // h
        return img.reshape(1, h, w).astype(np.float32) / 255.0
    if img.ndim == 2:
        return img[np.newaxis, :, :].astype(np.float32) / 255.0
    if img.ndim == 3:
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))
    raise ValueError(f"Unsupported image ndim: {img.ndim}")

# -------------------------
# HDF5 Reader Helper
# -------------------------

def _read_hdf5_sample_content(file_handle_or_path, is_packed: bool, idx: int = 0) -> Dict[str, Any]:
    """
    Reads a single raw sample from either a packed H5 handle or a single H5 file path.
    Returns dictionary with raw numpy arrays.
    """
    if is_packed:
        # file_handle_or_path is the open h5py.File object
        f = file_handle_or_path
        # Read directly from datasets
        ts = f['timestamp'][idx]
        ee_p = f['ee_position'][idx]
        ee_q = f['ee_orientation'][idx]
        fx_p = f['fixed_position'][idx]
        fx_q = f['fixed_orientation'][idx]
        # mpos = f['marker_positions'][idx]
        mpos = f['resampled_marker_positions'][idx]
        
        imgs_uint8 = f['digit_images'][idx] # (4, 3, H, W) presumably if packed, or flat? 
        # Note: If packed structure stores as (N, 4, ...) logic applies here.
        # Assuming packed stores pre-shaped or we reshape. 
        # Based on user code: f['digit_images'][idx] returns valid array.
        path_str = f['paths'][idx].decode('utf-8')
        
        return {
            'timestamp': ts,
            'digit_images': imgs_uint8, # uint8
            'ee_position': ee_p, 'ee_orientation': ee_q,
            'fixed_position': fx_p, 'fixed_orientation': fx_q,
            'marker_positions': mpos,
            'path': path_str
        }
    else:
        # file_handle_or_path is a Path object
        with h5py.File(file_handle_or_path, 'r') as f:
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
                try:
                    ds = g[f'img_{k}']
                except Exception as e:
                    print(f"Error accessing digit_{k} in {file_handle_or_path}: {e}")
                    raise e
                try:
                    raw = ds[:]
                except Exception as e:
                    print(f"Error reading raw data for digit_{k} in {file_handle_or_path}: {e}")
                    raise e
                try:
                    
                    h = int(ds.attrs['height'])
                    w = int(ds.attrs['width'])
                    step = int(ds.attrs['step'])
                    # print(f"Decoding digit_{k}: h={h}, w={w}, step={step}")
                    enc = str(ds.attrs['encoding'])
                except Exception as e:
                    print(f"Error reading attributes for digit_{k} in {file_handle_or_path}: {e}")
                    h = int(ds.attrs.get('height', 320))
                    w = int(ds.attrs.get('width', 240))
                    step = int(ds.attrs.get('step', 720))
                    enc = str(ds.attrs.get('encoding', 'bgr8'))
                    
                    
                decoded = _decode_image_1d(raw, h, w, step, enc)
                # Keep as HWC uint8 for consistency with packed loader, or CHW float
                # To match packed logic, let's return CHW float here directly or raw
                # converting to CHW Float immediately for ease
                imgs.append(_to_chw_float(decoded))
            
            # Stack to (4, C, H, W)
            imgs_stacked = np.stack(imgs, axis=0)
            
            return {
                'timestamp': ts,
                'digit_images': imgs_stacked, # float32 (0-1)
                'ee_position': ee_p, 'ee_orientation': ee_q,
                'fixed_position': fx_p, 'fixed_orientation': fx_q,
                'marker_positions': mpos,
                'path': str(file_handle_or_path)
            }

# -------------------------
# PyTorch Dataset
# -------------------------

class DLOH5SampleDataset(Dataset):
    def __init__(self, root_dir: str | Path,
                 blank_ref_root: str | Path= "",
                 is_packed: bool = True,
                 img_size_H: int = 320,
                 img_size_W: int = 240,
                 norm_stats_path: str | Path = "",
                 normalize: bool = False,
                 normalize_inputs: bool = False,
                 use_diff_images: bool = False,
                 use_blank_ref: bool = False,
                 transform: T.Compose | None = None,
                 
                 num_resample_points: int = 0,
                 # --- NEW ARGS FOR TEMPORAL PAIRING ---
                 past_window_steps: int = 5,       # Look back k steps (e.g. 5)
                 max_time_delta: float = 0.1,      # Max allowed time gap (seconds, assuming TS is ns check conversion)
                 device: torch.device | str | None = None):
        
        self.h5_path = Path(root_dir)
        self.is_packed = is_packed
        self.img_size_H = img_size_H
        self.img_size_W = img_size_W
        self.normalize = normalize
        self.normalize_inputs = normalize_inputs
        self.transform = transform
        self.use_diff_images = use_diff_images
        self.use_blank_ref = use_blank_ref
        self.num_resample_points = num_resample_points
        self.blank_reference_image_path = blank_ref_root
        self.device = torch.device(device) if device is not None else None
        self._device_in_worker_warned = False
        
        # Temporal settings
        self.past_window_steps = past_window_steps
        self.max_time_delta = max_time_delta 

        self.digit_serial_map = ['D21118', 'D21122', 'D21123', 'D21124']
        
        if self.is_packed:
            assert self.h5_path.suffix == '.h5', "For packed dataset, root_dir must be a .h5 file."
            self.main_ds_root = self.h5_path.parent 
            with h5py.File(self.h5_path, 'r') as f:
                self.length = len(f['paths'])
        else:
            self.main_ds_root = self.h5_path.parent
            self.files: List[Path] = []
            # followlinks=True so the combined-cable symlink-union dir (session dirs
            # are symlinks into c1/c2/c3) is descended into; real-dir roots (c1/c2/c3)
            # are unaffected since their session dirs are not symlinks.
            for dirpath, _, filenames in os.walk(self.h5_path, followlinks=True):
                for name in filenames:
                    if name.endswith('.h5'):
                        self.files.append(Path(dirpath) / name)
            # Sort ensures temporal order
            self.files.sort(key=lambda p: (p.parent.name, int(p.stem.split('_')[1])))
            self.length = len(self.files)

        self.load_normalization_stats(norm_stats_path)
        self.blank_refs = self.get_blank_refs()
        self.ref_cache = {}
        self.archive = None 

    def load_normalization_stats(self, path: str | Path):
        with open(path, "r") as f:
            stats = yaml.safe_load(f)
        # Keep stats on CPU - DataLoader workers run on CPU, GPU transfer happens after batching
        self.marker_mean = torch.tensor(stats["per_marker_stats"]["mean"])
        self.marker_std = torch.tensor(stats["per_marker_stats"]["std"])
        self.ee_mean = torch.tensor(stats["ee_position_stats"]["mean"])
        self.ee_std = torch.tensor(stats["ee_position_stats"]["std"])
        
        digit_stats = stats["digit_image_stats"]
            
        self.digit_mean = torch.tensor(digit_stats["mean"])
        self.digit_std = torch.tensor(digit_stats["std"])

    def normalize_sample(self, P, mean, std):
        # Move mean/std to same device as P for GPU compatibility
        return (P - mean) / std
    
    def denormalize_sample(self, P_norm, mean, std):
        # Move mean/std to same device as P_norm for GPU compatibility
        return P_norm * std + mean

    def get_blank_refs(self) -> np.ndarray:
        blank_refs = []
        for k in range(4):
            ref_path = Path(self.blank_reference_image_path) / f'{self.digit_serial_map[k]}.png'
            img = Image.open(ref_path).convert('RGB')
            
            if self.is_packed:
                hwc = cv2.resize(np.array(img), (self.img_size_W, self.img_size_H), interpolation=cv2.INTER_LINEAR)
            else:
                hwc = np.array(img)
            chw = _to_chw_float(hwc)

            blank_refs.append(chw)
            # to torch tensor 
        
        # blank_refs = torch.stack([torch.tensor(r) for r in blank_refs])
        blank_refs = np.stack(blank_refs, axis=0)  # (4, 3, H, W)
        return blank_refs

    def get_digit_refs(self, session_dir_name: str) -> List[np.ndarray]:
        if session_dir_name in self.ref_cache:
            return self.ref_cache[session_dir_name]

        ref_file = self.main_ds_root / "digit_refs" / session_dir_name / "digit_refs.h5"
        if not ref_file.exists():
             # Fallback logic
             ref_file = Path("src/datasets/ds_h5") / "digit_refs" / session_dir_name / "digit_refs.h5"

        with h5py.File(ref_file, 'r') as f:
            digit_refs_group = f['digit_refs']
            digit_refs = []
            for k in range(4):
                d_key = f'digit_{k+1}'
                if d_key not in digit_refs_group:
                     digit_refs.append(np.zeros((3,320,240), dtype=np.float32))
                     continue
                ds = digit_refs_group[d_key]
                ref_img = ds[:]
                h = int(ds.attrs.get('height', 320))
                w = int(ds.attrs.get('width', 240))
                step = int(ds.attrs.get('step', max(w, 1)))
                enc = str(ds.attrs.get('encoding', ''))
                decoded = _decode_image_1d(ref_img, h, w, step, enc)
                digit_refs.append(_to_chw_float(decoded))

        self.ref_cache[session_dir_name] = digit_refs
        return digit_refs

    def _get_archive(self):
        if self.archive is None:
            self.archive = h5py.File(self.h5_path, 'r', swmr=True)
        return self.archive

    def __len__(self):
        return self.length
    
    def _get_item_device(self) -> torch.device | None:
        if self.device is None:
            return None
        if self.device.type == "cuda" and get_worker_info() is not None:
            if not self._device_in_worker_warned:
                print("[dataset_seq] device transfer skipped in worker process; set num_workers=0 to enable.")
                self._device_in_worker_warned = True
            return None
        return self.device

    def _process_images(self, raw_imgs, path_str):
        """Standardizes raw images to Float Tensor and subtracts background."""

        # make sure image size is correct
        

        # 2. Subtract Background
        
        if self.use_blank_ref:
            imgs = raw_imgs - self.blank_refs
        else:
            session_name = Path(path_str).parent.name
            refs = self.get_digit_refs(session_name)
            imgs = raw_imgs - np.stack(refs, axis=0)
        return imgs

    def __getitem__(self, idx):

        t_start = time.perf_counter()
        # --------------------------------------------------------
        # 1. FETCH CURRENT SAMPLE
        # --------------------------------------------------------
        if self.is_packed:
            f = self._get_archive()
            raw_curr = _read_hdf5_sample_content(f, is_packed=True, idx=idx)
        else:
            raw_curr = _read_hdf5_sample_content(self.files[idx], is_packed=False)
        
        t_fetch_curr_end = time.perf_counter()

        # --------------------------------------------------------
        # 2. FETCH PAST SAMPLE (The Logic for "Good Pair")
        # --------------------------------------------------------
        prev_idx = idx - self.past_window_steps
        raw_prev = None
        
        # Conditions for a valid pair:
        # A. Index is within bounds
        # B. Same Session (check path parents)
        # C. Time Delta is small enough
        
        if prev_idx >= 0:
            if self.is_packed:
                raw_cand = _read_hdf5_sample_content(f, is_packed=True, idx=prev_idx)
            else:
                raw_cand = _read_hdf5_sample_content(self.files[prev_idx], is_packed=False)

            # Check B: Same Session
            path_curr = Path(raw_curr['path'])
            path_prev = Path(raw_cand['path'])
            
            # Usually strict equality of parent folder is enough for "Same Session"
            is_same_session = (path_curr.parent.name == path_prev.parent.name)

            # Check C: Time Delta
            # Assuming timestamps are comparable. If one is ns and one is s, this fails. 
            # Usually ROS bags use nanoseconds (int64). T=0.1s -> 100,000,000 ns.
            # ADJUST THIS SCALAR BASED ON YOUR DATA (1e9 for ns, 1 for s)
            TS_SCALE = 1e9 
            delta_t = abs(raw_curr['timestamp'] - raw_cand['timestamp']) / TS_SCALE
            
            is_time_good = (delta_t <= self.max_time_delta)

            if is_same_session and is_time_good:
                raw_prev = raw_cand

        # Fallback: If no valid past, duplicate current (Velocity = 0)
        if raw_prev is None:
            raw_prev = raw_curr

        # Stack logic: [4, 3, H, W] -> [4, 6, H, W]
        # Concatenate along channel dim (dim=1)
        
        imgs_curr = torch.from_numpy(raw_curr['digit_images']).float()
        imgs_prev = torch.from_numpy(raw_prev['digit_images']).float()
        
        
        # get ref 
        if self.use_diff_images:
            if self.use_blank_ref:
                refs = torch.from_numpy(self.blank_refs).float()
            else:
                session_name = Path(raw_curr['path']).parent.name
                ref_imgs = self.get_digit_refs(session_name)
                refs = torch.from_numpy(np.stack(ref_imgs, axis=0)).float()
            if refs.shape[2] != self.img_size_H or refs.shape[3] != self.img_size_W:
                refs = T.Resize((self.img_size_H, self.img_size_W))(refs)
        
        # make sure image size is correct
        if imgs_curr.shape[2] != self.img_size_H or imgs_curr.shape[3] != self.img_size_W:
            imgs_curr = T.Resize((self.img_size_H, self.img_size_W))(imgs_curr)
        if imgs_prev.shape[2] != self.img_size_H or imgs_prev.shape[3] != self.img_size_W:
            imgs_prev = T.Resize((self.img_size_H, self.img_size_W))(imgs_prev)
        

        # imgs_combined = torch.cat([imgs_curr, imgs_prev], dim=1)
        
        t_transform_end = time.perf_counter()
        
        # --------------------------------------------------------
        # 5. POSE & MARKERS (Using Current Frame Info)
        # --------------------------------------------------------
        ee_pos = raw_curr['ee_position']
        ee_ori = raw_curr['ee_orientation']
        fx_pos = raw_curr['fixed_position']
        fx_ori = raw_curr['fixed_orientation']
        mpos   = raw_curr['marker_positions']

        if not self.is_packed:
            ee_pos, ee_ori = get_ee_relative_to_fixed(ee_pos, ee_ori, fx_pos, fx_ori)
            mpos = get_markers_relative_to_fixed(mpos, fx_pos, fx_ori)
            # Resample the raw GT markers to K points iff K is requested AND differs
            # from the raw marker count. Two invariants preserved from the old `>14` guard:
            #   - num_resample_points == 0 is the no-resample sentinel (markers kept raw).
            #   - K == raw count is a no-op (avoid a redundant spline fit + float drift);
            #     K == 32 on 14 raw markers still resamples, exactly as before.
            # mpos.shape[0] is read dynamically so this is robust to a non-14 raw count.
            if self.num_resample_points and self.num_resample_points != mpos.shape[0]:
                mpos = fit_spline_unordered(mpos, num_samples=self.num_resample_points, smoothing=0, k=5)[0]
            
        
        ee_pos_fixed = torch.concat([torch.from_numpy(ee_pos).float(), torch.from_numpy(ee_ori).float()], dim=-1)
        
        # mpos = torch.from_numpy(mpos)
        t_pose_markers_end = time.perf_counter()

        # --------------------------------------------------------
        # 6. NORMALIZATION
        # --------------------------------------------------------
        if self.normalize_inputs:
            ee_pos_fixed = self.normalize_sample(ee_pos_fixed, self.ee_mean, self.ee_std)
            # Expand digit mean/std to 6 channels for normalization
            # Current: (3,) -> Need (6,)
            imgs_curr = self.normalize_sample(imgs_curr, self.digit_mean.view(1,3,1,1), self.digit_std.view(1,3,1,1))
            imgs_prev = self.normalize_sample(imgs_prev, self.digit_mean.view(1,3,1,1), self.digit_std.view(1,3,1,1))
            if self.use_diff_images:
                refs = self.normalize_sample(refs, self.digit_mean.view(1,3,1,1), self.digit_std.view(1,3,1,1))

        if self.use_diff_images:
            imgs_curr = imgs_curr - refs
            imgs_prev = imgs_prev - refs
        
        imgs_combined = torch.cat([imgs_curr, imgs_prev], dim=1)  # [4, 6, H, W]
        
        
        mpos = torch.from_numpy(mpos).float()
        
        if self.normalize:
            mpos = self.normalize_sample(mpos, self.marker_mean, self.marker_std)
        t_normalize_end = time.perf_counter()

        t_device_end = t_normalize_end



        
        
        # t_fetch_curr = t_fetch_curr_end - t_start
        # t_fetch_prev = t_fetch_prev_end - t_fetch_curr_end
        # t_process_images = t_process_images_end - t_fetch_prev_end
        # t_transform = t_transform_end - t_process_images_end
        # t_pose_markers = t_pose_markers_end - t_transform_end
        # t_normalize = t_normalize_end - t_pose_markers_end
        # t_device = t_device_end - t_normalize_end
        # t_total = t_device_end - t_start
        # print(
        #     f"[dataset_seq] idx={idx} "
        #     f"fetch_curr={t_fetch_curr*1e3:.2f}ms "
        #     f"fetch_prev={t_fetch_prev*1e3:.2f}ms "
        #     f"process_images={t_process_images*1e3:.2f}ms "
        #     f"transform={t_transform*1e3:.2f}ms "
        #     f"pose_markers={t_pose_markers*1e3:.2f}ms "
        #     f"normalize={t_normalize*1e3:.2f}ms "
        #     f"to_device={t_device*1e3:.2f}ms "
        #     f"total={t_total*1e3:.2f}ms"
        # )

        

        return {
            'digit_images': imgs_combined, # [4, 6, 320, 240]
            "ee_pose": ee_pos_fixed,
            'marker_positions': mpos,
            'path': raw_curr['path'],
        }

if __name__ == "__main__":

    # --- CONFIGURATION ---
    # Update these paths if your local setup changes
    data_path = Path("src/datasets/ds_h5/train")
    # data_path = Path("src/datasets/packed/train/train_packed.h5") 
    stats_path = Path("src/dlofeel_state/configs/norm_stats.yaml")
    blank_ref_path = "src/datasets/ds_h5/digit_blank_closed_refs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Dataset (Non-Packed for easier debugging usually, or Packed if you have the file)
    # We set past_window_steps=5 (~80ms if 60fps)
    dataset = DLOH5SampleDataset(
        root_dir=data_path,
        is_packed=False, # Set to True if testing the big .h5 file
        norm_stats_path=stats_path,
        blank_ref_root=blank_ref_path,
        use_blank_ref=False,
        use_diff_images=False,
        normalize=False,
        normalize_inputs=False, # Keep false to see raw pixels in plot
        past_window_steps =1,    # Gap between current and past frame
        max_time_delta=0.13,      # 500ms max gap
        num_resample_points=32,
        
        
    )

    
    
    with_images = False
    stats = compute_all_stats(  dataset,
                                num_workers=10,
                                num_out_samples=32,
                                with_images=with_images)

    # print("--- YAML READY FORMAT ---")
    if with_images:
        print("digit_image_stats:")
        print(f"  mean: {stats['digit_mean']}")
        print(f"  std:  {stats['digit_std']}")
    print("ee_position_stats:")
    print(f"  mean: {stats['ee_mean']}") # Slice [:3] manually if you only want Pos
    print(f"  std:  {stats['ee_std']}")
    print("per_marker_stats:")
    print(f"  mean: {stats['marker_mean']}")
    print(f"  std:  {stats['marker_std']}")


  