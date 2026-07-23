# import numpy as np
from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
from typing import Tuple  

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import h5py


def check_no_session_overlap(train_root: str, val_root: str) -> None:
    """
    Verify that there is no overlap of sessions between train and val datasets.
    Sessions are identified by the parent directory name of each sample file.
    
    Args:
        train_root: Path to training data directory
        val_root: Path to validation data directory
        
    Raises:
        ValueError: If overlapping sessions are found
    """
    train_path = Path(train_root)
    val_path = Path(val_root)
    
    # Collect session names from train
    train_sessions = set()
    for h5_file in train_path.rglob("*.h5"):
        session_name = h5_file.parent.name
        train_sessions.add(session_name)
    
    # Collect session names from val
    val_sessions = set()
    for h5_file in val_path.rglob("*.h5"):
        session_name = h5_file.parent.name
        val_sessions.add(session_name)
    
    # Check for overlap
    overlap = train_sessions & val_sessions
    
    if overlap:
        raise ValueError(
            f"Session overlap detected between train and val datasets!\n"
            f"Overlapping sessions ({len(overlap)}): {sorted(overlap)}\n"
            f"Train sessions ({len(train_sessions)}): {sorted(train_sessions)}\n"
            f"Val sessions ({len(val_sessions)}): {sorted(val_sessions)}"
        )
    
    print(f"✓ No session overlap: train has {len(train_sessions)} sessions, val has {len(val_sessions)} sessions")

def get_ee_relative_to_fixed(ee_pos, ee_quat, fixed_pos, fixed_quat):
    """
    Get EE pose relative to fixed frame
    ee_pos: [3] position in base frame
    ee_quat: [4] quaternion (x,y,z,w) in base frame  
    fixed_pos: [3] position in base frame
    fixed_quat: [4] quaternion (x,y,z,w) in base frame
    """

    # Convert quaternions to rotation matrices
    R_base_to_ee = Rotation.from_quat(ee_quat).as_matrix()
    R_base_to_fixed = Rotation.from_quat(fixed_quat).as_matrix()
    
    # Inverse of fixed transform
    R_fixed_to_base = R_base_to_fixed.T
    # t_fixed_to_base = -R_fixed_to_base @ fixed_pos
    
    # Compose: fixed->base->ee
    R_fixed_to_ee = R_fixed_to_base @ R_base_to_ee
    t_fixed_to_ee = R_fixed_to_base @ (ee_pos - fixed_pos)

    # Convert back to quaternion
    quat_relative = Rotation.from_matrix(R_fixed_to_ee).as_quat()
    
    return t_fixed_to_ee, quat_relative

def get_markers_relative_to_fixed(marker_positions, fixed_pos, fixed_quat):
    """
    Get marker positions relative to fixed frame.
    Notation: R_from_to (transforms vector 'from' frame 'to' frame)
    """
    
    # 1. Create the matrix that transforms FROM Fixed TO Base
    # (This is what a standard pose quaternion represents)
    R_base_to_fixed = Rotation.from_quat(fixed_quat).as_matrix()

    # 3. Translate markers (in Base frame) to be relative to Fixed origin
    # P_translated = P_base - P_fixed_origin
    translated = marker_positions - fixed_pos  # [M,3]
    
    # 4. Rotate markers into Fixed frame
    # P_fixed = R_base_to_fixed * P_translated
    # Note: We transpose 'translated' to shape [3,M] for matmul, then transpose back
    markers_relative = (R_base_to_fixed.T @ translated.T).T  # [M,3]
    
    return markers_relative

def _to_chw_float(x: np.ndarray) -> torch.Tensor:
    # x: HxW or HxWxC, uint8 or float
    if x.ndim == 2:
        x = x[..., None]
    t = torch.from_numpy(np.ascontiguousarray(x))
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
    
    # make C,H,W
    C ,H, W = 3 , 240, 320
    t = t.reshape(C, H, W)
    return t  # C,H,W

def _stack_digit_images(obj_array: np.ndarray) -> torch.Tensor:
    """
    obj_array: length-T object array; each entry is a list/array of S sensor images.
    Returns tensor [T, S, C, H, W].
    """
    img_tensors: List[torch.Tensor] = []
    for img in obj_array:                  # S loop
        img_tensors.append(_to_chw_float(img))
    return torch.stack(img_tensors, dim=0)  # [S,C,H,W]

def compute_ee_pos_mean_std(dataset):
    """Compute mean and std of all ee_positions across dataset."""
    sum_ = torch.zeros(3)
    sumsq_ = torch.zeros(3)
    count = 0
    for i in tqdm(range(len(dataset)), desc="ee pos stats"):
        # pos = dataset[i]["ee_position"].float()  # [3]
        with h5py.File(dataset.files[i], 'r') as f:
            fx_pos = f["fixed_position"][()]
            fx_ori = f["fixed_orientation"][()]
            ee_pos_raw = f["ee_position"][()]
            ee_ori_raw = f["ee_orientation"][()]
        ee_pos, ee_ori = get_ee_relative_to_fixed(
            ee_pos_raw, ee_ori_raw, fx_pos, fx_ori
        )
        sum_ += ee_pos
        sumsq_ += ee_pos ** 2
        count += 1
    mean = sum_ / count
    std = torch.sqrt(sumsq_ / count - mean ** 2)
    return mean, std

def compute_digit_mean_std(dataset):
    """
    Compute per-channel mean and std across all DIGIT images in the dataset.
    Each item has shape [S, C, H, W].
    Returns:
        mean: [C]
        std:  [C]
    """
    sum_ = torch.zeros(3)
    sumsq_ = torch.zeros(3)
    count = 0

    for i in tqdm(range(len(dataset)), desc="digit img stats"):
        # imgs = dataset[i]["digit_images"].float()  # [S,C,H,W]
        with h5py.File(dataset.files[i], 'r') as f:
            digit_group = f['digit_images']
            img_keys = list(digit_group.keys())
            imgs_list = [digit_group[key][:] for key in img_keys]
            imgs_obj = np.array(imgs_list, dtype=np.float32)  # [T, H, W, C]
        imgs = _stack_digit_images(imgs_obj)  # [S,C,H,W]
        S, C, H, W = imgs.shape
        n_pix = S * H * W
        sum_ += imgs.sum(dim=(0, 2, 3))            # sum over sensors, H, W
        sumsq_ += (imgs ** 2).sum(dim=(0, 2, 3))
        count += n_pix

    mean = sum_ / count
    std = torch.sqrt(sumsq_ / count - mean ** 2)
    return mean, std

def compute_all_stats(dataset, batch_size=128, num_workers=16,num_out_samples=14, with_images=True):
    """
    Computes Mean and Std for Digits, EE Pose, and Markers in a single pass.
    Drastically reduces disk I/O time.
    """
    print(f"Computing ALL stats using {num_workers} workers...")
    
    # 1. Disable Normalization to get raw data
    old_norm_state = dataset.normalize
    dataset.normalize = False
    dataset.normalize_inputs = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=True,
        pin_memory=True
    )

    # --- ACCUMULATORS (float64 for precision) ---
    # Digits: [R, G, B]
    img_sum = torch.zeros(3, dtype=torch.float32)
    img_sq_sum = torch.zeros(3, dtype=torch.float32)
    total_pixels = 0
    
    # EE Pose: [7] (3 pos + 4 ori)
    ee_sum = torch.zeros(7, dtype=torch.float32)
    ee_sq_sum = torch.zeros(7, dtype=torch.float32)
    
    # Markers: [128, 3]
    mark_sum = torch.zeros(num_out_samples, 3, dtype=torch.float32)
    mark_sq_sum = torch.zeros(num_out_samples, 3, dtype=torch.float32)
    
    total_samples = 0

    # --- SINGLE PASS LOOP ---
    for batch in tqdm(loader, desc="Scanning Dataset"):
        b= batch['ee_pose'].shape[0]
        # 1. DIGIT IMAGES
        # Shape: [B, 4, 3, H, W]
        if with_images:
            imgs = batch['digit_images'].to(dtype=torch.float32)
            img_sum += torch.sum(imgs[:,:,:3,:,:], dim=[0, 1, 3, 4])
            img_sq_sum += torch.sum(imgs[:,:,:3,:,:] ** 2, dim=[0, 1, 3, 4])
        
            b, s, c, h, w = batch['digit_images'].shape
            total_pixels += (b * s * c * h * w)
        
        # 2. EE POSE
        # Shape: [B, 7]
        ee = batch['ee_pose'].to(dtype=torch.float32)
        ee_sum += torch.sum(ee, dim=0)
        ee_sq_sum += torch.sum(ee ** 2, dim=0)

        # 3. MARKERS
        # Shape: [B, 12, 3]
        mark = batch['marker_positions'].to(dtype=torch.float32)
        mark_sum += torch.sum(mark, dim=0)
        mark_sq_sum += torch.sum(mark ** 2, dim=0)
        
        total_samples += b

    # --- COMPUTE FINAL STATS ---
    
    # Digits
    if with_images:
        digit_mean = img_sum / total_pixels
        digit_var = (img_sq_sum / total_pixels) - (digit_mean ** 2)
        digit_std = torch.sqrt(digit_var)

    # EE Pose
    ee_mean = ee_sum / total_samples
    ee_var = (ee_sq_sum / total_samples) - (ee_mean ** 2)
    ee_std = torch.sqrt(ee_var)

    # Markers
    mark_mean = mark_sum / total_samples
    mark_var = (mark_sq_sum / total_samples) - (mark_mean ** 2)
    mark_std = torch.sqrt(mark_var)

    # Restore state
    dataset.normalize = old_norm_state
    
    return {
        "digit_mean": digit_mean.float().tolist() if with_images else None,
        "digit_std": digit_std.float().tolist() if with_images else None,
        "ee_mean": ee_mean.float().tolist(),
        "ee_std": ee_std.float().tolist(),
        "marker_mean": mark_mean.float().tolist(),
        "marker_std": mark_std.float().tolist()
    }



