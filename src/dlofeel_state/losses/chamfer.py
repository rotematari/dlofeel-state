import torch


### ------------------------------ Mean Per-Point Distance (vectorized) ------------------------------
def mean_point_distance_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean per-point L2 distance for each sample in the batch.
    
    Args:
        pred: [B, N, 3] predicted points
        target: [B, N, 3] ground truth points
    
    Returns:
        [B] tensor: mean L2 distance per point, for each sample
    """
    # Per-point L2 distance: sqrt(x^2 + y^2 + z^2)
    l2_per_point = torch.norm(pred - target, dim=-1)  # [B, N]
    # Mean across points for each sample
    return l2_per_point.mean(dim=-1)  # [B]


### ------------------------------ Chamfer Distance Loss ------------------------------
def chamfer_loss(pred: torch.Tensor, target: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
    """
    Chamfer distance loss between predicted and target point clouds.
    
    Args:
        pred: [B, N, 3] predicted points
        target: [B, M, 3] ground truth points (typically N == M for DLO)
        bidirectional: if True, compute both pred->target and target->pred distances
    
    Returns:
        Scalar tensor: mean Chamfer distance across batch
    """
    # Compute pairwise distances: [B, N, M]
    # Using cdist for efficient pairwise distance computation
    dist_matrix = torch.cdist(pred, target, p=2)  # [B, N, M]
    
    # Forward: for each predicted point, find nearest ground truth point
    # [B, N] - distance from each pred point to its nearest gt point
    dist_pred_to_gt, _ = dist_matrix.min(dim=2)
    
    if bidirectional:
        # Backward: for each ground truth point, find nearest predicted point
        # [B, M] - distance from each gt point to its nearest pred point
        dist_gt_to_pred, _ = dist_matrix.min(dim=1)
        
        # Chamfer distance = mean of both directions
        chamfer = (dist_pred_to_gt.mean(dim=1) + dist_gt_to_pred.mean(dim=1)) / 2
    else:
        # Unidirectional: only pred -> gt
        chamfer = dist_pred_to_gt.mean(dim=1)
    
    return chamfer.mean()  # scalar


def chamfer_distance_per_sample(pred: torch.Tensor, target: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
    """
    Compute Chamfer distance for each sample in the batch (returns per-sample values).
    
    Args:
        pred: [B, N, 3] predicted points
        target: [B, M, 3] ground truth points
        bidirectional: if True, compute symmetric distance
    
    Returns:
        [B] tensor: Chamfer distance for each sample
    """
    dist_matrix = torch.cdist(pred, target, p=2)  # [B, N, M]
    
    dist_pred_to_gt, _ = dist_matrix.min(dim=2)  # [B, N]
    
    if bidirectional:
        dist_gt_to_pred, _ = dist_matrix.min(dim=1)  # [B, M]
        chamfer = (dist_pred_to_gt.mean(dim=1) + dist_gt_to_pred.mean(dim=1)) / 2
    else:
        chamfer = dist_pred_to_gt.mean(dim=1)
    
    return chamfer  # [B]


