import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from dlofeel_state.losses.der import compute_edges, compute_edge_lengths, compute_curvature_binormal
import torch


def sort_points_from_origin(points, momentum_weight=0.5):
    """
    Sorts 3D points into a wire path.
    - Start: Point closest to (0,0,0).
    - Second: Nearest neighbor to the start.
    - Rest: Follows momentum + proximity.
    """
    n = len(points)
    if n < 2:
        return points
        
    # Pre-calculate distance matrix for all points
    all_dists = cdist(points, points)
    
    # --- 1. Find First Point (Closest to Origin) ---
    # distance from 0,0,0 is just the norm of the point vector
    dists_to_origin = np.linalg.norm(points, axis=-1)
    start_idx = np.argmin(dists_to_origin)
    
    # --- 2. Find Second Point (Nearest Neighbor to First) ---
    # Get distances from the start point
    dists_from_start = all_dists[start_idx].copy()
    dists_from_start[start_idx] = np.inf  # Don't pick itself
    second_idx = np.argmin(dists_from_start)
    
    # Initialize the chain
    ordered_indices = [start_idx, second_idx]
    visited = set(ordered_indices)
    current_idx = second_idx
    
    # --- 3. Trace the rest using Momentum ---
    for _ in range(n - 2):
        # A. Vector from (i-2) to (i-1)
        p_prev = points[ordered_indices[-2]]
        p_curr = points[ordered_indices[-1]]
        prev_vector = p_curr - p_prev
        
        # B. Predict next position (Momentum)
        predicted_pos = p_curr + prev_vector
        
        # C. Calculate Costs
        # 1. Proximity (Distance from current point)
        dists_to_current = all_dists[current_idx]
        
        # 2. Momentum (Distance from predicted position)
        # Calculate distance from 'predicted_pos' to all other points
        dists_to_pred = np.linalg.norm(points - predicted_pos, axis=1)
        
        combined_cost = dists_to_current + (momentum_weight * dists_to_pred)
        
        # Mask visited
        combined_cost[list(visited)] = np.inf
        
        # Find winner
        next_idx = np.argmin(combined_cost)
        
        ordered_indices.append(next_idx)
        visited.add(next_idx)
        current_idx = next_idx

    return points[ordered_indices]


def sort_points_from_origin_gpu(points: torch.Tensor, momentum_weight: float = 0.5) -> torch.Tensor:
    """
    GPU-compatible version: Sorts 3D points into a wire path using PyTorch.
    - Start: Point closest to (0,0,0).
    - Second: Nearest neighbor to the start.
    - Rest: Follows momentum + proximity.
    
    Args:
        points: Tensor of shape (N, 3) on any device (CPU or CUDA)
        momentum_weight: Weight for momentum term in cost function
        
    Returns:
        Sorted points tensor of shape (N, 3) on the same device
    """
    n = points.shape[0]
    device = points.device
    
    if n < 2:
        return points
    
    # Pre-calculate distance matrix for all points using torch.cdist
    # cdist expects (B, N, D), so unsqueeze and squeeze
    all_dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0), p=2).squeeze(0)  # (N, N)
    
    # --- 1. Find First Point (Closest to Origin) ---
    dists_to_origin = torch.norm(points, dim=-1)  # (N,)
    start_idx = torch.argmin(dists_to_origin).item()
    
    # --- 2. Find Second Point (Nearest Neighbor to First) ---
    dists_from_start = all_dists[start_idx].clone()
    dists_from_start[start_idx] = float('inf')
    second_idx = torch.argmin(dists_from_start).item()
    
    # Initialize the chain
    ordered_indices = [start_idx, second_idx]
    
    # Visited mask (more efficient than set for GPU operations)
    visited_mask = torch.zeros(n, dtype=torch.bool, device=device)
    visited_mask[start_idx] = True
    visited_mask[second_idx] = True
    
    current_idx = second_idx
    
    # --- 3. Trace the rest using Momentum ---
    for _ in range(n - 2):
        # A. Vector from (i-2) to (i-1)
        p_prev = points[ordered_indices[-2]]
        p_curr = points[ordered_indices[-1]]
        prev_vector = p_curr - p_prev
        
        # B. Predict next position (Momentum)
        predicted_pos = p_curr + prev_vector
        
        # C. Calculate Costs
        # 1. Proximity (Distance from current point)
        dists_to_current = all_dists[current_idx]  # (N,)
        
        # 2. Momentum (Distance from predicted position)
        dists_to_pred = torch.norm(points - predicted_pos, dim=1)  # (N,)
        
        combined_cost = dists_to_current + (momentum_weight * dists_to_pred)
        
        # Mask visited points with inf
        combined_cost = combined_cost.masked_fill(visited_mask, float('inf'))
        
        # Find winner
        next_idx = torch.argmin(combined_cost).item()
        
        ordered_indices.append(next_idx)
        visited_mask[next_idx] = True
        current_idx = next_idx
    
    # Gather sorted points using index tensor
    indices_tensor = torch.tensor(ordered_indices, dtype=torch.long, device=device)
    return points[indices_tensor]

def calculate_angle(p_prev: torch.Tensor, p_curr: torch.Tensor, p_next: torch.Tensor) -> torch.Tensor:
    """
    Calculates the interior angle in radians at p_curr formed by 
    p_prev -> p_curr and p_curr -> p_next.
    """
    # Vectors pointing OUT from p_curr
    v_a = p_prev - p_curr
    v_b = p_next - p_curr
    
    norm_a = torch.norm(v_a)
    norm_b = torch.norm(v_b)
    
    if norm_a == 0 or norm_b == 0:
        return torch.tensor(0.0, device=p_curr.device)

    dot_product = torch.dot(v_a, v_b)
    
    # Clamp to avoid numerical errors outside [-1, 1]
    cos_angle = torch.clamp(dot_product / (norm_a * norm_b), -1.0, 1.0)
    angle = torch.acos(cos_angle)
    
    return angle

def get_adjacency_list(points: torch.Tensor, threshold: float = 0.05):
    """
    Pre-computes neighbors for every point in M based on Euclidean distance.
    Returns a list where index i contains indices of neighbors of point i.
    
    Args:
        points: Tensor of shape (N, 3)
        threshold: Distance threshold for adjacency
    """
    # Compute distance matrix using torch.cdist
    # cdist expects (B, N, D) so we unsqueeze and squeeze
    dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0), p=2).squeeze(0)  # (N, N)
    
    adjacency = []
    num_points = len(points)
    
    for i in range(num_points):
        # Indices where distance is < threshold, excluding self (dist > 0)
        mask = (dists[i] < threshold) & (dists[i] > 0)
        neighbors = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        adjacency.append(neighbors)
        
    return adjacency

def compute_kb(points: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute curvature binormal magnitude for a point sequence.
    
    Args:
        points: Tensor of shape (N, 3) or (B, N, 3)
        eps: Small value for numerical stability
        
    Returns:
        Scalar tensor representing total curvature magnitude
    """
    # Add batch dimension if needed
    if points.dim() == 2:
        points = points.unsqueeze(0)  # (N, 3) -> (1, N, 3)
    
    point_edges = compute_edges(points)
    point_lengths = compute_edge_lengths(point_edges)
    point_kb = compute_curvature_binormal(point_edges, point_lengths)
    
    # Return scalar metric: mean of curvature binormal magnitudes
    return point_kb.norm(dim=-1).mean()

def sort_points_from_origin_using_kb(points: torch.Tensor, alpha_degrees: float = 90):
    """
    Sort points into a wire path using curvature binormal minimization.
    
    Args:
        points: Tensor of shape (N, 3)
        alpha_degrees: Angle constraint in degrees
        
    Returns:
        Sorted points tensor of shape (N, 3)
    """
    import math
    
    marker_length = len(points)

    # Convert alpha to radians
    alpha_rad = math.radians(alpha_degrees)
    upper_bound_rad = math.pi + alpha_rad
    
    min_value = [float('inf')] # Mutable to allow updates in recursion
    best_path_indices = [None]
    
    # Pre-compute Neighbors (Distance < 5)
    neighbors_map = get_adjacency_list(points, threshold=0.05)
    
    # Recursive DFS Function
    def search(current_path_indices):
        # Base Case: Path length reached
        if len(current_path_indices) == marker_length:
            current_path_points = points[current_path_indices]

            dist = compute_kb(current_path_points)

            if dist < min_value[0]:
                min_value[0] = dist
                best_path_indices[0] = list(current_path_indices)
            return

        # Recursive Step
        last_idx = current_path_indices[-1]
        possible_next = neighbors_map[last_idx]
        
        for next_idx_tensor in possible_next:
            next_idx = next_idx_tensor.item()  # Convert tensor to int
            
            # Avoid loops (n not in {n1...ni})
            if next_idx in current_path_indices:
                continue
            
            # Smoothness Constraint
            if len(current_path_indices) >= 2:
                prev_idx = current_path_indices[-2]
                
                p_prev = points[prev_idx]
                p_curr = points[last_idx]
                p_next = points[next_idx]
                
                angle = calculate_angle(p_prev, p_curr, p_next)
                
                # Check constraint: angle < alpha OR angle > PI + alpha
                if angle < alpha_rad or angle > upper_bound_rad:
                    continue 

            search(current_path_indices + [next_idx])

    # Start search from every point
    # print(f"  > Searching over {len(M)} candidate points...")
    for start_idx in range(len(points)):
        search([start_idx])
        
    return points[best_path_indices[0]]



def fit_spline_unordered(points, num_samples=100, smoothing=0.1 , k=3, use_gpu=True):
    """
    Wrapper that sorts points first, then fits the spline.
    
    Args:
        points: Array or tensor of shape (N, 3)
        num_samples: Number of points to resample the spline to
        smoothing: Spline smoothing parameter (unused, kept for compatibility)
        k: Spline degree
        use_gpu: If True, use GPU for sorting (don't use in DataLoader workers)
    """
    # 1. Sort the points spatially
    # Note: Don't use GPU in DataLoader workers (forked processes can't use CUDA)
    if use_gpu and torch.cuda.is_available():
        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float32, device='cpu')
        sorted_points = sort_points_from_origin_gpu(points, momentum_weight=0.2)
    else:
        # CPU path - use numpy version (safe for DataLoader workers)
        if torch.is_tensor(points):
            points = points.detach().cpu().numpy()
        sorted_points = sort_points_from_origin(points, momentum_weight=0.2)
    
    
    # Convert to numpy for splprep
    if torch.is_tensor(sorted_points):
        sorted_points = sorted_points.detach().cpu().numpy()
    if sorted_points.ndim != 2:
        sorted_points = sorted_points.squeeze(0)
    
    # 2. Fit the spline
    # Transpose for splprep
    m = len(sorted_points)
    s = m - np.sqrt(2 * m)
    tck, u = splprep(sorted_points.T, s=s, k=k)
    
    # 3. Resample
    u_new = np.linspace(0, 1, num_samples)
    new_points = np.array(splev(u_new, tck)).T

    # Pull the control points from the tck coefficients
    
    
    return new_points, sorted_points


def fit_spline_with_control_points(points, num_samples=100, n_control_points=16, k=3, use_gpu=False):
    """
    Fits a B-spline to unordered points and returns both resampled points AND control points.
    
    The control points are derived from the spline coefficients (tck) and uniformly
    resampled to the requested count.
    
    Args:
        points: Array or tensor of shape (N, 3)
        num_samples: Number of points to resample the spline to
        n_control_points: Number of control points to return
        k: Spline degree
        use_gpu: If True, use GPU for sorting (don't use in DataLoader workers)
    
    Returns:
        resampled_points: ndarray of shape (num_samples, 3)
        control_points: ndarray of shape (n_control_points, 3)
    """
    # 1. Sort the points spatially
    if use_gpu and torch.cuda.is_available():
        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float32, device='cpu')
        sorted_points = sort_points_from_origin_gpu(points, momentum_weight=0.2)
    else:
        # CPU path - use numpy version (safe for DataLoader workers)
        if torch.is_tensor(points):
            points = points.detach().cpu().numpy()
        sorted_points = sort_points_from_origin(points, momentum_weight=0.2)
    
    # Convert to numpy for splprep
    if torch.is_tensor(sorted_points):
        sorted_points = sorted_points.detach().cpu().numpy()
    if sorted_points.ndim != 2:
        sorted_points = sorted_points.squeeze(0)
    
    # 2. Fit the spline
    m = len(sorted_points)
    s = m - np.sqrt(2 * m)
    tck, u = splprep(sorted_points.T, s=s, k=k)
    
    # 3. Resample the spline curve
    u_new = np.linspace(0, 1, num_samples)
    resampled_points = np.array(splev(u_new, tck)).T
    
    # 4. Extract control points from spline coefficients
    # tck = (t, c, k) where c is a list of coefficient arrays [cx, cy, cz]
    # Each has shape (num_knots,) - the actual control points
    t, c, k_degree = tck
    
    # c contains the actual B-spline control polygon vertices
    # Stack into (M, 3) where M is the number of native control points
    native_ctrl_pts = np.column_stack(c)  # Shape: (num_native_ctrl, 3)
    
    # Uniformly resample to get exactly n_control_points
    n_native = len(native_ctrl_pts)
    if n_native >= n_control_points:
        # Subsample
        indices = np.linspace(0, n_native - 1, n_control_points, dtype=int)
        control_points = native_ctrl_pts[indices]
    else:
        # Interpolate to get more points
        t_native = np.linspace(0, 1, n_native)
        t_new = np.linspace(0, 1, n_control_points)
        control_points = np.zeros((n_control_points, 3))
        for dim in range(3):
            control_points[:, dim] = np.interp(t_new, t_native, native_ctrl_pts[:, dim])
    
    return resampled_points, control_points

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create a clean wire
    n = 50
    t = np.linspace(0, 4*np.pi, n)
    x = np.cos(t)
    y = np.sin(t)
    z = t / 5.0
    clean_wire = np.column_stack([x, y, z])

    # 2. SHUFFLE the data (simulate unordered point cloud)
    np.random.shuffle(clean_wire)
    shuffled_wire = clean_wire # Now points[0] might be next to points[23]

    # 3. Fit (The function handles the sorting)
    resampled_wire, ordered_wire = fit_spline_unordered(shuffled_wire, num_samples=200, smoothing=0)

    # 4. Visualization
    fig = plt.figure(figsize=(12, 5))

    # Plot 1: The messy input (if we connected lines in input order)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(shuffled_wire[:,0], shuffled_wire[:,1], shuffled_wire[:,2], 'r-', alpha=0.5, label='Input Order (Messy)')
    ax1.scatter(shuffled_wire[:,0], shuffled_wire[:,1], shuffled_wire[:,2], c='k', s=10)
    ax1.set_title("Input (Shuffled)")

    # Plot 2: The sorted and fitted result
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(resampled_wire[:,0], resampled_wire[:,1], resampled_wire[:,2], 'b-', linewidth=3, label='Fitted Spline')
    # Visualize the sorting logic with a gradient color
    ax2.scatter(ordered_wire[:,0], ordered_wire[:,1], ordered_wire[:,2], c=np.arange(n), cmap='viridis', s=20, label='Sorted Points')
    ax2.set_title("Result (Sorted & Fitted)")

    plt.show()