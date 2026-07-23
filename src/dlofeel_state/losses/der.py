import torch
import torch.nn.functional as F

# ============================================================================
# PIPELINE: Points → Edges → Bishop Frame → Material Frame
# ============================================================================

def pipeline_points_to_material_frame(points_3d):
    """
    Complete pipeline: ordered 3D points → material frame
    
    Input: 
        points_3d: torch.Tensor of shape (batch, n_vertices, 3)
                   Example: torch.tensor([[[0,0,0], [1,0,0], [2,0.5,0], [3,1,0]]])
    
    Output:
        m1, m2: Material frame vectors (batch, n_edges, 3)
        theta: Twist angles (batch, n_edges)
        kb: Discrete curvature binormal (batch, n_edges, 3)
        b_u, b_v: Bishop frame (batch, n_edges, 3)
    """
    
    # STEP 1: Compute edges (tangent vectors)
    edges = compute_edges(points_3d)
    
    # STEP 2: Compute edge lengths
    edge_lengths = compute_edge_lengths(edges)
    
    # STEP 3: Compute discrete curvature binormal (kb)
    kb = compute_curvature_binormal(edges, edge_lengths)
    
    # STEP 4: Initialize first Bishop frame vector
    init_direction = torch.tensor([0.0, 1.0, 0.0])  # arbitrary direction
    b_u0 = compute_bishop_u0(edges[: , 0], init_direction)
    
    # STEP 5: Compute full Bishop frame (u, v)
    b_u, b_v = compute_bishop_frame(b_u0, edges, kb)
    
    # STEP 6: Optimize twist angle θ to minimize elastic energy
    # (or set theta=0 for twist-free, or prescribed values)
    theta = optimize_twist_angle(kb, b_u, b_v, edge_lengths)
    
    # STEP 7: Compute material frame from Bishop + twist
    m1, m2 = compute_material_frame(theta, b_u, b_v)
    
    return m1, m2, theta, kb, b_u, b_v


# ============================================================================
# STEP 1: COMPUTE EDGES
# ============================================================================

def compute_edges(vertices):
    """
    Compute edge vectors from vertices
    
    Input:  vertices (batch, n_vert, 3)
    Output: edges (batch, n_edge, 3) where n_edge = n_vert - 1
    """
    edges = vertices[:, 1:] - vertices[:, :-1]
    return edges


# ============================================================================
# STEP 2: COMPUTE EDGE LENGTHS
# ============================================================================

def compute_edge_lengths(edges):
    """
    Compute length of each edge
    
    Input: edges (batch, n_edge, 3)
    Output: edge_lengths (batch, n_edge)
    """
    
    edge_lengths = torch.norm(edges, dim=2)
    return edge_lengths


# ============================================================================
# STEP 3: COMPUTE DISCRETE CURVATURE BINORMAL (kb)
# ============================================================================

def compute_curvature_binormal(edges, edge_lengths):
    """
    Discrete curvature binormal:  measures bending at each vertex
    From DER paper equation 1
    
    Input: 
        edges (batch, n_edge, 3)
        edge_lengths (batch, n_edge)
    Output:  
        kb (batch, n_edge, 3) - curvature binormal at each edge
    """
    batch, n_edge, _ = edges. shape
    kb = torch.zeros_like(edges)
    
    # kb[i] = 2 * (e[i-1] × e[i]) / (|e[i-1]| * |e[i]| + e[i-1]·e[i])
    kb[: , 1:] = torch.clamp(
        2 * torch.cross(edges[:, :-1], edges[:, 1:], dim=2) / 
        (edge_lengths[: , :-1] * edge_lengths[: , 1:] + 
         (edges[:, :-1] * edges[:, 1:]).sum(dim=2)).unsqueeze(dim=2),
        min=-20, max=20
    )
    
    return kb


# ============================================================================
# STEP 4: INITIALIZE FIRST BISHOP FRAME VECTOR
# ============================================================================

def compute_bishop_u0(e0, init_direction):
    """
    Initialize first edge's Bishop frame u-vector
    
    Input: 
        e0: first edge vector (batch, 3)
        init_direction: arbitrary 3D vector not aligned with e0
    Output: 
        u0: first Bishop u-vector (batch, 3)
    """
    batch = e0.size(0)
    init_direction = init_direction.view(1, 3).repeat(batch, 1)
    
    # Construct perpendicular vector using double cross product
    N_0 = torch.cross(e0, init_direction)
    u0 = F.normalize(torch.cross(N_0, e0), dim=1)
    
    return u0


# ============================================================================
# STEP 5: COMPUTE BISHOP FRAME (u, v)
# ============================================================================

def compute_bishop_frame(u0, edges, kb):
    """
    Compute twist-minimized Bishop frame for all edges
    
    Input: 
        u0: first Bishop u-vector (batch, 3)
        edges: edge vectors (batch, n_edge, 3)
        kb: curvature binormal (batch, n_edge, 3)
    Output:
        b_u, b_v: Bishop frame vectors (batch, n_edge, 3)
    """
    batch, n_edge, _ = edges. shape
    device = edges.device
    err = 1e-6
    
    # Initialize first edge
    b_u = u0.unsqueeze(1)  # (batch, 1, 3)
    b_v = F.normalize(torch.cross(edges[:, 0], b_u[: , 0], dim=1)).unsqueeze(1)
    
    # Compute rotation angles from curvature
    magnitude = (kb * kb).sum(dim=2)
    sinPhi, cosPhi = extract_sin_cos(magnitude)
    
    # Create quaternions for rotation
    q = quaternion_from_axis_angle(cosPhi, sinPhi, kb)
    
    # Parallel transport along the rod
    for i in range(1, n_edge):
        # Check if rotation is too small (numerical stability)
        needs_rotation = (1.0 - cosPhi[: , i]) > err
        
        # Rotate u and v using quaternions
        u_rotated = quaternion_rotate_vector(q[: , i], b_u[:, i-1])
        v_rotated = F.normalize(torch.cross(edges[:, i], u_rotated, dim=1))
        
        # Use previous frame if rotation is negligible
        u_new = torch.where(needs_rotation. unsqueeze(1), u_rotated, b_u[: , i-1])
        v_new = torch.where(needs_rotation.unsqueeze(1), v_rotated, b_v[:, i-1])
        
        b_u = torch.cat([b_u, u_new. unsqueeze(1)], dim=1)
        b_v = torch.cat([b_v, v_new.unsqueeze(1)], dim=1)
    
    return b_u, b_v


# ============================================================================
# STEP 6: OPTIMIZE TWIST ANGLE θ
# ============================================================================

def optimize_twist_angle(kb, b_u, b_v, edge_lengths, 
                         bend_stiffness=5e-5, twist_stiffness=1e-5):
    """
    Optimize twist angle to minimize elastic energy
    
    For simplicity, this version returns theta=0 (no twist)
    For full optimization, see the Theseus optimizer in DEFORM_func.py
    
    Input:
        kb: curvature binormal (batch, n_edge, 3)
        b_u, b_v:  Bishop frame (batch, n_edge, 3)
        edge_lengths: (batch, n_edge)
    Output:
        theta: twist angles (batch, n_edge)
    """
    batch, n_edge, _ = kb.shape
    
    # Option 1: No twist (theta = 0)
    theta = torch.zeros(batch, n_edge, device=kb.device)
    
    # Option 2: Prescribed boundary conditions
    # theta[: , 0] = boundary_theta_start
    # theta[:, -1] = boundary_theta_end
    # ...  optimize interior points ... 
    
    # Option 3: Use full non-linear optimization (from DEFORM)
    # See non_linear_opt_theta_full() in the repository
    
    return theta


# ============================================================================
# STEP 7: COMPUTE MATERIAL FRAME
# ============================================================================

def compute_material_frame(theta, b_u, b_v):
    """
    Rotate Bishop frame by twist angle to get material frame
    
    Input:
        theta: twist angles (batch, n_edge)
        b_u, b_v:  Bishop frame vectors (batch, n_edge, 3)
    Output:
        m1, m2: Material frame vectors (batch, n_edge, 3)
    """
    cos_theta = torch.cos(theta).unsqueeze(2)  # (batch, n_edge, 1)
    sin_theta = torch.sin(theta).unsqueeze(2)
    
    # 2D rotation matrix applied to (u, v)
    m1 = cos_theta * b_u + sin_theta * b_v
    m2 = -sin_theta * b_u + cos_theta * b_v
    
    return m1, m2


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_sin_cos(magnitude):
    """Extract sin and cos from rotation magnitude for numerical stability"""
    constant = 4.0
    sin_phi = torch.sqrt(torch.clamp(magnitude / (constant + magnitude), 1e-10))
    cos_phi = torch.sqrt(torch.clamp(constant / (constant + magnitude), 1e-10))
    return sin_phi, cos_phi


def quaternion_from_axis_angle(cos_phi, sin_phi, kb):
    """Create quaternion from axis-angle representation"""
    batch, n_edge, _ = kb.shape
    kb_normalized = F.normalize(kb, dim=2)
    
    # Quaternion:  [w, x, y, z] = [cos(φ/2), sin(φ/2) * axis]
    q = torch.cat([
        cos_phi. unsqueeze(2),  # w component
        sin_phi.unsqueeze(2) * kb_normalized  # vector component
    ], dim=2)
    
    return q


def quaternion_rotate_vector(q, v):
    """Rotate vector v by quaternion q"""
    # q = [w, x, y, z], v = [x, y, z]
    # Result: q * [0, v] * q^*
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[: , 1], v[:, 2]
    
    # Optimized quaternion-vector multiplication
    t_x = 2 * (y * vz - z * vy)
    t_y = 2 * (z * vx - x * vz)
    t_z = 2 * (x * vy - y * vx)
    
    v_rot_x = vx + w * t_x + (y * t_z - z * t_y)
    v_rot_y = vy + w * t_y + (z * t_x - x * t_z)
    v_rot_z = vz + w * t_z + (x * t_y - y * t_x)
    
    return torch.stack([v_rot_x, v_rot_y, v_rot_z], dim=1)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__": 
    # Example:  5 points forming a simple curve
    points = torch.tensor([
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.5, 0.0],
         [3.0, 1.0, 0.0],
         [4.0, 1.0, 0.5]]
    ])  # Shape: (1, 5, 3)
    
    print("Input points shape:", points.shape)
    print("Points:\n", points)
    
    # Run the pipeline
    m1, m2, theta, kb, b_u, b_v = pipeline_points_to_material_frame(points)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"\nEdges (tangents): {points. shape[1]-1} edges")
    print(f"Curvature binormal (kb) shape: {kb.shape}")
    print(f"Bishop frame (u, v) shape: {b_u.shape}, {b_v.shape}")
    print(f"Twist angles (θ) shape: {theta.shape}")
    print(f"Material frame (m1, m2) shape: {m1.shape}, {m2.shape}")
    
    print("\nMaterial frame m1 (first 3 edges):")
    print(m1[0, :3])
    print("\nMaterial frame m2 (first 3 edges):")
    print(m2[0, : 3])
    
    # Verify orthonormality
    print("\n" + "="*60)
    print("VERIFICATION:")
    print("="*60)
    
    # Check m1 · m2 = 0 (orthogonal)
    dot_product = (m1 * m2).sum(dim=2)
    print(f"m1 · m2 (should be ~0): {dot_product[0, : 3]}")
    
    # Check |m1| = 1, |m2| = 1 (normalized)
    m1_norm = torch.norm(m1, dim=2)
    m2_norm = torch.norm(m2, dim=2)
    print(f"|m1| (should be ~1): {m1_norm[0, :3]}")
    print(f"|m2| (should be ~1): {m2_norm[0, :3]}")