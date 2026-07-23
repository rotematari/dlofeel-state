import torch

# ============================================================================
# BENDING ENERGY CALCULATION
# ============================================================================

def compute_bending_energy(points_3d, m1, m2, kb, edge_lengths, 
                          rest_curvature_prev=None, rest_curvature_next=None,
                          bend_stiffness=5e-5):
    """
    Compute bending energy of a deformable linear object (DLO)
    Based on Discrete Elastic Rods (DER) paper
    
    The bending energy measures resistance to curvature changes from rest state
    
    Input:
        points_3d: Current vertices (batch, n_vert, 3)
        m1, m2: Material frame vectors (batch, n_edge, 3)
        kb: Discrete curvature binormal (batch, n_edge, 3)
        edge_lengths: Length of each edge (batch, n_edge)
        rest_curvature_prev:  Undeformed curvature (batch, n_edge, 2) [optional]
        rest_curvature_next: Undeformed curvature (batch, n_edge, 2) [optional]
        bend_stiffness: Material property (scalar or (batch, n_edge))
    
    Output:
        bending_energy: Total bending energy (batch,)
        curvature:  Current material curvature (batch, n_edge, 2)
        voronoi_lengths: Region lengths for energy computation (batch, n_edge)
    """
    
    batch, n_edge, _ = kb.shape
    device = kb.device
    
    # STEP 1: Compute Voronoi region lengths
    voronoi_lengths = compute_voronoi_lengths(edge_lengths)
    
    # STEP 2: Project curvature onto material frame
    current_curvature = project_curvature_to_material_frame(kb, m1, m2)
    
    # STEP 3: Get rest curvature (if not provided, assume straight rod)
    if rest_curvature_prev is None: 
        rest_curvature_prev = torch.zeros(batch, n_edge, 2, device=device)
    if rest_curvature_next is None:
        rest_curvature_next = torch.zeros(batch, n_edge, 2, device=device)
    
    # STEP 4: Compute curvature differences
    curvature_diff_prev = current_curvature - rest_curvature_prev
    curvature_diff_next = current_curvature - rest_curvature_next
    
    # STEP 5: Handle bend stiffness
    if isinstance(bend_stiffness, (int, float)):
        bend_stiffness = torch.ones(batch, n_edge, device=device) * bend_stiffness
    
    # Create stiffness for prev and next with proper indexing
    bend_stiffness_prev = torch.cat([
        torch.zeros(batch, 1, device=device), 
        bend_stiffness[: , :-1]
    ], dim=1).unsqueeze(-1)  # (batch, n_edge, 1)
    
    bend_stiffness_next = torch.cat([
        torch.zeros(batch, 1, device=device),
        bend_stiffness[: , 1:]
    ], dim=1).unsqueeze(-1)  # (batch, n_edge, 1)
    
    # STEP 6: Compute bending energy
    # Energy = 0.5 * Σ[(k_prev*(W-W_rest_prev)² + k_next*(W-W_rest_next)²) / L_voronoi]
    
    # Skip first edge (index 0) as it has no predecessor
    voronoi_unsqueezed = voronoi_lengths[: , 1:]. unsqueeze(-1)  # (batch, n_edge-1, 1)
    
    energy_prev = (bend_stiffness_prev[: , 1:] * 
                   curvature_diff_prev[:, 1:] * 
                   curvature_diff_prev[:, 1:]).sum(dim=2)  # Sum over 2 curvature components
    
    energy_next = (bend_stiffness_next[:, 1:] * 
                   curvature_diff_next[:, 1:] * 
                   curvature_diff_next[: , 1:]).sum(dim=2)
    
    bending_energy = 0.5 * ((energy_prev + energy_next) / voronoi_unsqueezed. squeeze(-1)).sum(dim=1)
    
    return bending_energy, current_curvature, voronoi_lengths


# ============================================================================
# HELPER FUNCTION 1: VORONOI REGION LENGTHS
# ============================================================================

def compute_voronoi_lengths(edge_lengths):
    """
    Compute Voronoi region lengths (used for energy integration)
    Voronoi length at vertex i = (edge[i-1] + edge[i]) / 2
    
    Input:
        edge_lengths: (batch, n_edge)
    Output:
        voronoi_lengths: (batch, n_edge)
    """
    batch, n_edge = edge_lengths.shape
    device = edge_lengths.device
    
    # First edge has no predecessor
    voronoi = torch.zeros(batch, 1, device=device)
    
    # For interior vertices:  sum of adjacent edges
    voronoi = torch.cat([
        voronoi,
        edge_lengths[: , :-1] + edge_lengths[:, 1:]
    ], dim=1)
    
    return voronoi


# ============================================================================
# HELPER FUNCTION 2: PROJECT CURVATURE TO MATERIAL FRAME
# ============================================================================

def project_curvature_to_material_frame(kb, m1, m2):
    """
    Project discrete curvature binormal onto material frame
    This gives the two components of curvature (κ₁, κ₂) in material coordinates
    
    From DER paper equation 2: 
    ω₁ = kb · m2
    ω₂ = -kb · m1
    
    Input:
        kb: Curvature binormal (batch, n_edge, 3)
        m1, m2: Material frame (batch, n_edge, 3)
    Output:
        curvature: (batch, n_edge, 2) where [: ,: ,0]=ω₁, [:,:,1]=ω₂
    """
    # Component 1: projection onto m2
    omega_1 = (kb * m2).sum(dim=2).unsqueeze(2)  # (batch, n_edge, 1)
    
    # Component 2: negative projection onto m1
    omega_2 = -(kb * m1).sum(dim=2).unsqueeze(2)  # (batch, n_edge, 1)
    
    # Combine into 2D curvature vector
    curvature = torch. cat([omega_1, omega_2], dim=2)  # (batch, n_edge, 2)
    
    return curvature


# ============================================================================
# HELPER FUNCTION 3: COMPUTE REST CURVATURE
# ============================================================================

def compute_rest_curvature(rest_points, rest_m1, rest_m2, rest_kb):
    """
    Compute rest (undeformed) curvature from rest configuration
    
    Input:
        rest_points: Rest vertices (batch, n_vert, 3)
        rest_m1, rest_m2: Rest material frame (batch, n_edge, 3)
        rest_kb: Rest curvature binormal (batch, n_edge, 3)
    Output:
        rest_curvature_prev, rest_curvature_next:  (batch, n_edge, 2)
    """
    batch, n_edge, _ = rest_kb.shape
    device = rest_kb. device
    
    # Project rest curvature
    rest_curvature = project_curvature_to_material_frame(rest_kb, rest_m1, rest_m2)
    
    # Create prev and next versions (offset by one edge)
    rest_curvature_prev = torch.zeros(batch, n_edge, 2, device=device)
    rest_curvature_next = torch.zeros(batch, n_edge, 2, device=device)
    
    rest_curvature_prev[:, 1:] = rest_curvature[: , :-1]
    rest_curvature_next[:, 1:] = rest_curvature[:, 1:]
    
    return rest_curvature_prev, rest_curvature_next


# ============================================================================
# COMPLETE EXAMPLE WITH FULL PIPELINE
# ============================================================================

def compute_bending_energy_from_points(points_3d, rest_points=None, 
                                       bend_stiffness=5e-5):
    """
    Complete pipeline:  3D points → bending energy
    
    Input: 
        points_3d: Current configuration (batch, n_vert, 3)
        rest_points: Rest configuration (batch, n_vert, 3) [optional]
        bend_stiffness: Material stiffness
    Output:
        bending_energy: Total bending energy (batch,)
        details: Dictionary with intermediate values
    """
    
    # Import from previous pipeline
    from dlofeel_state.losses.der import pipeline_points_to_material_frame
    
    # CURRENT CONFIGURATION
    m1, m2, theta, kb, b_u, b_v = pipeline_points_to_material_frame(points_3d)
    edges = points_3d[: , 1:] - points_3d[:, :-1]
    edge_lengths = torch.norm(edges, dim=2)
    
    # REST CONFIGURATION
    if rest_points is not None:
        rest_m1, rest_m2, rest_theta, rest_kb, _, _ = pipeline_points_to_material_frame(rest_points)
        rest_curvature_prev, rest_curvature_next = compute_rest_curvature(
            rest_points, rest_m1, rest_m2, rest_kb
        )
    else:
        # Assume straight rod at rest (zero curvature)
        rest_curvature_prev = None
        rest_curvature_next = None
    
    # COMPUTE BENDING ENERGY
    bending_energy, current_curvature, voronoi_lengths = compute_bending_energy(
        points_3d, m1, m2, kb, edge_lengths,
        rest_curvature_prev, rest_curvature_next,
        bend_stiffness
    )
    
    # Return detailed information
    details = {
        'material_frame_m1': m1,
        'material_frame_m2': m2,
        'curvature_binormal': kb,
        'current_curvature': current_curvature,
        'voronoi_lengths': voronoi_lengths,
        'edge_lengths': edge_lengths,
        'twist_angles': theta
    }
    
    return bending_energy, details


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__": 
    
    print("="*70)
    print("BENDING ENERGY CALCULATION EXAMPLE")
    print("="*70)
    
    # Example 1: Bent rod vs straight rest configuration
    # Current configuration:  curved
    current_points = torch.tensor([
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.5, 0.0],
         [3.0, 1.0, 0.0],
         [4.0, 1.2, 0.0]]
    ])
    
    # Rest configuration:  straight
    rest_points = torch.tensor([
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0],
         [4.0, 0.0, 0.0]]
    ])
    
    print("\nCurrent configuration (bent):")
    print(current_points[0])
    print("\nRest configuration (straight):")
    print(rest_points[0])
    
    # Compute bending energy
    bend_stiffness = 1e-4  # Material stiffness parameter
    bending_energy, details = compute_bending_energy_from_points(
        current_points, 
        rest_points, 
        bend_stiffness
    )
    
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"\nBending Energy: {bending_energy. item():.6e}")
    print(f"Bend Stiffness: {bend_stiffness:.6e}")
    
    print(f"\nCurvature binormal (kb) shape: {details['curvature_binormal'].shape}")
    print(f"Current curvature (ω) shape: {details['current_curvature'].shape}")
    
    print("\nCurvature at each vertex (first 4):")
    print(details['current_curvature'][0, :4])
    
    # Example 2: Same configuration for current and rest (should give ~0 energy)
    print(f"\n{'='*70}")
    print("VERIFICATION: Same current and rest configuration")
    print(f"{'='*70}")
    
    same_config = current_points.clone()
    energy_zero, _ = compute_bending_energy_from_points(
        same_config,
        same_config,  # Same as rest
        bend_stiffness
    )
    print(f"Bending Energy (should be ~0): {energy_zero.item():.6e}")
    
    # Example 3: More bent = more energy
    print(f"\n{'='*70}")
    print("VERIFICATION: More bending = More energy")
    print(f"{'='*70}")
    
    highly_bent = torch.tensor([
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [2.0, 1.0, 0.0],  # More bent
         [3.0, 2.0, 0.0],
         [4.0, 2.5, 0.0]]
    ])
    
    energy_high, _ = compute_bending_energy_from_points(
        highly_bent,
        rest_points,
        bend_stiffness
    )
    
    print(f"Original bending:  {bending_energy.item():.6e}")
    print(f"Higher bending:    {energy_high.item():.6e}")
    print(f"Ratio:             {energy_high.item() / bending_energy.item():.2f}x")