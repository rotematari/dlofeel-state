"""Numeric parity gate for the ported physics-informed losses.

The losses were copied verbatim from the frozen original (simplepredict) and
only had their imports rewired. These golden values were captured by running the
identical snippets against the ORIGINAL modules (simplepredict.losses,
simplepredict.loss.der) with $ORIG/src on sys.path; the ported functions must
reproduce them bit-for-bit (atol 1e-6). If a golden ever drifts, the port
changed behaviour, which this gate forbids.
"""
import torch

# Golden values captured from the frozen original (see module docstring).
CHAMFER_GOLDEN = [0.2198209911584854, 0.23318079113960266]
KB_GOLDEN = [
    [
        [0.0, 0.0, 0.0],
        [-6.67783260345459, 3.6402902603149414, -1.3834254741668701],
        [-1.8150157928466797, -0.08516129106283188, -3.9229419231414795],
        [0.2568078637123108, 0.25099390745162964, -1.8718122243881226],
        [-0.5779662728309631, 4.325934886932373, -3.9532041549682617],
    ]
]


def test_chamfer_distance_per_sample_matches_original():
    from dlofeel_state.losses.chamfer import chamfer_distance_per_sample

    torch.manual_seed(0)
    a = torch.rand(2, 16, 3)
    b = torch.rand(2, 16, 3)
    out = chamfer_distance_per_sample(a, b)

    assert torch.allclose(out, torch.tensor(CHAMFER_GOLDEN), atol=1e-6)


def test_compute_curvature_binormal_matches_original():
    from dlofeel_state.losses.der import (
        compute_curvature_binormal,
        compute_edge_lengths,
        compute_edges,
    )

    torch.manual_seed(0)
    points = torch.rand(1, 6, 3)
    edges = compute_edges(points)
    edge_lengths = compute_edge_lengths(edges)
    kb = compute_curvature_binormal(edges, edge_lengths)

    assert torch.allclose(kb, torch.tensor(KB_GOLDEN), atol=1e-6)
