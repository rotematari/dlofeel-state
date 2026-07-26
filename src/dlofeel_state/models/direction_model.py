"""
Direction-input ablation model.

Replaces the Sparsh ViT image backbone with a non-differentiable CV pipeline
that extracts DLO direction vectors [vx, vy] from each tactile camera image.
These are projected to latent tokens and fed into the same cross-attention
decoder used by the full model.

Self-contained: does NOT import from sparsh_enc.py so it avoids the heavy
Sparsh/lightning dependency chain.  CV processing is delegated to
direction_lines.get_line_params_from_diff (single source of truth).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    import torchcurves.functional as tcf
except ImportError:
    tcf = None

from dlofeel_state.models.direction_lines import (
    get_line_params_from_diff,
)


# ---------------------------------------------------------------------------
# Shared utilities (duplicated from sparsh_enc.py to avoid sparsh dependency)
# ---------------------------------------------------------------------------

def get_sinusoidal_pos_encoding(n_positions: int, d_model: int) -> torch.Tensor:
    position = torch.arange(n_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, n_positions, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class PoseBackbone(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_dim = config['model_params']['pose_backbone'].get('hidden_dim', 256)
        latent_dim = config['model_params'].get('latent_dim', 256)
        use_spline = config['model_params'].get('use_spline', False)
        num_output_pts = config["n_control_points"] if use_spline else config["num_output_pts"]
        out_dim = latent_dim * num_output_pts
        dropout = config['dropout']

        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, pose):
        return self.mlp(pose)


class DifferentiableBSpline(nn.Module):
    def __init__(self, n_control_points, n_output_points, degree=3):
        super().__init__()
        self.n_control_points = n_control_points
        self.n_output_points = n_output_points
        self.degree = degree
        t = torch.linspace(-1, 1, n_output_points)
        self.register_buffer('t', t)

    def forward(self, control_points, knots=None):
        if tcf is None:
            raise ImportError("torchcurves is not installed")
        B = control_points.shape[0]
        control_points_fp32 = control_points.float()
        spline_args_fp32 = self.t.unsqueeze(1).expand(-1, B).float().contiguous()
        knots_fp32 = knots.float().contiguous() if knots is not None else None
        curve_points = tcf.bspline_curves(spline_args_fp32, control_points_fp32, knots=knots_fp32, degree=self.degree)
        return curve_points.permute(1, 0, 2).to(control_points.dtype)


class DirectionExtractor(nn.Module):
    """
    Non-differentiable module that extracts DLO direction vectors from
    tactile diff images using classical CV (threshold + line fitting).

    Delegates to ``get_line_params_from_diff`` from direction_lines.py
    so the CV pipeline is identical to the offline processor.

    No normalization is applied — the original pipeline works on raw pixel
    differences.  The training script should pass ``normalize_inputs=False``
    or ensure the images reaching this module are un-normalized diffs.

    Input:  [B, N_digits, 6, H, W]  (temporal pair: channels 0-2 = current RGB diff)
    Output: [B, N_digits, 2]         (unit direction per camera, detached)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.n_digits = config.get("n_digits", 4)

    @torch.no_grad()
    def forward(self, imgs: Tensor) -> Tensor:
        """
        Args:
            imgs: [B, N_digits, 6, H, W] float tensor (diff images, un-normalized).
        Returns:
            [B, N_digits, 2] direction vectors (detached, on same device as imgs).
        """
        B, N, C, H, W = imgs.shape
        current_frames = imgs[:, :, :3, :, :]  # [B, N, 3, H, W]

        frames_np = current_frames.cpu().numpy()

        directions = np.zeros((B, N, 2), dtype=np.float32)
        for b in range(B):
            for d in range(N):
                hwc = np.transpose(frames_np[b, d], (1, 2, 0))  # CHW -> HWC
                line_params = get_line_params_from_diff(hwc)     # [vx, vy, x0, y0]
                directions[b, d] = line_params[:2]               # keep [vx, vy]

        return torch.tensor(directions, dtype=imgs.dtype, device=imgs.device)


# ---------------------------------------------------------------------------
# Direction ablation model
# ---------------------------------------------------------------------------

class DLODirectionModel(nn.Module):
    """
    Ablation model: tactile images -> CV direction extraction -> cross-attention decoder.

    Same decoder architecture as DLOStateModel (with_att=True path), but the
    image backbone is replaced by a frozen CV pipeline + learnable projection.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.n_markers = config["num_output_pts"]
        self.use_spline = config["model_params"].get("use_spline", False)
        self.n_control_pts = config.get("n_control_points", self.n_markers // 2)
        self.spline_degree = config.get("spline_degree", 3)

        min_cp = self.spline_degree + 1
        if self.n_control_pts < min_cp:
            print(f"Warning: n_control_points ({self.n_control_pts}) < spline_degree+1 ({min_cp}), using {min_cp}")
            self.n_control_pts = min_cp

        self.n_out_pts = self.n_control_pts if self.use_spline else self.n_markers
        self.latent_dim = config["model_params"].get("latent_dim", 512)
        self.n_digits = config.get("n_digits", 4)
        dropout = config["dropout"]

        # --- Direction extraction (non-differentiable) ---
        self.direction_extractor = DirectionExtractor(config)

        # --- Direction -> token projection ---
        self.direction_proj = nn.Sequential(
            nn.Linear(2, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        # Camera identity embeddings so the model distinguishes which sensor
        self.digit_embedding = nn.Parameter(
            torch.randn(1, self.n_digits, self.latent_dim) * 0.02
        )

        # --- Pose encoder (same architecture as DLOStateModel) ---
        self.pose_enc = PoseBackbone(config)

        # --- Cross-attention decoder (same as DLOStateModel with_att=True) ---
        num_heads = max(8, self.latent_dim // 32)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(self.latent_dim)

        use_sinusoidal = config["model_params"].get("use_sinusoidal_pos_encoding", True)
        if use_sinusoidal:
            self.register_buffer(
                "marker_pos_encoding",
                get_sinusoidal_pos_encoding(self.n_out_pts, self.latent_dim),
            )
        else:
            self.marker_pos_encoding = nn.Parameter(
                torch.randn(1, self.n_out_pts, self.latent_dim) * 0.02
            )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(self.latent_dim)

        # --- Per-token MLP decoder ---
        self.simple_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.LayerNorm(self.latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim // 2, self.latent_dim // 4),
            nn.LayerNorm(self.latent_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim // 4, 3),
        )

        # --- Optional spline ---
        if self.use_spline:
            self.spline_layer = DifferentiableBSpline(
                n_control_points=self.n_control_pts,
                n_output_points=self.n_markers,
                degree=self.spline_degree,
            )

    def forward(self, pos: Tensor, imgs: Tensor | None = None) -> Tensor:
        B = imgs.shape[0] if imgs is not None else pos.shape[0]

        # 1. Extract directions from images (non-differentiable)
        if imgs is not None:
            directions = self.direction_extractor(imgs)           # [B, 4, 2]
            dir_tokens = self.direction_proj(directions)          # [B, 4, latent_dim]
            dir_tokens = dir_tokens + self.digit_embedding        # add camera identity
        else:
            dir_tokens = torch.zeros(B, self.n_digits, self.latent_dim, device=pos.device, dtype=pos.dtype)

        # 2. Pose -> marker queries
        pos_embed = self.pose_enc(pos)                                       # [B, n_out_pts * latent_dim]
        marker_features = pos_embed.view(B, self.n_out_pts, self.latent_dim) # [B, K, D]
        marker_features = marker_features + self.marker_pos_encoding

        # 3. Cross-attention: marker queries attend to direction tokens
        attn_out, _ = self.cross_attn(
            query=marker_features,   # [B, K, D]
            key=dir_tokens,          # [B, 4, D]
            value=dir_tokens,
        )
        latent = self.cross_attn_norm(marker_features + attn_out)

        # 4. Self-attention among markers
        sa_out, _ = self.self_attn(query=latent, key=latent, value=latent)
        latent = self.self_attn_norm(latent + sa_out)

        # 5. Decode per-token -> [B, K, 3]
        out_points = self.simple_decoder(latent)

        # 6. Optional spline smoothing
        if self.use_spline:
            return self.spline_layer(out_points)
        return out_points


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = {
        "num_output_pts": 32,
        "n_control_points": 32,
        "spline_degree": 3,
        "img_size_H": 320,
        "img_size_W": 240,
        "dropout": 0.1,
        "with_att": True,
        "only_pose": False,
        "n_digits": 4,
        "normalize_inputs": False,
        "model_params": {
            "latent_dim": 128,
            "use_spline": False,
            "use_sinusoidal_pos_encoding": True,
            "pose_backbone": {"hidden_dim": 256},
            "image_backbone": {
                "type": "sparsh_vit",
                "model_type": "vit_small",
                "single_encoder": True,
            },
        },
    }

    model = DLODirectionModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DLODirectionModel trainable parameters: {total_params / 1e6:.4f}M")

    dummy_pos = torch.randn(2, 7, device=device)
    dummy_imgs = torch.randn(2, 4, 6, 320, 240, device=device)

    out = model(dummy_pos, dummy_imgs)
    print("Output shape:", out.shape)
    assert out.shape == (2, 32, 3), f"Unexpected shape: {out.shape}"

    # Test imgs=None fallback
    out_no_img = model(dummy_pos, None)
    print("Output shape (no imgs):", out_no_img.shape)
    assert out_no_img.shape == (2, 32, 3), f"Unexpected shape: {out_no_img.shape}"

    print("All smoke tests PASSED.")
