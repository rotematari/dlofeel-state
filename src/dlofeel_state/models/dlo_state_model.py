import math

import torch
import torch.nn as nn

from dlofeel_state.models.backbones.pose_encoder import PoseBackbone
from dlofeel_state.models.backbones.resnet_backbone import ImgBackbone
from dlofeel_state.models.backbones.sparsh_backbone import SparshImgBackbone
from dlofeel_state.models.bspline import DifferentiableBSpline


def get_sinusoidal_pos_encoding(n_positions: int, d_model: int) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings.

    Args:
        n_positions: Number of positions (e.g., number of markers)
        d_model: Embedding dimension

    Returns:
        Tensor of shape [1, n_positions, d_model]
    """
    position = torch.arange(n_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe = torch.zeros(1, n_positions, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)

    return pe


class DLOStateModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.n_markers = config['num_output_pts']
        self.use_spline = config['model_params'].get('use_spline', False)
        self.n_control_pts = config.get('n_control_points', self.n_markers // 2)
        self.spline_degree = config.get('spline_degree', 3)

        min_control_pts = self.spline_degree + 1
        if self.n_control_pts < min_control_pts:
            print(f"Warning: n_control_points ({self.n_control_pts}) < spline_degree+1 ({min_control_pts}), using {min_control_pts}")
            self.n_control_pts = min_control_pts

        print(f"Spline config: {self.n_control_pts} control points -> {self.n_markers} output points (degree {self.spline_degree})")

        if self.use_spline:
            self.n_out_pts = self.n_control_pts
        else:
            self.n_out_pts = self.n_markers

        self.latent_dim = config['model_params'].get('latent_dim', 512)
        self.n_digits = config['model_params'].get('n_digits', 4)
        self.with_att = config['with_att']
        self.only_pose = config['only_pose']
        dropout = config['dropout']



        # Encoders
        if not self.only_pose:
            backbone_type = config['model_params']['image_backbone'].get("type", "resnet18")
            print("Image Backbone type:", backbone_type)
            if backbone_type == "sparsh_vit":
                self.img_enc = SparshImgBackbone(config, n_digits=self.n_digits)
            else:
                self.img_enc = ImgBackbone(config, n_digits=self.n_digits)

        self.pose_enc = PoseBackbone(config)

        # --- Decoder Logic ---
        final_output_dim = 3 * self.n_out_pts # Predict output points


        if self.with_att:
            # ATTENTION MODE
            # We process [B, N, latent], so the Linear layer acts on 'latent'
            # Each token predicts ITSELF (x,y,z), so output is just 3
            new_latent_dim = self.latent_dim
            final_output_dim = 3

            if not self.only_pose:
                # FUSION: Single Cross-Attention (Query=Markers, Key/Value=Image)
                # Using 12 heads for latent_dim=384 (32 dims per head)
                # Rule of thumb: head_dim = latent_dim / num_heads should be 32-64
                num_heads = max(8, self.latent_dim // 32)  # 12 for 384, 16 for 512
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=self.latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                self.cross_attn_norm = nn.LayerNorm(self.latent_dim)

                # Positional encoding for marker queries
                self.use_sinusoidal_pos = config['model_params'].get('use_sinusoidal_pos_encoding', True)
                if self.use_sinusoidal_pos:
                    print("Using SINUSOIDAL positional encoding for marker queries")
                    # Register as buffer (not a parameter, but moves with model to device)
                    self.register_buffer(
                        'marker_pos_encoding',
                        get_sinusoidal_pos_encoding(self.n_out_pts, self.latent_dim)
                    )
                else:
                    print("Using LEARNED positional encoding for marker queries")
                    self.marker_pos_encoding = nn.Parameter(
                        torch.randn(1, self.n_out_pts, self.latent_dim) * 0.02
                    )

                # Digit-specific embedding (distinguishes which camera)
                self.digit_embedding = nn.Parameter(torch.randn(1, self.n_digits, 1, self.latent_dim))
                self.self_attn = nn.MultiheadAttention(
                    embed_dim=self.latent_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                self.self_attn_norm = nn.LayerNorm(self.latent_dim)

            else:
                # POSE ONLY: Encoder (Self-Attention on Markers)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=8,
                    dim_feedforward=4*self.latent_dim,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

                # Positional encoding for pose-only mode
                self.use_sinusoidal_pos = config['model_params'].get('use_sinusoidal_pos_encoding', True)
                if self.use_sinusoidal_pos:
                    self.register_buffer(
                        'marker_pos_encoding',
                        get_sinusoidal_pos_encoding(self.n_out_pts, self.latent_dim)
                    )
                else:
                    self.marker_pos_encoding = nn.Parameter(
                        torch.randn(1, self.n_out_pts, self.latent_dim) * 0.02
                    )

        else:
            # MLP MODE (No Attention)
            # After your reshape changes: img_feat is now [B, n_patches, embed_dim]
            # So we flatten to [B, n_patches * embed_dim] and concat with pose [B, latent_dim]
            if self.only_pose:
                new_latent_dim = self.latent_dim
            else:
                # Calculate number of patches based on image size (img_size / patch_size)^2
                img_H = config.get('img_size_H', 224)
                img_W = config.get('img_size_W', 224)
                patch_size = 16  # ViT patch size
                n_patches = (img_H // patch_size) * (img_W // patch_size)
                # img_feat: [B, n_patches, embed_dim] -> flatten -> [B, n_patches * embed_dim]
                # concat with pose: [B, n_patches * embed_dim + latent_dim]
                new_latent_dim = n_patches * self.embed_dim + self.latent_dim
                print(f"MLP mode: n_patches={n_patches}, embed_dim={self.embed_dim}, input_dim={new_latent_dim}")

        # Per-marker decoder (used in MLP mode or when use_global_decoder=False)
        self.simple_decoder = nn.Sequential(
            nn.Linear(new_latent_dim, new_latent_dim//2),
            nn.LayerNorm(new_latent_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(new_latent_dim//2, new_latent_dim//4),
            nn.LayerNorm(new_latent_dim//4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(new_latent_dim//4, final_output_dim)
        )
        if self.use_spline:
            # Differentiable B-spline layer
            self.spline_layer = DifferentiableBSpline(
                n_control_points=self.n_control_pts,
                n_output_points=self.n_markers,
                degree=self.spline_degree
        )

    def forward(self, pos, imgs=None):
        B = imgs.shape[0] if imgs is not None else pos.shape[0]

        # 1. Encode inputs
        pos_embed = self.pose_enc(pos)  # [B, latent_dim]
        if not self.only_pose and imgs is not None:
            img_feat = self.img_enc(imgs)   # [B, n_digits, n_patches, latent_dim]
            # Normalize backbone outputs to [B, N_tokens, latent_dim] for attention.
            # Sparsh may already return 3D, while ResNet path returns 4D.
            if img_feat.dim() == 4:
                img_feat = img_feat.flatten(1, 2)

        if self.with_att:
            # Initialize queries
            # marker_features = pos_embed.unsqueeze(1).repeat(1, self.n_markers, 1) # [B, N, latent]
            marker_features = pos_embed.view(B, self.n_out_pts, self.latent_dim)
            marker_features = marker_features + self.marker_pos_encoding

            if not self.only_pose and imgs is not None:
                # Single cross-attention: markers query image features
                attn_out, _ = self.cross_attn(
                    query=marker_features,  # [B, N_out_pts, latent]
                    key=img_feat,           # [B, N_patches, latent]
                    value=img_feat
                )
                # Residual connection + LayerNorm
                latent_features = self.cross_attn_norm(marker_features + attn_out)  # [B, N, latent]
                self_attn_out,  _ = self.self_attn(
                    query=latent_features,
                    key=latent_features,
                    value=latent_features
                )
                latent_features = self.self_attn_norm(latent_features + self_attn_out)
            else:
                latent_features = self.transformer_encoder(
                    marker_features
                ) # [B, N, latent]
            # Prediction
            # Output: [B, n_out_pts * 3]

            out_points_flat = self.simple_decoder(latent_features)
            out_points = out_points_flat.view(B, self.n_out_pts, 3)
        else:
            # MLP Mode
            if not self.only_pose and imgs is not None:
                latent_features = torch.cat((img_feat.flatten(1), pos_embed), dim=1)
            else:
                latent_features = pos_embed
            # Input: [B, Big_Dim] -> Decoder -> [B, 3*N_out_pts]
            out_points_flat = self.simple_decoder(latent_features)
            # Reshape to [B, N_out_pts, 3]
            out_points = out_points_flat.view(B, self.n_out_pts, 3)

        # Apply Spline Layer
        # [B, N_out_pts, 3] -> [B, N_markers, 3]
        if self.use_spline:
            marker_positions = self.spline_layer(out_points)
        else:
            marker_positions = out_points


        return marker_positions
