import torch.nn as nn


class PoseBackbone(nn.Module):
    def __init__(self, config: dict):  # Match ResNet-34 output
        super().__init__()
        self.config = config

        hidden_dim = config['model_params']['pose_backbone'].get('hidden_dim', 256)
        latent_dim = config['model_params'].get('latent_dim', 256)
        use_spline = config['model_params'].get('use_spline', False)
        if use_spline:
            num_output_pts = config["n_control_points"]
        else:
            num_output_pts = config["num_output_pts"]

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

            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, pose):
        return self.mlp(pose)  # [B, out_dim]
