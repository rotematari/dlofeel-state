import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
from torch import Tensor


class ImgBackbone(nn.Module):
    def __init__(self, config: dict, out_dim=1024,n_digits=4):
        super().__init__()
        # replace with SPARSH later
        # self.digit_encoder = models.resnet18(replace_stride_with_dilation=(False, False, False), num_classes=out_dim) # layer3/layer4 dilated)
        self.config = config
        
        out_dim = config['model_params'].get('latent_dim', out_dim)
        n_digits = config.get('n_digits', n_digits)
        self.img_size = config['img_size']
        self.single_encoder = config['model_params']['image_backbone']["single_encoder"]
        
        if self.single_encoder:
            print("Using SINGLE encoder for all digit images.")
            resnet = models.resnet18(num_classes=out_dim)
            self.digit_encoder = self._build_digit_encoder(resnet)
            self.bottleneck = self._build_bottleneck()
            self.proj_fc = self._build_proj_fc(out_dim)
        else:
            print("Using ENSEMBLE of encoders for digit images.")
            
            self.digit_encoders = nn.ModuleList()
            self.bottlenecks = nn.ModuleList()
            self.proj_fcs = nn.ModuleList()
            for _ in range(n_digits):
                resnet = models.resnet18(num_classes=out_dim)
                self.digit_encoders.append(self._build_digit_encoder(resnet))
                self.bottlenecks.append(self._build_bottleneck())
                self.proj_fcs.append(self._build_proj_fc(out_dim))
            # self.digit_encoders = nn.ModuleList([
            #     models.resnet18(num_classes=out_dim) for _ in range(n_digits)
            # ])

    def _build_digit_encoder(self, resnet: nn.Module) -> nn.Sequential:
        # Drop fc and avgpool
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, # /4 downsampling so far
            resnet.layer1,  # /4 (no stride change)
            resnet.layer2,  # /8 downsampling total
            # resnet.layer3,  # /16 downsampling total
        )

    def _build_bottleneck(self) -> nn.Conv2d:
        return nn.Conv2d(128, 32, kernel_size=1)

    def _build_proj_fc(self, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.img_size//8 * self.img_size//8 * 32, out_dim)
        )

    def _encode_digits(self, feats: Tensor, proj_fc: nn.Module) -> Tensor:
        feats = self.bottleneck(feats)      # [B*N, 32, H/8, W/8]
        feats = feats.flatten(-2)           # [B*N, 32, H*W]
        feats = feats.flatten(1)            # [B*N, 32*H*W]
        return proj_fc(feats)               # [B*N, out_dim]

    def forward(self, digit_images):  # x: [B,3,H,W]
        if self.single_encoder:
            B,N,C,H,W = digit_images.shape # [B,num_digits,C,H,W]
            x = digit_images.view(B*N, C, H, W)  # [B*num_digits, C, H, W]
            feats = self.digit_encoder(x)  # [B*N, 128, H/8, W/8]
            feats = self._encode_digits(feats, self.proj_fc)
            return feats.view(B, N, -1)  # [B, num_digits, out_dim]
        else:
            B,N,C,H,W = digit_images.shape # [B,num_digits,C,H,W]
            
            tactile_features = []
            for i in range(N):
                feat = self.digit_encoders[i](digit_images[:, i])  # [B, 128, H/8, W/8]
                feat = self.bottlenecks[i](feat)  # [B, 32, H/8, W/8]
                feat = feat.flatten(-2)  # [B, 32, H*W]
                feat = self.proj_fcs[i](feat.flatten(1))  # [B, out_dim]
                tactile_features.append(feat)
            
            return torch.stack(tactile_features, dim=1)  # [B, num_digits, out_dim]
    
    
class PoseBackbone(nn.Module):
    def __init__(self, config: dict):  # Match ResNet-34 output
        super().__init__()
        self.config = config
        
        hidden_dim = config['model_params']['pose_backbone'].get('hidden_dim', 256)
        latent_dim = config['model_params'].get('latent_dim', 512)
        num_output_pts = config['num_output_pts']
        out_dim = latent_dim
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
class DLOStateModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.n_markers = config['num_output_pts']
        self.latent_dim = config['model_params'].get('latent_dim', 512)
        self.n_digits = config['model_params'].get('n_digits', 4)
        self.with_att = config['with_att']
        self.only_pose = config['only_pose']
        dropout = config['dropout']
        
        # Encoders
        if not self.only_pose:
            self.img_enc = ImgBackbone(config, n_digits=self.n_digits)
        
        self.pose_enc = PoseBackbone(config)

        # --- Decoder Logic ---
        final_output_dim = 3 * self.n_markers # Default for MLP mode

        if self.with_att:
            # ATTENTION MODE
            # We process [B, N, latent], so the Linear layer acts on 'latent'
            # Each token predicts ITSELF (x,y,z), so output is just 3
            new_latent_dim = self.latent_dim
            final_output_dim = 3 

            if not self.only_pose:
                # FUSION: Decoder (Query=Markers, Key=Image)
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=self.latent_dim,
                    nhead=8,
                    dim_feedforward=4*self.latent_dim,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
                
                # Positional encoding
                self.marker_pos_encoding = nn.Parameter(
                    torch.randn(1, self.n_markers, self.latent_dim) * 0.02
                )
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
                # You might want Positional Encoding here too for the Pose Only mode!
                self.marker_pos_encoding = nn.Parameter(
                    torch.randn(1, self.n_markers, self.latent_dim) * 0.02
                )

        else:
            # MLP MODE (No Attention)
            # We flatten everything: Input is [B, (N+1)*latent]
            # Output is ALL markers [B, 3*N]
            if self.only_pose:
                new_latent_dim = self.latent_dim 
            else:
                new_latent_dim = (self.n_digits + 1) * self.latent_dim
            
        self.simple_decoder = nn.Sequential(
            nn.Linear(new_latent_dim, new_latent_dim//2),
            nn.LayerNorm(new_latent_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(new_latent_dim//2, new_latent_dim//4),
            nn.LayerNorm(new_latent_dim//4),
            nn.ReLU(),
            # FIX: Use the calculated final_output_dim
            nn.Linear(new_latent_dim//4, final_output_dim)
        )
            
    def forward(self, pos, imgs=None):
        B = imgs.shape[0] if imgs is not None else pos.shape[0]
        
        # 1. Encode inputs
        pos_embed = self.pose_enc(pos)  # [B, latent_dim]
        if not self.only_pose:
            img_feat = self.img_enc(imgs)   # [B, num_digits, latent_dim]
        
        if self.with_att:
            # Initialize queries
            marker_features = pos_embed.unsqueeze(1).repeat(1, self.n_markers, 1) # [B, N, latent]
            
            # Add positional encoding (Important for both cases!)
            marker_features = marker_features + self.marker_pos_encoding

            if not self.only_pose:
                latent_features = self.transformer_decoder(
                    tgt=marker_features, 
                    memory=img_feat
                ) # [B, N, latent]
            else:
                latent_features = self.transformer_encoder(
                    marker_features
                ) # [B, N, latent]
            
            # Prediction
            # Input: [B, N, latent] -> Decoder -> [B, N, 3]
            marker_positions = self.simple_decoder(latent_features)
            
        else:
            # MLP Mode
            if not self.only_pose:
                latent_features = torch.cat((img_feat.flatten(1), pos_embed), dim=1)
            else:
                latent_features = pos_embed
            
            # Input: [B, Big_Dim] -> Decoder -> [B, 3*N]
            marker_positions = self.simple_decoder(latent_features)
            # Reshape to [B, N, 3]
            marker_positions = marker_positions.view(B, self.n_markers, 3)

        return marker_positions


# class DLOStateModel(nn.Module):
#     def __init__(self, config: dict):
#         super().__init__()
#         self.config = config
#         self.n_markers = config['num_output_pts']
#         self.latent_dim = config['model_params'].get('latent_dim', 512)
#         self.n_digits = config['model_params'].get('n_digits', 4)
#         self.with_att = config['with_att']
#         self.only_pose = config['only_pose']
#         dropout = config['dropout']
        
#         # Encoders
#         if not self.only_pose:
#             self.img_enc = ImgBackbone(config, n_digits=self.n_digits)
        
#         self.pose_enc = PoseBackbone(config)
#         # Decoder
#         if self.with_att:

#             if not self.only_pose:
#                 decoder_layer = nn.TransformerDecoderLayer(
#                 d_model=self.latent_dim,
#                 nhead=8,
#                 dim_feedforward=4*self.latent_dim,
#                 dropout=dropout,
#                 batch_first=True
#                 )
#                 self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
#                 # Add positional encoding for markers
#                 self.marker_pos_encoding = nn.Parameter(
#                     torch.randn(1, self.n_markers, self.latent_dim) * 0.02
#                 )
#             else:
#                 # transformer encoder only on pose
#                 encoder_layer = nn.TransformerEncoderLayer(
#                 d_model=self.latent_dim,
#                 nhead=8,
#                 dim_feedforward=4*self.latent_dim,
#                 dropout=dropout,
#                 batch_first=True
#                 )
#                 self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         ## if with no attention
        
#             new_latent_dim = self.latent_dim
#         else:
#             new_latent_dim = (self.n_digits + 1)*self.latent_dim
#         if self.only_pose:
#             new_latent_dim = self.latent_dim 
            
#         self.simple_decoder = nn.Sequential(
#             nn.Linear(new_latent_dim, new_latent_dim//2),
#             nn.LayerNorm(new_latent_dim//2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(new_latent_dim//2, new_latent_dim//4),
#             nn.LayerNorm(new_latent_dim//4),
#             nn.ReLU(),
#             nn.Linear(new_latent_dim//4, 3*self.n_markers))
            
#     def forward(self, pos, imgs = None):  # imgs:[B,n_digits,3,H,W]
#         B = imgs.shape[0] if imgs is not None else pos.shape[0]
        
#         # 1. Encode inputs
#         pos_embed = self.pose_enc(pos)  # [B, latent_dim]
#         if not self.only_pose:
#             img_feat = self.img_enc(imgs)   # [B, num_digits, out_dim]
        
        
        
#         if self.with_att:
#             # 2. Initialize marker features from end-effector pose
#             # marker_features = self.marker_init(pos_embed)  # [B, n_markers * latent_dim]
#             marker_features = pos_embed.repeat(1, self.n_markers)  # [B, n_markers * latent_dim]
#             marker_features = marker_features.view(B, self.n_markers, self.latent_dim)  # [B, n_markers, latent_dim]
#             # 3. Add positional encoding
#             marker_features = marker_features + self.marker_pos_encoding # [B, n_markers, latent_dim]
#             if not self.only_pose:
#                 # The decoder handles BOTH Self-Attention (inter-marker) and Cross-Attention (marker-to-image)
#                 # tgt = Queries (Markers), memory = Keys (Images)
#                 latent_features = self.transformer_decoder(
#                     tgt=marker_features, 
#                     memory=img_feat
#                 ) # [B, n_markers, latent_dim]
#             else:
                
#                 latent_features = self.transformer_encoder(
#                     marker_features
#                 ) # [B, n_markers, latent_dim]
#         else:
#             if not self.only_pose:
#                 latent_features = torch.cat((img_feat.flatten(1), pos_embed), dim=1) # [B, (n_digits +1)*latent_dim]
#             else:
#                 latent_features = pos_embed # [B, latent_dim]
                
#         marker_positions = self.simple_decoder(latent_features).view(B, self.n_markers, 3)  # [B, n_markers, 3]

#         return marker_positions



if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


