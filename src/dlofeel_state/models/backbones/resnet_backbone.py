import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models


class ImgBackbone(nn.Module):
    """
    ResNet18-based image backbone that preserves spatial tokens (like ViT).
    Output shape: [B, N_digits, Num_Tokens, latent_dim]
    """
    def __init__(self, config: dict, out_dim=1024, n_digits=4):
        super().__init__()
        self.config = config

        self.out_dim = config['model_params'].get('latent_dim', out_dim)
        self.n_digits = config.get('n_digits', n_digits)
        self.img_size_H = config['img_size_H']
        self.img_size_W = config['img_size_W']
        self.single_encoder = config['model_params']['image_backbone']["single_encoder"]

        # ResNet layer2 outputs 128 channels at /8 spatial resolution
        self.feature_channels = 128
        # Calculate number of spatial tokens (like ViT patches)
        self.num_tokens = (self.img_size_H // 8) * (self.img_size_W // 8)

        if self.single_encoder:
            print(f"Using SINGLE ResNet encoder for all digit images. Tokens per image: {self.num_tokens}")
            resnet = models.resnet18(weights=None)
            self.digit_encoder = self._build_digit_encoder(resnet)
            self.proj_fc = nn.Linear(self.feature_channels, self.out_dim)
        else:
            print(f"Using ENSEMBLE of ResNet encoders for digit images. Tokens per image: {self.num_tokens}")
            self.digit_encoders = nn.ModuleList()
            self.proj_fcs = nn.ModuleList()
            for _ in range(self.n_digits):
                resnet = models.resnet18(weights=None)
                self.digit_encoders.append(self._build_digit_encoder(resnet))
                self.proj_fcs.append(nn.Linear(self.feature_channels, self.out_dim))

    def _build_digit_encoder(self, resnet: nn.Module) -> nn.Sequential:
        # Modify first conv to accept 6 channels (temporal pair)
        new_conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=resnet.conv1.bias is not None
        )
        resnet.conv1 = new_conv1
        # Use up to layer2 for /8 downsampling
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # /4 downsampling
            resnet.layer1,   # /4 (no stride change)
            resnet.layer2,   # /8 downsampling total, 128 channels
        )

    def forward(self, digit_images: Tensor) -> Tensor:
        """
        Args:
            digit_images: [B, N_digits, C=6, H, W] temporal pair images
        Returns:
            [B, N_digits, Num_Tokens, latent_dim] - spatial tokens like ViT
        """
        B, N, C, H, W = digit_images.shape

        if self.single_encoder:
            # Batch all digits together
            x = digit_images.view(B * N, C, H, W)
            feats = self.digit_encoder(x)  # [B*N, 128, H/8, W/8]

            # Reshape to tokens: [B*N, 128, H/8, W/8] -> [B*N, num_tokens, 128]
            feats = feats.flatten(2).transpose(1, 2)  # [B*N, num_tokens, 128]

            # Project to latent dim
            feats = self.proj_fc(feats)  # [B*N, num_tokens, latent_dim]

            # Reshape to [B, N_digits, num_tokens, latent_dim]
            return feats.view(B, N, self.num_tokens, self.out_dim)
        else:
            tactile_features = []
            for i in range(N):
                feat = self.digit_encoders[i](digit_images[:, i])  # [B, 128, H/8, W/8]
                feat = feat.flatten(2).transpose(1, 2)  # [B, num_tokens, 128]
                feat = self.proj_fcs[i](feat)  # [B, num_tokens, latent_dim]
                tactile_features.append(feat)

            return torch.stack(tactile_features, dim=1)  # [B, N_digits, num_tokens, latent_dim]
