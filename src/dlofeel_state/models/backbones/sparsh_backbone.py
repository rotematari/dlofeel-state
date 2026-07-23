from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from dlofeel_state.models.backbones._sparsh_import import (
    VisionTransformer,
    vit_base,
    vit_small,
    vit_tiny,
)


class SparshImgBackbone(nn.Module):
    def __init__(self, config: dict, out_dim=1024, n_digits=4):
        super().__init__()
        self.config = config
        out_dim = config['model_params'].get('latent_dim', out_dim)

        n_digits = config.get('n_digits', n_digits)
        self.image_size_H = config['img_size_H']
        self.image_size_W = config['img_size_W']
        self.n_digits = n_digits
        self.patch_size = 16
        self.num_patches = (self.image_size_H // self.patch_size) * (self.image_size_W // self.patch_size)  # Assuming patch_size=16
        self.single_encoder = config['model_params']['image_backbone']["single_encoder"]

        model_type = config['model_params']['image_backbone'].get("model_type", "vit_small")
        pretrained_weights_path = config['model_params']['image_backbone'].get(
            "pretrained_weights_path"
        )
        configured_pos_embed_fn = config['model_params']['image_backbone'].get(
            "pos_embed_fn", "auto"
        )
        if configured_pos_embed_fn not in {"auto", "sinusoidal", "learned"}:
            raise ValueError(
                f"Invalid pos_embed_fn='{configured_pos_embed_fn}'. "
                "Use one of: 'auto', 'sinusoidal', 'learned'."
            )
        self._pos_embed_fn_is_auto = configured_pos_embed_fn == "auto"
        self.backbone_pos_embed_fn: Literal["sinusoidal", "learned"]
        if configured_pos_embed_fn == "auto":
            if not pretrained_weights_path:
                # No checkpoint to infer from: keep the ViT default behavior.
                self.backbone_pos_embed_fn = "learned"
                print("Backbone pos_embed_fn=auto but no checkpoint provided; defaulting to 'learned'.")
            else:
                self.backbone_pos_embed_fn = self._infer_checkpoint_pos_embed_fn(pretrained_weights_path)
                print(
                    f"Backbone pos_embed_fn=auto -> inferred '{self.backbone_pos_embed_fn}' "
                    f"from checkpoint {pretrained_weights_path}."
                )
        else:
            self.backbone_pos_embed_fn = configured_pos_embed_fn


        def build_backbone()-> tuple[VisionTransformer, int]:
            if model_type == 'vit_small':
                backbone = vit_small(
                    patch_size=self.patch_size,
                    num_register_tokens=1,
                    in_chans=6,
                    pos_embed_fn=self.backbone_pos_embed_fn,
                )
                embed_dim = 384
            elif model_type == 'vit_base':
                backbone = vit_base(
                    patch_size=self.patch_size,
                    num_register_tokens=1,
                    in_chans=6,
                    pos_embed_fn=self.backbone_pos_embed_fn,
                )
                embed_dim = 768
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            return backbone, embed_dim

        # Config for partial unfreezing
        self.unfreeze_last_n_blocks = config['model_params']['image_backbone'].get('unfreeze_last_n_blocks', 0)
        # Config for multi-scale fusion
        self.multi_scale_fusion = config['model_params']['image_backbone'].get('multi_scale_fusion', False)
        fusion_layers_cfg = config['model_params']['image_backbone'].get('fusion_layers', [6, 12])  # Default: layer 6 and 12
        # Ensure last layer (12) is always included
        self.fusion_layers = sorted(set(fusion_layers_cfg) | {12})
        # Only one layer (last) => no multi-scale; disable fusion
        if self.fusion_layers == [12]:
            self.multi_scale_fusion = False
        # Fusion method: 'concat' (DINOv2-style) or 'learned_sum' (learned weighted sum)
        self.fusion_method = config['model_params']['image_backbone'].get('fusion_method', 'learned_sum')

        if self.single_encoder:
            print("Using SINGLE Sparsh encoder for all digit images.")
            self.backbone, self.embed_dim = build_backbone()
            if pretrained_weights_path:
                self._load_weights(self.backbone, pretrained_weights_path)
                self._apply_freezing_strategy(self.backbone)

            # Multi-scale fusion layers
            if self.multi_scale_fusion:
                if self.fusion_method == 'concat':
                    # DINOv2-style: concatenate + project
                    print(f"Multi-scale fusion enabled with layers: {self.fusion_layers} (concat + projection)")
                    fused_dim = self.embed_dim * len(self.fusion_layers)
                    self.fusion_proj = nn.Linear(fused_dim, self.embed_dim)
                elif self.fusion_method == 'learned_sum':
                    # Learned weighted sum (attentive probing)
                    print(f"Multi-scale fusion enabled with layers: {self.fusion_layers} (learned weighted sum)")
                    self.layer_weights = nn.Parameter(torch.ones(len(self.fusion_layers)))
                else:
                    raise ValueError(f"Unknown fusion_method: {self.fusion_method}. Use 'concat' or 'learned_sum'")

            self.proj_fc = self._build_proj_fc(out_dim, self.embed_dim)
        else:
            print("Using ENSEMBLE of Sparsh encoders for digit images.")
            self.backbones = nn.ModuleList()
            self.proj_fcs = nn.ModuleList()
            for i in range(self.n_digits):
                backbone, embed_dim = build_backbone()
                if i == 0:
                    self.embed_dim = embed_dim  # Store embed_dim for _encode_batch
                if pretrained_weights_path:
                    self._load_weights(backbone, pretrained_weights_path)
                self.backbones.append(backbone)
                self.proj_fcs.append(self._build_proj_fc(out_dim, embed_dim))

    def _build_proj_fc(self, out_dim: int, embed_dim: int) -> nn.Module:
        if embed_dim == out_dim:
            return nn.Identity()
        return nn.Linear(embed_dim, out_dim)

    def _apply_freezing_strategy(self, backbone: VisionTransformer) -> None:
        """Apply freezing strategy to backbone based on config.

        If unfreeze_last_n_blocks > 0, freezes all parameters except the last N blocks.
        Otherwise, freezes all backbone parameters.
        """
        # First, freeze everything
        for p in backbone.parameters():
            p.requires_grad = False

        if self.unfreeze_last_n_blocks > 0:
            # Get total number of blocks
            num_blocks = len(backbone.blocks)
            unfreeze_from = max(0, num_blocks - self.unfreeze_last_n_blocks)

            print(f"Partial freeze: Unfreezing last {self.unfreeze_last_n_blocks} blocks (blocks {unfreeze_from} to {num_blocks-1})")

            # Unfreeze last N blocks
            for i in range(unfreeze_from, num_blocks):
                for p in backbone.blocks[i].parameters():
                    p.requires_grad = True

            # Always unfreeze the final norm layer
            if hasattr(backbone, 'norm'):
                for p in backbone.norm.parameters():
                    p.requires_grad = True

            # Count trainable params
            trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            total = sum(p.numel() for p in backbone.parameters())
            print(f"Backbone: {trainable:,} / {total:,} parameters trainable ({100*trainable/total:.1f}%)")
        else:
            print("Freezing all Sparsh backbone weights.")

    @staticmethod
    def _infer_checkpoint_pos_embed_fn(path: str) -> Literal["sinusoidal", "learned"]:
        """Infer checkpoint positional embedding style from backbone keys."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']
        has_sinusoidal = False
        has_learned = False
        for key in state_dict:
            if key == "student_encoder.backbone.pos_embed":
                has_learned = True
            elif key.startswith("student_encoder.backbone.pos_embed."):
                has_sinusoidal = True
        if has_sinusoidal and not has_learned:
            return "sinusoidal"
        if has_learned and not has_sinusoidal:
            return "learned"
        raise ValueError(
            "Could not uniquely infer checkpoint pos_embed type. "
            f"Found learned={has_learned}, sinusoidal={has_sinusoidal} in {path}."
        )

    def _load_weights(self, backbone: VisionTransformer, path: str) -> None:
        print(f"Loading weights from {path}...")
        checkpoint = torch.load(path, map_location='cpu',weights_only=False)
        state_dict = checkpoint['model']


        backbone_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('student_encoder.backbone.'):
                new_k = k.replace('student_encoder.backbone.', '')
                backbone_state_dict[new_k] = v

        ckpt_pos_embed_fn = "sinusoidal" if "pos_embed.frequency_bands" in backbone_state_dict else "learned"
        model_pos_embed_fn = "learned" if isinstance(backbone.pos_embed, nn.Parameter) else "sinusoidal"
        if ckpt_pos_embed_fn != model_pos_embed_fn and ckpt_pos_embed_fn == "auto":
            raise ValueError(
                "Backbone pos_embed mismatch between checkpoint and model. "
                f"Checkpoint='{ckpt_pos_embed_fn}', model='{model_pos_embed_fn}'. "
                "Set model_params.image_backbone.pos_embed_fn to match checkpoint (or use 'auto')."
            )

        msg = backbone.load_state_dict(backbone_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}")
        missing_pos_embed_keys = [
            key for key in msg.missing_keys
            if key == "pos_embed" or key.startswith("pos_embed.")
        ]
        if missing_pos_embed_keys:
            if self._pos_embed_fn_is_auto:
                raise ValueError(
                    "Checkpoint load missed positional embedding keys: "
                    f"{missing_pos_embed_keys}. This indicates an incompatible "
                    "or incomplete backbone positional embedding load."
                )
            else:
                print(
                    f"[WARNING] Missing pos_embed keys from checkpoint: {missing_pos_embed_keys}. "
                    f"Allowed because pos_embed_fn was explicitly set to '{self.backbone_pos_embed_fn}'."
                )

        # CRITICAL CHECK:
        embed_weight = backbone.patch_embed.proj.weight
        if 'patch_embed.proj.weight' in msg.missing_keys:
            print("\n[CRITICAL WARNING] The input projection weights were NOT loaded!")
            print("Your model is processing images with a RANDOM initialization.")
        elif embed_weight.shape[1] == 6:
            print(f"\n[INFO] Loaded input weights with shape {embed_weight.shape}. (Check if these are just RGB copied or real 6-channel weights)")
        print(f"Loaded weights into Sparsh backbone from {path}.")

    def get_layer_weights(self) -> dict:
        """Get the current learned layer weights (only for 'learned_sum' fusion method).

        Returns dict with layer indices and their normalized weights (after softmax).
        Useful for understanding which layers the model finds most useful.
        """
        if not self.multi_scale_fusion or self.fusion_method != 'learned_sum':
            return {}

        with torch.no_grad():
            weights = torch.softmax(self.layer_weights, dim=0).cpu().numpy()

        return {f"layer_{layer}": float(w) for layer, w in zip(self.fusion_layers, weights)}

    def _encode_batch(self, backbone: VisionTransformer, proj_fc: nn.Module, x: Tensor) -> Tensor:
        """Standard encoding without multi-scale fusion."""
        features = backbone(x)
        features = features.reshape(x.shape[0]//self.n_digits,-1,self.embed_dim)   # [B,N, Num_Tokens, Dim]
        return proj_fc(features)

    def _encode_batch_multiscale(self, backbone: VisionTransformer, x: Tensor) -> Tensor:
        """Multi-scale encoding using backbone's get_intermediate_layers method.

        Supports two fusion methods:
        - 'concat': DINOv2-style concatenation of layer features + linear projection
        - 'learned_sum': Learned weighted sum (attentive probing) of layer features
        """
        B = x.shape[0]

        # Convert fusion_layers from 1-indexed to 0-indexed for backbone
        # Config uses 1-indexed (e.g., [4, 8, 12] for layers 4, 8, 12)
        # get_intermediate_layers uses 0-indexed (e.g., [3, 7, 11])
        layer_indices = [l - 1 for l in self.fusion_layers]

        # Get intermediate layer outputs - this handles pos_embed interpolation internally
        layer_outputs = backbone.get_intermediate_layers(x, n=layer_indices, norm=True)
        # layer_outputs is a tuple of tensors, each [B, Tokens, embed_dim]

        if self.fusion_method == 'concat':
            # DINOv2-style: concatenate + project
            fused_features = torch.cat(layer_outputs, dim=-1)  # [B, Tokens, embed_dim * len(fusion_layers)]
            fused_features = self.fusion_proj(fused_features)  # [B, Tokens, embed_dim]
        elif self.fusion_method == 'learned_sum':
            # Learned weighted sum (attentive probing)
            # Apply softmax to get normalized weights
            weights = torch.softmax(self.layer_weights, dim=0)
            # Weighted sum of layer outputs
            fused_features = sum(w * layer for w, layer in zip(weights, layer_outputs))  # [B, Tokens, embed_dim]

        # Reshape to [B, N_digits*Tokens, embed_dim]
        fused_features = fused_features.reshape(x.shape[0]//self.n_digits,-1,self.embed_dim)

        return self.proj_fc(fused_features)

    def forward(self, digit_images: Tensor) -> Tensor:
        B, N, C, H, W = digit_images.shape
        if self.single_encoder:
            x = digit_images.view(B * N, C, H, W)

            if self.multi_scale_fusion:
                feats = self._encode_batch_multiscale(self.backbone, x)
            else:
                feats = self._encode_batch(self.backbone, self.proj_fc, x)
            return feats

        tactile_features = []
        for i in range(N):
            if self.multi_scale_fusion:
                feats = self._encode_batch_multiscale(self.backbone, digit_images[:, i])
            else:
                feats = self._encode_batch(self.backbone, self.proj_fc, digit_images[:, i])
            tactile_features.append(feats)
        return torch.stack(tactile_features, dim=1)
