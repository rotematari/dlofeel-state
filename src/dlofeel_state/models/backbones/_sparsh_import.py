"""Single import choke point for the Sparsh submodule (``third_party/sparsh``).

Every backbone that needs a Sparsh ViT imports its builders from here, so all
submodule wiring lives in one file:

    from dlofeel_state.models.backbones._sparsh_import import (
        vit_small, vit_base, vit_tiny, VisionTransformer,
    )

Sparsh is CC-BY-NC 4.0. It is consumed as a pinned git submodule and never
redistributed inside this package; only import glue lives here.

Two non-obvious things this module does:

1. Upstream's importable package is ``tactile_ssl`` at the *submodule root*
   (a PEP 420 namespace package), not ``sparsh.tactile_ssl``. We put
   ``third_party/sparsh`` on ``sys.path`` and import ``tactile_ssl`` directly,
   and keep the ``sys.modules['tactile_ssl']`` alias so checkpoints pickled with
   references under ``tactile_ssl.*`` still resolve.

2. The pinned submodule is NOT behaviorally identical to the copy the original
   models were trained against. The licensing audit found exactly one functional
   local edit in the vendored ViT: ``interpolate_pos_encoding`` reshapes the
   learned/sinusoidal ``pos_embed`` using the *original* patch grid
   (``t1/w1/h1`` from ``self.img_size``/``self.patch_size``) rather than the
   *target* grid derived from the input. Upstream at the pinned commit reshapes
   with the target grid, which raises whenever the input size differs from
   ``img_size``. We re-apply the vendored edit by monkeypatching the method so
   ported models reproduce the vendored ViT exactly.
"""

import sys

import rootutils
import torch.nn as nn

_ROOT = rootutils.find_root(search_from=__file__, indicator=".git")
_SPARSH = _ROOT / "third_party" / "sparsh"
if str(_SPARSH) not in sys.path:
    sys.path.insert(0, str(_SPARSH))

import tactile_ssl  # noqa: E402  (from third_party/sparsh)
from tactile_ssl.model.vision_transformer import (  # noqa: E402
    VisionTransformer,
    vit_base,
    vit_small,
    vit_tiny,
)

# Preserve the legacy alias so checkpoints pickled under 'tactile_ssl' resolve.
sys.modules.setdefault("tactile_ssl", tactile_ssl)


def _vendored_interpolate_pos_encoding(self, img_shape, img_dtype, device):
    """Vendored ``VisionTransformer.interpolate_pos_encoding`` (verbatim).

    Ported from the copy the original checkpoints were produced with. The only
    difference from the pinned submodule is that the ``pos_embed`` reshape uses
    the original grid (``t1/w1/h1`` from ``self.img_size``/``self.patch_size``)
    before interpolating to the target grid, instead of reshaping with the
    target grid.
    """
    previous_dtype = img_dtype
    pos_embed = None
    if self.pos_embed_fn == "sinusoidal":
        pos_embed = self.pos_embed(device).float().unsqueeze(0)
    elif self.pos_embed_fn == "learned":
        pos_embed = self.pos_embed.float()
    else:
        raise NotImplementedError("Unknown position embedding function")

    if self.is_video:
        _, _, t, h, w = img_shape
        if h == self.img_size[0] and w == self.img_size[1] and t == self.num_frames:
            return pos_embed

        dim = pos_embed.shape[-1]
        t0 = t // self.tubelet_size
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        t1 = self.num_frames // self.tubelet_size
        w1 = self.img_size[1] // self.patch_size
        h1 = self.img_size[0] // self.patch_size

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, t1, w1, h1, dim).permute(0, 4, 1, 2, 3),
            mode="trilinear",
            antialias=self.interpolate_antialias,
            size=(t0, w0, h0),
        )
        assert (t0, w0, h0) == pos_embed.shape[-3:]
        pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        return pos_embed

    _, _, h, w = img_shape
    if h == self.img_size[0] and w == self.img_size[1]:
        return pos_embed

    dim = pos_embed.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    w1 = self.img_size[1] // self.patch_size
    h1 = self.img_size[0] // self.patch_size

    pos_embed = nn.functional.interpolate(
        pos_embed.reshape(1, w1, h1, dim).permute(0, 3, 1, 2),
        mode="bicubic",
        antialias=self.interpolate_antialias,
        size=(w0, h0),
    )
    assert (w0, h0) == pos_embed.shape[-2:]
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return pos_embed.to(previous_dtype)


# Re-apply the one functional vendored edit to the pinned submodule.
VisionTransformer.interpolate_pos_encoding = _vendored_interpolate_pos_encoding

__all__ = ["vit_small", "vit_base", "vit_tiny", "VisionTransformer"]
