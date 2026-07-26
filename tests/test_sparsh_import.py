"""Tests for the Sparsh import shim (third_party/sparsh choke point).

TEST A (smoke): the shim re-exports the ViT builders and they construct.
TEST B (teeth): the shim's monkeypatched ``interpolate_pos_encoding`` reproduces
the *vendored* Sparsh ViT exactly on a non-default input size (the one functional
local edit the licensing audit found). The pristine submodule at the pinned
commit lacks that edit and would crash on the same input.
"""

import pytest
import rootutils
import torch


def test_sparsh_builders_import():
    from dlofeel_state.models.backbones._sparsh_import import (
        VisionTransformer,
        vit_small,
    )

    m = vit_small()
    assert isinstance(m, VisionTransformer)


def test_shim_reproduces_vendored_pos_encoding():
    # Shim path: importing the builder executes the shim module, which applies
    # the re-applied interpolate_pos_encoding monkeypatch as a side effect.
    from dlofeel_state.models.backbones._sparsh_import import (
        vit_small as shim_vit_small,
    )

    # Vendored reference lives beside the repo at ../repo/src (sparsh.tactile_ssl.*).
    repo_root = rootutils.find_root(search_from=__file__, indicator=".git")
    vendored_src = repo_root.parent / "repo" / "src"
    vendored_vt = (
        vendored_src / "sparsh" / "tactile_ssl" / "model" / "vision_transformer.py"
    )
    if not vendored_vt.exists():
        pytest.skip(f"vendored Sparsh reference not found at {vendored_vt}")

    import sys

    if str(vendored_src) not in sys.path:
        sys.path.insert(0, str(vendored_src))
    from sparsh.tactile_ssl.model.vision_transformer import (
        vit_small as vendored_vit_small,
    )

    # Identical weights in both models so only the pos-encoding code path can differ.
    torch.manual_seed(0)
    vendored = vendored_vit_small().eval()
    torch.manual_seed(0)
    shim = shim_vit_small().eval()
    shim.load_state_dict(vendored.state_dict())

    # Non-default input size (default is 224) => interpolate_pos_encoding must run
    # interpolation. The unpatched pristine method reshapes with the wrong grid and
    # raises; the vendored/patched method reshapes with the original grid.
    torch.manual_seed(123)
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out_vendored = vendored(x)
        out_shim = shim(x)

    assert out_shim.shape == out_vendored.shape
    assert torch.allclose(out_shim, out_vendored, atol=1e-5), (
        f"shim ViT diverges from vendored: max abs diff "
        f"{(out_shim - out_vendored).abs().max().item():.3e}"
    )
