"""The ``dlofeel-train`` boolean-flag contract (``TrainConfig`` + ``BoolFlagAction``).

Three accepted CLI forms per boolean field -- ``--flag``, ``--no-flag`` and an
explicit ``--flag <bool>`` -- plus the omit case, which must leave a tri-state
``bool | None`` field at ``None`` so the model YAML keeps deciding that knob.

Parsing only: no dataset, checkpoint, or GPU is touched.
"""

import argparse

import draccus
import pytest

from dlofeel_state.configs.train import TrainConfig

# Every tri-state (``bool | None``) knob the training CLI exposes.
TRISTATE = [
    "with_att",
    "use_diff_images",
    "use_blank_ref",
    "only_pose",
    "normalize",
    "normalize_inputs",
    "single_encoder",
    "use_spline",
    "use_weighted_loss",
    "use_wandb",
]

# Plain ``bool`` knobs: same three forms, but omitting them yields the field default.
PLAIN = ["drop_augment", "deterministic", "deterministic_warn_only"]


def _parse(*args: str) -> TrainConfig:
    return draccus.parse(TrainConfig, args=list(args), exit_on_error=False)


@pytest.mark.parametrize("name", TRISTATE + PLAIN)
def test_bare_flag_sets_true(name: str) -> None:
    assert getattr(_parse(f"--{name}"), name) is True


@pytest.mark.parametrize("name", TRISTATE + PLAIN)
def test_negated_flag_sets_false(name: str) -> None:
    assert getattr(_parse(f"--no-{name}"), name) is False


@pytest.mark.parametrize("name", TRISTATE + PLAIN)
@pytest.mark.parametrize("literal,expected", [("true", True), ("false", False)])
def test_explicit_value_form(name: str, literal: str, expected: bool) -> None:
    """``--flag <bool>`` stays valid -- docs/training.md and docs/loss.md use it."""
    assert getattr(_parse(f"--{name}", literal), name) is expected


@pytest.mark.parametrize("name", TRISTATE)
def test_omitted_tristate_flag_stays_none(name: str) -> None:
    """``None`` is load-bearing: it means "do not override the YAML config"."""
    assert getattr(_parse(), name) is None


@pytest.mark.parametrize("name", PLAIN)
def test_omitted_plain_flag_keeps_field_default(name: str) -> None:
    assert getattr(_parse(), name) is False


def test_bare_flag_does_not_swallow_the_next_option() -> None:
    """The optional value must not consume a following ``--flag``/``--flag value`` pair."""
    cfg = _parse("--with_att", "--no-use_wandb", "--single_encoder", "--lr", "7.1e-5")
    assert (cfg.with_att, cfg.use_wandb, cfg.single_encoder, cfg.lr) == (
        True,
        False,
        True,
        7.1e-5,
    )


def test_negated_flag_rejects_a_value() -> None:
    """``--no-flag false`` is a double negative; it errors instead of guessing."""
    with pytest.raises(argparse.ArgumentError, match="takes no value"):
        _parse("--no-use_wandb", "false")


def test_flags_coexist_with_draccus_reserved_config_path(tmp_path) -> None:
    """draccus's own ``--config_path`` override file still wins over nothing else.

    Guards the split that ``--train_config_path`` exists for: ``--config_path``
    feeds draccus a TrainConfig-shaped YAML, while the model-architecture YAML
    rides on ``--train_config_path``. A CLI flag still beats the override file.
    """
    override = tmp_path / "override.yaml"
    override.write_text("with_att: false\nepochs: 3\n")

    cfg = _parse("--config_path", str(override))
    assert (cfg.with_att, cfg.epochs) == (False, 3)

    cfg = _parse("--config_path", str(override), "--with_att")
    assert (cfg.with_att, cfg.epochs) == (True, 3)
