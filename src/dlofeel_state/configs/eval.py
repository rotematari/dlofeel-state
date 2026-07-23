"""Typed draccus config for the ``dlofeel-eval`` console entrypoint.

Thin CLI surface over the parity-locked ``engine.evaluate.evaluate()``: this
dataclass carries only the run-level knobs (data/checkpoint/output paths and the
eval hyperparameters that mirror the original ``simplepredict.evaluate`` argparse
CLI). The large model-architecture config stays as YAML loaded from the
checkpoint folder (or ``config_path``), the standard draccus split between a typed
run config and a YAML model config.

Fields without defaults come first so this is a valid dataclass; every field is a
draccus CLI flag (e.g. ``--data_root``, ``--ckpt_path``) and can be overridden from
a YAML file passed with ``--config_path`` at the draccus level.
"""

from dataclasses import dataclass


@dataclass
class EvalConfig:
    # ----- required run paths (mirror the original evaluate.py __main__) -----
    data_root: str
    """Directory of eval samples (``sample*.h5`` / session dirs)."""

    ckpt_path: str
    """Path to the model checkpoint ``.pt`` (a full state_dict; loaded strict)."""

    blank_ref_root: str
    """Directory of DIGIT blank reference images (overrides the model config)."""

    # ----- optional run knobs (mirror the original argparse defaults) -----
    out_dir: str = "results_eval"
    """Output root; results land under ``<out_dir>/<ckpt-parent-name>/``."""

    batch_size: int | None = None
    """Eval batch size; ``None`` falls back to the model config's ``batch_size``."""

    low_rmse_threshold: float = 0.055
    """Per-sample RMSE (m) below which samples are logged to low_rmse_samples.json."""

    # ----- new-repo overrides (not in the original CLI) -----
    model_config_path: str | None = None
    """Path to the model-architecture config YAML. ``None`` auto-loads
    ``<ckpt_dir>/config.yaml`` exactly like the original evaluate.py __main__.

    Named ``model_config_path`` (not ``config_path``) because draccus reserves the
    flag ``--config_path`` for its own YAML-override loader; ``configs/eval/k16.yaml``
    is such a draccus override, loaded via that reserved flag."""

    norm_stats_path: str | None = None
    """Override the model config's ``norm_stats_path`` (the normalization stats YAML).
    ``None`` keeps whatever the loaded model config specifies."""
