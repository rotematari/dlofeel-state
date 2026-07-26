"""Typed draccus config for the ``dlofeel-train`` console entrypoint.

Thin CLI surface over the ported training loop (``engine.train.train()``): this
dataclass mirrors, one-for-one, every flag of the original
``simplepredict.train`` argparse CLI, with the same names and default values. The
large model-architecture config still lives in a YAML (``train_config_path``,
default ``configs/train/default.yaml``); this dataclass only carries the
run-level overrides that the CLI exposed. draccus's ``main()`` turns these into
the keyword arguments the ported ``train(...)`` function expects.

Every field has a default, so field order is unconstrained. The tri-state
booleans (``bool | None``, default ``None``) preserve the original CLI's
``--flag`` / ``--no-flag`` / omit semantics: ``None`` means "do not override the
value the YAML config already carries".
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ----- dataset roots (argparse: --train_data_root / --val_data_root / --blank_ref_root) -----
    train_data_root: str | None = None
    """Directory containing training ``sample*.h5`` (else read from the YAML config)."""

    val_data_root: str | None = None
    """Directory containing validation ``sample*.h5`` (else read from the YAML config)."""

    blank_ref_root: str | None = None
    """Directory of DIGIT blank reference images for diff images (else from the YAML config)."""

    # ----- config / stats paths (argparse: --config_path / --norm_stats_path) -----
    train_config_path: str = "configs/train/default.yaml"
    """Path to the model/training YAML config (the original ``--config_path``).

    Renamed ``train_config_path`` because draccus reserves the flag
    ``--config_path`` for its own YAML-override loader; the two coexist in
    ``--help`` (``--config_path`` loads a draccus override file, ``--train_config_path``
    is the model config this run trains from)."""

    norm_stats_path: str | None = None
    """Path to the normalization-stats YAML (else read from the YAML config)."""

    # ----- output (argparse: --out_dir) -----
    out_dir: str = "results"
    """Output root; checkpoints land under ``<out_dir>/checkpoints/<exp_name>/``."""

    # ----- resampling (argparse: --num_resample_points) -----
    num_resample_points: int | None = None
    """Number of markers to resample (overrides the config's ``num_output_pts``)."""

    # ----- tri-state boolean flags (argparse add_bool_flag: --x / --no-x, default None) -----
    with_att: bool | None = None
    """Whether to use the attention mechanism."""

    use_diff_images: bool | None = None
    """Whether to use difference images as input."""

    use_blank_ref: bool | None = None
    """Whether to use a blank reference image for diff images."""

    only_pose: bool | None = None
    """Whether to use only pose as input (no images)."""

    normalize: bool | None = None
    """Whether to normalize the target (marker) data."""

    normalize_inputs: bool | None = None
    """Whether to normalize the input data."""

    single_encoder: bool | None = None
    """Whether to use a single encoder for all digit images."""

    use_spline: bool | None = None
    """Whether to fit a spline to the predicted nodes."""

    use_weighted_loss: bool | None = None
    """Whether to keep the physics loss weights; ``False`` zeros w_edges/w_lengths/w_kb."""

    # ----- fine-tune init (argparse: --init_from) -----
    init_from: str | None = None
    """Checkpoint ``.pt`` to load weights from (state_dict, strict=False) for
    fine-tune init. Weights-only: no optimizer/epoch resume."""

    # ----- spline / backbone (argparse: --n_control_points / --image_backbone_type) -----
    n_control_points: int | None = None
    """Number of spline control points (only used when ``use_spline``)."""

    image_backbone_type: str | None = None
    """Override the image backbone type from config (``sparsh_vit`` or ``resnet18``)."""

    # ----- image / optimization knobs (argparse scalar overrides) -----
    img_size_H: int | None = None
    """Image height override (else from the YAML config)."""

    img_size_W: int | None = None
    """Image width override (else from the YAML config)."""

    batch_size: int | None = None
    """Batch size override (else from the YAML config)."""

    epochs: int | None = None
    """Number of epochs override (else from the YAML config)."""

    lr: float | None = None
    """Learning rate override (else from the YAML config)."""

    weight_decay: float | None = None
    """Weight-decay override (else from the YAML config)."""

    dropout: float | None = None
    """Dropout override (else from the YAML config)."""

    num_workers: int | None = None
    """DataLoader worker count override (else from the YAML config)."""

    seed: int | None = None
    """RNG seed override (else from the YAML config)."""

    # ----- wandb (argparse: --wandb_project / --wandb_entity / --use_wandb) -----
    wandb_project: str = "dlo-state-estimation"
    """WandB project name."""

    wandb_entity: str = "dlo"
    """WandB entity (username or team)."""

    use_wandb: bool | None = None
    """Whether to use WandB for logging (``None`` -> the training loop defaults to True)."""

    # ----- fast-testing limits (argparse: --max_batches / --max_eval_batches) -----
    max_batches: int = 0
    """Max training batches per epoch (0 = unlimited). Maps to ``max_batches_per_epoch``."""

    max_eval_batches: int = 0
    """Max validation batches per epoch (0 = full eval)."""

    # ----- ablation controls (argparse: --active_sensors / --exp_name / --model_type / --drop_augment) -----
    active_sensors: str | None = None
    """Comma-separated sensor indices or names, e.g. ``'0,1,2,3'`` or ``'fu,fd,ru,rd'``
    (fu=0, fd=1, ru=2, rd=3). ``None`` = all. Parsed to ``list[int]`` in ``main()``."""

    exp_name: str | None = None
    """Override experiment name (for ablation scripts). Maps to ``exp_name_override``.
    Default: time + params."""

    model_type: str = "dlo_state"
    """Model architecture: ``'dlo_state'`` (default Sparsh ViT) or ``'direction'``
    (CV direction ablation). Resolved to a model class in ``main()``."""

    drop_augment: bool = False
    """Wrap the train set with the 15-mask sensor-drop augmentation (default off)."""

    # ----- determinism (argparse: --deterministic / --deterministic_warn_only) -----
    deterministic: bool = False
    """Enable deterministic PyTorch behavior for better reproducibility."""

    deterministic_warn_only: bool = False
    """With ``deterministic``, warn instead of error on non-deterministic ops."""
