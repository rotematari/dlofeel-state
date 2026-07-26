"""Import / CLI dry-run tests for the ported experiment drivers (Task 13b).

Each ported driver under ``experiments/`` is a curate-and-port of the original
``simplepredict`` experiment scripts with imports/entrypoints rewired to the
``dlofeel_state`` package. These tests prove the WIRING resolves against the new
package -- every module-level (and, at ``--help`` time, argparse-time) import
loads and the CLI parser is reachable -- WITHOUT running any real experiment.
Full experiment runs need the released dataset (out of scope here), so nothing
below touches data, a checkpoint, or a GPU.

Two proof modes:
  * ``--help`` via subprocess (rc == 0): argparse forces every module-level
    import to execute, so a clean ``--help`` exercises the driver's full import
    chain plus its CLI wiring, then exits before any data/GPU work.
  * importlib module import: loads a library-style driver's module object;
    importing runs the module body, NOT its ``main()``.

Robustness: a ``--help`` that fails purely because an OPTIONAL heavy third-party
dependency is absent from the env is SKIPPED with the missing module named (never
silently passed). A failure that names ``dlofeel_state`` or ``simplepredict`` is a
real rewiring regression and FAILS.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments"

# (test_id, path relative to experiments/). Every ported PYTHON driver with a CLI.
# The task's required-coverage set is the first eight; the viz/plot drivers are
# folded in so the whole ported surface is wiring-checked.
HELP_DRIVERS = [
    ("eval_diff_bspline", "table1_model_comparison/eval_diff_bspline.py"),
    ("tactile_sensor_ablation", "table2_sensor_ablation/tactile_sensor_ablation.py"),
    ("eval_tactile_drop", "table2_sensor_ablation/eval_tactile_drop.py"),
    ("train_tactile_drop", "table2_sensor_ablation/train_tactile_drop.py"),
    ("train_session_scaling", "fig4_data_efficiency/train_session_scaling.py"),
    ("eval_tactile_noise", "fig6_noise/eval_tactile_noise.py"),
    ("viz_noise_sweep", "fig6_noise/viz_noise_sweep.py"),
    ("gen_k_norm_stats", "k_resample/gen_k_norm_stats.py"),
    ("plot_k_resample", "k_resample/plot_k_resample.py"),
    ("build_combined_union", "fig5_transfer/build_combined_union.py"),
    ("plot_session_scaling_summary", "fig4_data_efficiency/plot_session_scaling_summary.py"),
]


def _missing_optional_dep(stderr: str) -> str | None:
    """Return a missing third-party module name if that (and not a rewiring bug)
    is why ``--help`` failed; else ``None``.

    A ``ModuleNotFoundError`` for a package OTHER than ``dlofeel_state`` /
    ``simplepredict`` means the port is wired but the env lacks a heavy runtime
    dep -> caller SKIPs. A miss naming ``dlofeel_state`` (or a leftover
    ``simplepredict``) is a real regression -> caller must FAIL (return None).
    """
    m = re.search(r"ModuleNotFoundError: No module named ['\"]([\w.]+)['\"]", stderr)
    if not m:
        return None
    top = m.group(1).split(".")[0]
    if top in {"dlofeel_state", "simplepredict"}:
        return None  # rewiring regression -- must FAIL, not skip
    return m.group(1)


@pytest.mark.parametrize("name,relpath", HELP_DRIVERS, ids=[d[0] for d in HELP_DRIVERS])
def test_driver_help_exits_zero(name: str, relpath: str) -> None:
    """``python <driver> --help`` exits 0, proving imports + CLI wiring resolve."""
    script = EXP / relpath
    assert script.is_file(), f"ported driver missing: {script}"

    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # the return code is the assertion below, not an exception
    )

    if proc.returncode != 0:
        missing = _missing_optional_dep(proc.stderr)
        if missing is not None:
            pytest.skip(f"{name}: optional dependency {missing!r} not installed in env")
        raise AssertionError(
            f"{name} --help exited {proc.returncode}\n--- stderr (tail) ---\n{proc.stderr[-3000:]}"
        )

    assert "usage" in proc.stdout.lower(), f"{name}: --help produced no usage text"


def test_build_combined_union_imports_as_library() -> None:
    """importlib-load the pure-planner driver: the body runs, ``main()`` does not.

    Covers the "library-style driver -> assert import succeeds" clause explicitly
    and confirms the rewired module binds and its disk-free planner works.
    """
    script = EXP / "fig5_transfer" / "build_combined_union.py"
    spec = importlib.util.spec_from_file_location("build_combined_union_under_test", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # executes the module body, never __main__

    assert hasattr(mod, "plan_union_links"), "ported build_combined_union missing plan_union_links"
    # Pure planner on a synthetic cable->session map (no disk): shared bare session
    # names survive as distinct <cable>__<session> dests.
    pairs = mod.plan_union_links({"c1": ["/x/sA"], "c2": ["/y/sA"]})
    assert [dest for _, dest in pairs] == ["c1__sA", "c2__sA"]


# ---------------------------------------------------------------------------
# Shell drivers: the argv they hand to the ``dlofeel-train`` console script
# ---------------------------------------------------------------------------
# The shell drivers build their train command as a bash array, so the only
# faithful way to check it is to let each driver run and intercept what it would
# have executed. Two capture channels, one per driver shape:
#
#   "exec" -- a PATH shim named ``uv`` records argv and exits 0, so the driver
#             walks its whole command sequence (including the ``uv run python``
#             helper steps) without launching anything real.
#   "echo" -- run_c3_xblock.sh guards on released data + a trained checkpoint and
#             ``exit 2``s without them, so it only runs under ``--dry-run``, which
#             prints each command instead of executing it. Its own ``CMD :`` echo
#             is then the capture point. Safe to whitespace-split: every token it
#             passes is space-free.
#
# The reserved-flag trap this guards: draccus owns ``--config_path`` for its own
# YAML-override loader, so the model-architecture YAML must go to
# ``--train_config_path`` (see TrainConfig.train_config_path). Passing it to
# ``--config_path`` fails at decode time, before any training work.

_UV_SHIM = """#!/usr/bin/env bash
# Stand-in for `uv`: append argv (unit-separated, one line per call) and exit 0.
{
  sep=""
  for a in "$@"; do printf '%s%s' "$sep" "$a"; sep=$'\\x1f'; done
  printf '\\n'
} >> "$DLOFEEL_ARGV_CAPTURE"
exit 0
"""

# (test_id, path relative to experiments/, extra driver args, channel, expected
# number of dlofeel-train invocations).
TRAIN_DRIVERS = [
    ("run_model_comparison", "table1_model_comparison/run_model_comparison.sh", [], "exec", 3),
    ("run_diff_bspline", "table1_model_comparison/run_diff_bspline.sh", [], "exec", 1),
    ("run_k_resample", "k_resample/run_k_resample.sh", [], "exec", 5),
    ("run_c3_xblock", "fig5_transfer/run_c3_xblock.sh", ["--dry-run"], "echo", 3),
]

_ECHOED_TRAIN = re.compile(r"uv run dlofeel-train\s+(\S.*)$")


def _capture_train_argvs(name: str, relpath: str, extra: list[str], channel: str, tmp_path: Path) -> list[list[str]]:
    """Run a shell driver harmlessly and return each ``dlofeel-train`` argv it built.

    Returned argv lists exclude the leading ``run dlofeel-train`` launcher tokens,
    so they are exactly what draccus would see.
    """
    script = EXP / relpath
    assert script.is_file(), f"shell driver missing: {script}"

    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "uv"
    shim.write_text(_UV_SHIM)
    shim.chmod(0o755)
    capture = tmp_path / "argv_capture.txt"
    capture.touch()

    env = dict(os.environ)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"
    env["DLOFEEL_ARGV_CAPTURE"] = str(capture)

    proc = subprocess.run(
        ["bash", str(script), *extra],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # the return code is the assertion below, not an exception
    )
    assert proc.returncode == 0, (
        f"{name} exited {proc.returncode} under the dry-run shim\n"
        f"--- stderr (tail) ---\n{proc.stderr[-3000:]}"
    )

    if channel == "exec":
        argvs = [
            argv[2:]
            for argv in (line.split("\x1f") for line in capture.read_text().splitlines())
            if argv[:2] == ["run", "dlofeel-train"]
        ]
    else:
        argvs = [
            m.group(1).split()
            for line in proc.stdout.splitlines()
            if (m := _ECHOED_TRAIN.search(line))
        ]
    return argvs


@pytest.fixture(scope="module")
def train_config_fields() -> set[str]:
    from dlofeel_state.configs.train import TrainConfig

    return {f.name for f in dataclasses.fields(TrainConfig)}


@pytest.mark.parametrize(
    "name,relpath,extra,channel,n_expected", TRAIN_DRIVERS, ids=[d[0] for d in TRAIN_DRIVERS]
)
def test_driver_train_argv_uses_real_trainconfig_flags(
    name: str, relpath: str, extra: list[str], channel: str, n_expected: int,
    tmp_path: Path, train_config_fields: set[str],
) -> None:
    """Every flag a driver passes to ``dlofeel-train`` names a real TrainConfig field.

    In particular the model YAML must NOT ride on draccus's reserved
    ``--config_path``: that flag loads a draccus override file whose keys are
    TrainConfig fields, so handing it the model-architecture YAML raises
    ``DecodingError`` on ``model_params``, ``num_output_pts``, etc.

    A ``--no-<field>`` token is the negated form of ``<field>`` (see
    ``BoolFlagAction``), so it is checked against the same field set.
    """
    argvs = _capture_train_argvs(name, relpath, extra, channel, tmp_path)
    assert len(argvs) == n_expected, f"{name}: captured {len(argvs)} train invocations, want {n_expected}"

    for argv in argvs:
        flags = [t for t in argv if t.startswith("--")]
        assert "--config_path" not in flags, (
            f"{name}: passes the model YAML through draccus's reserved --config_path; "
            "use --train_config_path"
        )
        assert "--train_config_path" in flags, f"{name}: no --train_config_path in {flags}"
        unknown = [f for f in flags if f.removeprefix("--no-").removeprefix("--") not in train_config_fields]
        assert not unknown, f"{name}: flags absent from TrainConfig: {unknown}"


@pytest.mark.parametrize(
    "name,relpath,extra,channel,n_expected", TRAIN_DRIVERS, ids=[d[0] for d in TRAIN_DRIVERS]
)
def test_driver_train_argv_parses_as_trainconfig(
    name: str, relpath: str, extra: list[str], channel: str, n_expected: int, tmp_path: Path
) -> None:
    """The strong form: each captured argv decodes into a TrainConfig via draccus.

    Covers the flag NAMES and the flag FORMS together -- the drivers lean on the
    bare ``--with_att`` and negated ``--no-use_wandb`` spellings that
    ``BoolFlagAction`` provides, so a regression in either surfaces here.
    """
    import draccus

    from dlofeel_state.configs.train import TrainConfig

    argvs = _capture_train_argvs(name, relpath, extra, channel, tmp_path)
    assert len(argvs) == n_expected

    for argv in argvs:
        cfg = draccus.parse(TrainConfig, args=argv, exit_on_error=False)
        assert cfg.train_config_path.endswith(".yaml")
        # Every driver asks for attention + normalized inputs and silences W&B.
        assert cfg.with_att is True
        assert cfg.normalize_inputs is True
        # Omitted flags must stay None so the model YAML keeps deciding.
        assert cfg.only_pose is None or "--only_pose" in argv
