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

import importlib.util
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
