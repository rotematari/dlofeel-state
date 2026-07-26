#!/usr/bin/env python3
"""
Build the c1+c2+c3 symlink-union train dir for the cross-cable (union) norm-stats pass.

The union ``norm_stats`` must be computed over a single real ``DLOH5SampleDataset`` —
``compute_all_stats`` mutates ``dataset.normalize`` (``utils.py:192-194``), so a
``ConcatDataset``/``Subset`` over the three cables cannot be used (the toggle would not
reach the underlying datasets, silently leaving normalization on). This script
materializes that single dataset as a directory of symlinks: each per-cable session
dir ``src/datasets/<cable>/ds/train/<session>`` is linked into
``src/datasets/combined/ds/train/<cable>__<session>``.

The ``<cable>__`` prefix guarantees no ``parent.name`` collisions across cables (the
dataset sorts/indexes by ``(parent.name, int(stem.split('_')[1]))`` at
``dataset_seq.py:205``; the prefix only changes ``parent.name``, not the
``sample_<n>.h5`` stem the integer index reads). The union dir holds FAST_DATA-absolute
symlinks and is gitignored, exactly like c1/c2/c3.

``--dry-run`` prints the planned (src, dest) pairs and a collision check, touching
nothing. ``--selftest`` / ``--help`` exercise the pure ``plan_union_links`` planner on a
synthetic cable->session map and never walk the disk.

Usage (from repo root):
    uv run python experiments/fig5_transfer/build_combined_union.py
    uv run python experiments/fig5_transfer/build_combined_union.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CABLES = ("c1", "c2", "c3")
DEFAULT_DATASETS_ROOT = "src/datasets"
DEFAULT_UNION_TRAIN = "src/datasets/combined/ds/train"
_TRAIN_SUBPATH = "ds/train"


# ----------------------------- Pure planner (no disk walk) -----------------------------


def plan_union_links(cable_sessions: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Plan the union symlinks from a cable -> session-dir map.

    Each session of each cable becomes one ``(abs_src, dest_name)`` pair, where
    ``dest_name`` is ``<cable>__<session-basename>`` (the basename so the prefix is the
    only collision-relevant component). The planner is pure and disk-free: ``cable_sessions``
    values are the absolute (or any) source paths of the per-cable session dirs.

    Args:
        cable_sessions: Map ``cable -> [session_dir_path, ...]``. Order is preserved.

    Returns:
        List of ``(abs_src, dest_name)`` pairs, one per session, in cable-then-session
        order. ``abs_src`` is the resolved absolute source path; ``dest_name`` is the
        union-dir-relative link name.

    Raises:
        AssertionError: if two sessions map to the same ``dest_name`` (a collision the
            ``<cable>__`` prefix is meant to prevent).
    """
    pairs: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for cable, sessions in cable_sessions.items():
        for session_path in sessions:
            src = str(Path(session_path).resolve())
            dest_name = f"{cable}__{Path(session_path).name}"
            assert dest_name not in seen, (
                f"dest collision: {dest_name!r} from {src!r} and {seen[dest_name]!r}"
            )
            seen[dest_name] = src
            pairs.append((src, dest_name))
    return pairs


# ----------------------------- Disk discovery + link creation (lazy os import) -----------------------------


def discover_cable_sessions(
    datasets_root: str | Path,
    cables: tuple[str, ...] = DEFAULT_CABLES,
) -> dict[str, list[str]]:
    """Walk ``<datasets_root>/<cable>/ds/train`` for each cable's session dirs.

    A "session" is a direct child directory of the cable's ``ds/train`` (the dataset's
    non-packed layout is ``<session>/sample_<n>.h5``, ``dataset_seq.py:200-205``).

    Args:
        datasets_root: Repo-relative or absolute ``src/datasets`` root.
        cables: Cable subdir names to include (default ``("c1", "c2", "c3")``).

    Returns:
        Map ``cable -> [sorted session-dir absolute paths]``. Cables whose
        ``ds/train`` is missing are skipped with a warning (a cable may be unmounted).
    """
    root = Path(datasets_root)
    cable_sessions: dict[str, list[str]] = {}
    for cable in cables:
        train_dir = root / cable / _TRAIN_SUBPATH
        if not train_dir.is_dir():
            print(f"[build_combined_union] WARNING: {train_dir} missing — skipping cable {cable!r}.")
            continue
        sessions = sorted(str(p.resolve()) for p in train_dir.iterdir() if p.is_dir())
        if not sessions:
            print(f"[build_combined_union] WARNING: no session dirs under {train_dir} — skipping.")
            continue
        cable_sessions[cable] = sessions
    return cable_sessions


def build_union(
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
    union_train: str | Path = DEFAULT_UNION_TRAIN,
    cables: tuple[str, ...] = DEFAULT_CABLES,
    force: bool = False,
    dry_run: bool = False,
) -> list[tuple[str, str]]:
    """Materialize the union train dir as ``<cable>__<session>`` symlinks.

    Idempotent: a link already pointing at the correct source is left untouched; a stale
    or wrong-target link is replaced only under ``force`` (otherwise it is an error, to
    avoid silently clobbering). ``dry_run`` prints the plan and returns without touching
    disk.

    Args:
        datasets_root: ``src/datasets`` root holding the per-cable dirs.
        union_train: Destination union train dir (``src/datasets/combined/ds/train``).
        cables: Cable subdir names to union.
        force: Replace existing links whose target differs from the plan.
        dry_run: Print the planned pairs + collision check; create nothing.

    Returns:
        The planned ``(abs_src, dest_name)`` pairs.
    """
    cable_sessions = discover_cable_sessions(datasets_root, cables)
    pairs = plan_union_links(cable_sessions)
    union_dir = Path(union_train)

    print(f"[build_combined_union] {len(pairs)} session links planned across {len(cable_sessions)} cables.")
    for src, dest_name in pairs:
        print(f"  {dest_name}  ->  {src}")

    if dry_run:
        print("[build_combined_union] --dry-run: no links created.")
        return pairs

    union_dir.mkdir(parents=True, exist_ok=True)
    created = relinked = skipped = 0
    for src, dest_name in pairs:
        dest = union_dir / dest_name
        src_path = Path(src)
        if dest.is_symlink() or dest.exists():
            current = Path(dest.readlink()).resolve() if dest.is_symlink() else None
            if current is not None and current == src_path:
                skipped += 1
                continue
            if not force:
                raise FileExistsError(
                    f"{dest} exists and points elsewhere ({current}); pass --force to relink."
                )
            dest.unlink()
            dest.symlink_to(src_path)
            relinked += 1
        else:
            dest.symlink_to(src_path)
            created += 1

    print(f"[build_combined_union] done: {created} created, {relinked} relinked, {skipped} already-correct.")
    return pairs


# ----------------------------- Selftest (synthetic; no disk) -----------------------------


def _selftest() -> None:
    """Probe ``plan_union_links`` on a synthetic cable->session map; exit 0.

    Asserts: every dest is ``<cable>__<session>``-prefixed and unique, pair count equals
    the sum of per-cable session counts, and a genuine cross-cable name clash (same
    bare session name under two cables) does NOT collide after prefixing. Never walks disk.
    """
    cable_sessions = {
        "c1": ["/fast/c1/ds/train/sessionA", "/fast/c1/ds/train/sessionB"],
        "c2": ["/fast/c2/ds/train/sessionA", "/fast/c2/ds/train/sessionC"],  # 'sessionA' shared name
        "c3": ["/fast/c3/ds/train/sessionD"],
    }
    pairs = plan_union_links(cable_sessions)

    expected_count = sum(len(v) for v in cable_sessions.values())
    assert len(pairs) == expected_count, f"pair count {len(pairs)} != Σ sessions {expected_count}"

    dests = [dest for _, dest in pairs]
    assert len(set(dests)) == len(dests), f"duplicate dest names: {dests}"

    for cable, sessions in cable_sessions.items():
        for session_path in sessions:
            expected = f"{cable}__{Path(session_path).name}"
            assert expected in dests, f"missing planned dest {expected!r}"

    # The shared bare name 'sessionA' must survive as two distinct prefixed dests.
    assert "c1__sessionA" in dests and "c2__sessionA" in dests, "prefix failed to disambiguate shared name"

    # A real dest collision (same cable, same session basename under two paths) must raise.
    collided = False
    try:
        plan_union_links({"c1": ["/a/dupe", "/b/dupe"]})
    except AssertionError:
        collided = True
    assert collided, "plan_union_links failed to detect a real dest collision"

    print(
        f"selftest OK: plan_union_links -> {len(pairs)} unique <cable>__<session> dests "
        "(= Σ per-cable sessions), shared bare names disambiguated, collisions raise."
    )


# ----------------------------- CLI -----------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build the c1+c2+c3 symlink-union train dir for the cross-cable norm-stats pass."
    )
    p.add_argument("--datasets_root", type=str, default=DEFAULT_DATASETS_ROOT,
                   help="src/datasets root holding per-cable dirs.")
    p.add_argument("--union_train", type=str, default=DEFAULT_UNION_TRAIN,
                   help="Destination union train dir (default: src/datasets/combined/ds/train).")
    p.add_argument("--cables", type=str, nargs="+", default=list(DEFAULT_CABLES),
                   help="Cable subdir names to union (default: c1 c2 c3).")
    p.add_argument("--force", action="store_true",
                   help="Relink existing links whose target differs from the plan.")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Print the planned (src, dest) pairs + collision check; create nothing.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the synthetic plan_union_links probe and exit 0 (no disk walk).")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return

    build_union(
        datasets_root=args.datasets_root,
        union_train=args.union_train,
        cables=tuple(args.cables),
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
