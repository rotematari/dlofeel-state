"""Regression guard: no machine-specific absolute paths in executable code.

The repo must run on any machine, so checked-in code carries no hard-coded
paths tied to the author's filesystem. This scans executable sources (``.py``
and ``.sh``) under ``src/``, ``scripts/``, and ``experiments/`` for the two
machine-specific absolute prefixes and asserts zero hits.

Scope is deliberately narrow:
- Only the literal prefixes ``/home/rotem`` and ``/media/rotem`` fail the guard.
- Repo-root-relative defaults (e.g. ``src/datasets/ds_h5/...``) are allowed.
- The token ``FAST_DATA`` in a comment or doc is allowed.
Build/cache artifacts (``__pycache__``, ``*.egg-info``, ``.pytest_cache``) are
excluded so generated files can never trip the guard.
"""

import rootutils

# Machine-specific absolute prefixes that must never appear in checked-in code.
FORBIDDEN_PREFIXES = ("/home/rotem", "/media/rotem")

# Directories holding executable sources that ship with the repo.
SCAN_DIRS = ("src", "scripts", "experiments")

# Extensions treated as executable code.
CODE_SUFFIXES = (".py", ".sh")

# Path fragments whose files are generated/cached, never authored.
EXCLUDE_FRAGMENTS = ("__pycache__", ".egg-info", ".pytest_cache")


def _iter_code_files():
    root = rootutils.find_root(search_from=__file__, indicator=".git")
    for scan_dir in SCAN_DIRS:
        base = root / scan_dir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in CODE_SUFFIXES:
                continue
            if any(frag in str(path) for frag in EXCLUDE_FRAGMENTS):
                continue
            yield path


def test_no_machine_specific_absolute_paths():
    offenders = []
    for path in _iter_code_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for prefix in FORBIDDEN_PREFIXES:
                if prefix in line:
                    offenders.append(f"{path}:{lineno}: {line.strip()}")

    assert not offenders, (
        "Machine-specific absolute paths found in executable code "
        f"(forbidden prefixes {FORBIDDEN_PREFIXES}):\n" + "\n".join(offenders)
    )
