#!/usr/bin/env python3
"""Fetch DLOFeel-State artifacts (dataset + trained weights) from the Hugging Face Hub.

PLACEHOLDER STUB. The repo IDs below name Hub repositories that do not exist
yet; they are finalized when the packed dataset and checkpoints are uploaded.
The interface (constants, CLI, and the two download functions) is stable now so
downstream code and docs can wire against it before the artifacts land. Running
it against the current placeholder IDs raises a Hub "repo not found" error.
"""

import argparse
from pathlib import Path

import rootutils
from huggingface_hub import snapshot_download

# Placeholder Hub repositories, finalized when artifacts are uploaded.
DATASET_REPO_ID = "rotematari/dlofeel-state-data"
WEIGHTS_REPO_ID = "rotematari/dlofeel-state-weights"

# Repo-root-relative default so the destination is machine-independent.
_ROOT = rootutils.find_root(search_from=__file__, indicator=".git")
DEFAULT_DEST = _ROOT / "data"


def download_dataset(dest: Path) -> str:
    """Download the packed dataset repo into ``dest`` and return the local path."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
    )


def download_weights(dest: Path) -> str:
    """Download the trained-weights repo into ``dest`` and return the local path."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=WEIGHTS_REPO_ID,
        repo_type="model",
        local_dir=str(dest),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DLOFeel-State dataset and/or weights from the Hugging Face Hub."
    )
    parser.add_argument(
        "--what",
        choices=["dataset", "weights", "all"],
        default="all",
        help="Which artifact(s) to fetch.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination directory (default: <repo_root>/data).",
    )
    args = parser.parse_args()

    if args.what in ("dataset", "all"):
        path = download_dataset(args.dest / "dataset")
        print(f"Dataset downloaded to: {path}")
    if args.what in ("weights", "all"):
        path = download_weights(args.dest / "weights")
        print(f"Weights downloaded to: {path}")


if __name__ == "__main__":
    main()
