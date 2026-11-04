"""Repo-wide locations, so scripts behave the same from any working directory."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
EVENT_FAMILIES_DIR = DATA_DIR / "event_families"
# Large artifacts (activations, checkpoints, per-family rates) live outside git. On the cluster, point
# COHERENCE_STORE at shared storage; locally it defaults to a gitignored directory in the repo.
STORE = Path(os.environ.get("COHERENCE_STORE", REPO_ROOT / "store"))
