"""Repo-wide locations, so scripts behave the same from any working directory.

COHERENCE_ROOT overrides the repo root (tests point it at a throwaway repo). COHERENCE_STORE is where large artifacts
live: on the cluster, point it at shared storage; locally it defaults to a gitignored directory in the repo.
The functions read the environment at call time; the constants are fixed at import, for the pipeline scripts.
"""

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(os.environ.get("COHERENCE_ROOT", Path(__file__).resolve().parents[2]))


def store_root() -> Path:
    return Path(os.environ.get("COHERENCE_STORE", repo_root() / "store"))


def experiments_dir() -> Path:
    return repo_root() / "experiments"


def data_dir() -> Path:
    return repo_root() / "data"


REPO_ROOT = repo_root()
DATA_DIR = data_dir()
DATASETS_DIR = DATA_DIR / "datasets"
EVENT_FAMILIES_DIR = DATA_DIR / "event_families"
STORE = store_root()
