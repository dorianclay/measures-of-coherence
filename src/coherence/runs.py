"""Run records: experiments/NNN-<slug>/runs/<run-id>/run.json.

`record` writes how a run was produced. Code running under it adds what the run measured through `log`.
"""

import json
import os
from pathlib import Path

from coherence import datasets as datasets_, protocols
from coherence.hashing import sha256
from coherence.paths import store_root


def read(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


def write(run_dir: Path, run: dict) -> None:
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(json.dumps(run, indent=2) + "\n")
    tmp.replace(run_dir / "run.json")


def current_run_dir() -> Path | None:
    """The directory of the run this process belongs to, or None outside `record`."""
    return Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ else None


def manifest_entry(path: Path) -> dict:
    """An artifact's place in the manifest. Paths are relative to $COHERENCE_STORE, which differs between machines."""
    path = Path(path).resolve()
    store = store_root().resolve()
    if not path.is_relative_to(store):
        raise ValueError(f"Artifact {path} is outside $COHERENCE_STORE ({store}); write large outputs under $RUN_STORE")
    return {"path": str(path.relative_to(store)), "size": path.stat().st_size, "sha256": sha256(path)}


def log(*, model_ref: str | None = None, protocol: str | None = None, seed: int | None = None,
        datasets: list[str] = (), artifacts: list[Path] = ()) -> None:
    """Add what this run measured to its record. Does nothing outside `record`, so scripts still run standalone.

    Datasets are names in data/datasets.json; each must still match its recorded version. Artifacts are files
    under $RUN_STORE; logging a path again replaces its manifest entry.
    """
    if protocol is not None:
        protocols.get(protocol)
    run_dir = current_run_dir()
    if run_dir is None:
        return
    for name in datasets:
        datasets_.path(name)
    entries = [manifest_entry(a) for a in artifacts]
    run = read(run_dir)
    run.update({k: v for k, v in {"model_ref": model_ref, "protocol": protocol, "seed": seed}.items() if v is not None})
    if datasets:
        run["datasets"] = {**run.get("datasets", {}), **{name: datasets_.version(name) for name in datasets}}
    if entries:
        new_paths = {e["path"] for e in entries}
        run["artifacts"] = [a for a in run.get("artifacts", []) if a["path"] not in new_paths] + entries
    write(run_dir, run)
