"""Run records: experiments/NNN-<slug>/runs/<run-id>/run.json.

`record` writes how a run was produced. Code running under it adds what the run measured through `log`.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from functools import cache
from pathlib import Path

from coherence import datasets as datasets_, protocols
from coherence.hashing import sha256_tree, size
from coherence.paths import experiments_dir, store_root


def read(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


def write(run_dir: Path, run: dict) -> None:
    tmp = run_dir / "run.json.tmp"
    tmp.write_text(json.dumps(run, indent=2) + "\n")
    tmp.replace(run_dir / "run.json")


def find(run_id: str) -> Path:
    """The directory of a run, in whichever experiment it belongs to."""
    matches = list(experiments_dir().glob(f"*/runs/{run_id}/run.json"))
    if not matches:
        raise KeyError(f"No run {run_id} under {experiments_dir()}")
    return matches[0].parent


RUN_ID = r"\d{8}-\d{4}-[0-9a-f]{4}"
CHECKPOINT_REF = re.compile(rf"^({RUN_ID})@step(\d+)$")


def model_path(model_ref: str) -> Path:
    """Where the eval checkpoint a `<run-id>@step<N>` model reference names is stored."""
    match = CHECKPOINT_REF.match(model_ref)
    if not match:
        raise ValueError(f"{model_ref!r} isn't a checkpoint reference (<run-id>@step<N>)")
    run_id, step = match.group(1), int(match.group(2))
    evals = {a["step"]: a for a in read(find(run_id)).get("artifacts", []) if a.get("checkpoint") == "eval"}
    if step not in evals:
        raise ValueError(f"Run {run_id} has no eval checkpoint at step{step}; its eval checkpoints are at steps "
                         f"{', '.join(map(str, sorted(evals))) or '(none)'}")
    return store_root() / evals[step]["path"]


def current_run_dir() -> Path | None:
    """The directory of the run this process belongs to, or None outside `record`."""
    return Path(os.environ["RUN_DIR"]) if "RUN_DIR" in os.environ else None


def manifest_entry(path: Path, hashed: bool = True) -> dict:
    """An artifact's place in the manifest: a file or a directory under $COHERENCE_STORE, by its path relative to the
    store, which differs between machines."""
    path = Path(path).resolve()
    store = store_root().resolve()
    if not path.is_relative_to(store):
        raise ValueError(f"Artifact {path} is outside $COHERENCE_STORE ({store}); write large outputs under $RUN_STORE")
    entry = {"path": str(path.relative_to(store)), "size": size(path)}
    return {**entry, "sha256": sha256_tree(path)} if hashed else entry


def log(*, model_ref: str | None = None, protocol: str | None = None, seed: int | None = None,
        planned_steps: int | None = None, datasets: list[str] = (), artifacts: list[Path] = ()) -> None:
    """Add what this run measured to its record. Does nothing outside `record`, so scripts still run standalone.

    A model reference is `<hf-id>@<revision>` or an eval checkpoint, `<run-id>@step<N>`, which must exist.
    A training run gives its planned_steps; it stays running across segments until progress reaches them.
    Datasets are names in data/datasets.json; each must still match its recorded version. Artifacts are files
    under $RUN_STORE; logging a path again replaces its manifest entry.
    """
    if protocol is not None:
        protocols.get(protocol)
    if model_ref is not None and CHECKPOINT_REF.match(model_ref):
        model_path(model_ref)
    run_dir = current_run_dir()
    if run_dir is None:
        return
    for name in datasets:
        datasets_.path(name)
    entries = [manifest_entry(a) for a in artifacts]
    run = read(run_dir)
    run.update({k: v for k, v in {"model_ref": model_ref, "protocol": protocol, "seed": seed,
                                   "planned_steps": planned_steps}.items() if v is not None})
    if datasets:
        run["datasets"] = {**run.get("datasets", {}), **{name: datasets_.version(name) for name in datasets}}
    if entries:
        new_paths = {e["path"] for e in entries}
        run["artifacts"] = [a for a in run.get("artifacts", []) if a["path"] not in new_paths] + entries
    write(run_dir, run)


def step() -> int:
    """The last training step this run completed, in any segment; 0 for a new run or outside `record`."""
    run_dir = current_run_dir()
    return read(run_dir).get("step", 0) if run_dir else 0


def progress(step: int) -> None:
    """Record that the training run has completed `step` steps."""
    run_dir = current_run_dir()
    if run_dir is None:
        return
    run = read(run_dir)
    run["step"] = step
    run["segments"][-1]["steps"][1] = step
    write(run_dir, run)


def checkpoint(path: Path, *, step: int, kind: str = "eval") -> None:
    """List a training checkpoint under $RUN_STORE in the manifest.

    An eval checkpoint is weights only, saved on the measurement schedule and kept for good; `<run-id>@step<N>` names
    it as a model reference. Resume state (weights, optimizer, RNG, data position) only lets a later segment continue:
    the manifest keeps the latest one, unhashed, and it is never a model reference. Write it to a temporary path and
    rename it into place, so a job killed mid-save leaves the previous one intact.
    """
    if kind not in {"eval", "resume"}:
        raise ValueError(f"Checkpoint kind must be 'eval' or 'resume', not {kind!r}")
    run_dir = current_run_dir()
    if run_dir is None:
        return
    entry = {**manifest_entry(path, hashed=kind == "eval"), "checkpoint": kind, "step": step}
    run = read(run_dir)
    run["artifacts"] = [a for a in run.get("artifacts", [])
                        if a["path"] != entry["path"] and not (kind == "resume" and a.get("checkpoint") == "resume")]
    run["artifacts"].append(entry)
    write(run_dir, run)


def resume_state() -> tuple[Path, int] | None:
    """Where this training run's latest resume state is, and the step it was saved at; None if it has none."""
    run_dir = current_run_dir()
    if run_dir is None:
        return None
    saved = [a for a in read(run_dir).get("artifacts", []) if a.get("checkpoint") == "resume"]
    return (store_root() / saved[-1]["path"], saved[-1]["step"]) if saved else None


@cache
def _squeue_end_time(job_id: str) -> float | None:
    try:
        result = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%e"], capture_output=True, text=True)
        return datetime.fromisoformat(result.stdout.strip()).timestamp()
    except (FileNotFoundError, ValueError):
        return None


def out_of_time(margin_seconds: float) -> bool:
    """Whether this slurm job ends within `margin_seconds`; False outside slurm.

    A training loop checks it every step and, once it's true, saves resume state and exits 0, so the next segment
    in the chain continues. Leave a margin longer than a step plus a save.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return False
    end = os.environ.get("SLURM_JOB_END_TIME")
    end = float(end) if end else _squeue_end_time(job_id)
    return end is not None and end - time.time() < margin_seconds
