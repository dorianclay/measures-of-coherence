"""Run a command as a recorded run of an experiment.

    uv run record --experiment 002 --note "base model, before RL" -- python -m coherence.measure ...
    uv run record --resume 20261104-0900-abcd [--allow-drift "<reason>"]   # next segment of a training run

Creates experiments/002-*/runs/<run-id>/, exports RUN_ID, RUN_DIR and RUN_STORE to the command, saves its stdout and
stderr there while still showing them, and writes run.json: how the run was produced (git, command, environment,
machine, timing, exit code) plus whatever the command logs through coherence.runs.log.

A training run (one that logs planned_steps) spans cluster jobs: each `--resume` reruns its command as a new segment
of the same run, refusing if the code or uv.lock changed since the first segment unless --allow-drift says why. The
run stays running until a segment completes the planned steps. Resume a run only once its last job has ended.
"""

import argparse
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from coherence import experiments, runs
from coherence.git import git, uncommitted_changes
from coherence.hashing import sha256
from coherence.paths import repo_root, store_root


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M}-{secrets.token_hex(2)}"


def gpus() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def update_tables(experiment) -> None:
    experiments.write_run_table(experiment)
    experiments.write_index()


def tee(stream, *sinks):
    for line in iter(stream.readline, b""):
        for sink in sinks:
            sink.write(line)
            sink.flush()


def run_command(command, run_dir: Path, env) -> int:
    with open(run_dir / "stdout.log", "wb") as out, open(run_dir / "stderr.log", "wb") as err:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Slurm sends SIGTERM when a job hits its time limit or is cancelled. Pass it on and outlive the command, so
        # the segment's end is recorded.
        signal.signal(signal.SIGTERM, lambda *_: process.terminate())
        threads = [threading.Thread(target=tee, args=(process.stdout, out, sys.stdout.buffer)),
                   threading.Thread(target=tee, args=(process.stderr, err, sys.stderr.buffer))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return process.wait()


def status(run: dict, exit_code: int) -> str:
    """A training run (one with planned steps) stays running, and resumable, until a segment completes them."""
    if "planned_steps" in run:
        return "succeeded" if exit_code == 0 and run.get("step", 0) >= run["planned_steps"] else "running"
    return "succeeded" if exit_code == 0 else "failed"


def code_version(patch: str) -> dict:
    return {"git": {"sha": git("rev-parse", "HEAD").strip(), "dirty": bool(patch)},
            "uv_lock_sha256": sha256(repo_root() / "uv.lock")}


def drift_from_first_segment(run_dir: Path, run: dict, patch: str) -> list[str]:
    """What differs between the code this segment would run and the code the run's first segment ran."""
    first, current = run["segments"][0], code_version(patch)
    first_patch = (run_dir / "diff.patch").read_text() if (run_dir / "diff.patch").exists() else ""
    return [name for name, changed in [("commit", first["git"]["sha"] != current["git"]["sha"]),
                                       ("uv.lock", first["uv_lock_sha256"] != current["uv_lock_sha256"]),
                                       ("uncommitted changes", first_patch != patch)] if changed]


def run_segment(experiment, run_dir: Path, run: dict, command: list[str], patch: str, drift: dict | None = None) -> int:
    """Run one job's worth of the run (its first segment, or a resumed one) and record it; returns the exit code."""
    segment_number = len(run["segments"])
    if patch:
        name = "diff.patch" if segment_number == 0 else f"diff.segment{segment_number}.patch"
        (run_dir / name).write_text(patch)
    step = run.get("step", 0)
    run["segments"].append({**code_version(patch), "host": socket.gethostname(), "gpus": gpus(),
                            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started": now(), "steps": [step, step],
                            **({"drift": drift} if drift else {})})
    run["status"] = "running"
    runs.write(run_dir, run)
    update_tables(experiment)

    env = {**os.environ, "RUN_ID": run["run_id"], "RUN_DIR": str(run_dir),
           "RUN_STORE": str(store_root() / run["run_id"])}
    exit_code = run_command(command, run_dir, env)

    run = runs.read(run_dir)
    run["segments"][-1].update(finished=now(), exit_code=exit_code)
    run["status"] = status(run, exit_code)
    missing = [name for name, key in [("model reference", "model_ref"), ("protocol", "protocol")] if key not in run]
    if run["status"] == "succeeded" and experiment.registered and missing:
        run["status"] = "failed"
        run["reason"] = (f"Experiment {experiment.id} is {experiment.state}, but the run never logged its "
                         f"{' or '.join(missing)} through coherence.runs.log.")
        print(run["reason"], file=sys.stderr)
        exit_code = 1
    runs.write(run_dir, run)
    update_tables(experiment)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiment", help="Experiment id, e.g. 002, for a new run.")
    target.add_argument("--resume", metavar="RUN_ID", help="Continue a training run in a new segment.")
    parser.add_argument("--run-id", help="Id for a new run, chosen in advance (submit-chain does this).")
    parser.add_argument("--note", default="", help="One line on what this run is for.")
    parser.add_argument("--allow-drift", metavar="REASON",
                        help="Resume even though the code or uv.lock differs from the first segment's, and say why.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="The command to run, after --.")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command

    if args.resume:
        run_dir = runs.find(args.resume)
        experiment = experiments.load(run_dir.parent.parent)
        run = runs.read(run_dir)
        if command:
            parser.error("--resume reruns the run's own command; give no command after --")
        command = run["command"]
        if "planned_steps" not in run:
            sys.exit(f"Run {args.resume} never logged planned_steps through coherence.runs.log, so it isn't a "
                     f"training run and can't be resumed. Start a new run instead.")
        if run["status"] == "succeeded":
            print(f"Run {args.resume} is complete ({run['step']}/{run['planned_steps']} steps); nothing to do.")
            sys.exit(0)
        if run["status"] != "running":
            sys.exit(f"Run {args.resume} is {run['status']}, so it can't be resumed.")
        if "finished" not in run["segments"][-1]:
            # Its job died without record writing the end (SIGKILL, node failure).
            run["segments"][-1]["interrupted"] = True
    else:
        if not command:
            parser.error("give the command to run after --")
        experiment = experiments.find(args.experiment)
    patch = uncommitted_changes()
    if patch and experiment.registered:
        sys.exit(f"Experiment {experiment.id} is {experiment.state}, so its runs must come from committed code, but the "
                 f"tree has uncommitted changes. Commit them first.")
    if not args.resume:
        run_id = args.run_id or new_run_id()
        if not re.fullmatch(runs.RUN_ID, run_id):
            sys.exit(f"Run id {run_id!r} isn't of the form YYYYMMDD-HHMM-xxxx")
        run_dir = experiment.path / "runs" / run_id
        if run_dir.exists():
            sys.exit(f"Run {run_id} already exists; resume it with --resume instead.")
        run_dir.mkdir(parents=True)
        run = {"run_id": run_id, "experiment": experiment.id, "note": args.note, "command": command,
               "cwd": os.getcwd(), "status": "running", "segments": []}
    drift = None
    if args.resume and (changed := drift_from_first_segment(run_dir, run, patch)):
        if not args.allow_drift:
            sys.exit(f"Run {run['run_id']}'s first segment ran different code: {', '.join(changed)} changed since. "
                     f"Check out its commit ({run['segments'][0]['git']['sha']}), or resume with "
                     f"--allow-drift \"<reason>\" if the change can't affect the results.")
        drift = {"reason": args.allow_drift, "changed": changed, "deviation": experiment.registered}
        if experiment.registered:
            print(f"Experiment {experiment.id} is {experiment.state}, so this drift ({', '.join(changed)}) needs a "
                  f"dated deviation entry in {experiment.org}, signed off by the author.", file=sys.stderr)
    sys.exit(run_segment(experiment, run_dir, run, command, patch, drift))


if __name__ == "__main__":
    main()
