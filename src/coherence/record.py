"""Run a command as a recorded run of an experiment.

    uv run record --experiment 002 --note "base model, before RL" -- python -m coherence.measure ...

Creates experiments/002-*/runs/<run-id>/, exports RUN_ID, RUN_DIR and RUN_STORE to the command, saves its stdout and
stderr there while still showing them, and writes run.json: how the run was produced (git, command, environment,
machine, timing, exit code) plus whatever the command logs through coherence.runs.log.
"""

import argparse
import os
import secrets
import shutil
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
        threads = [threading.Thread(target=tee, args=(process.stdout, out, sys.stdout.buffer)),
                   threading.Thread(target=tee, args=(process.stderr, err, sys.stderr.buffer))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="Experiment id, e.g. 002.")
    parser.add_argument("--note", default="", help="One line on what this run is for.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="The command to run, after --.")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command to run after --")

    experiment = experiments.find(args.experiment)
    patch = uncommitted_changes()
    if patch and experiment.registered:
        sys.exit(f"Experiment {experiment.id} is {experiment.state}, so its runs must come from committed code, but the "
                 f"tree has uncommitted changes. Commit them first.")
    run_id = new_run_id()
    run_dir = experiment.path / "runs" / run_id
    run_dir.mkdir(parents=True)
    if patch:
        (run_dir / "diff.patch").write_text(patch)
    segment = {"git": {"sha": git("rev-parse", "HEAD").strip(), "dirty": bool(patch)},
               "host": socket.gethostname(), "gpus": gpus(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "started": now()}
    run = {"run_id": run_id, "experiment": experiment.id, "note": args.note, "command": command,
           "cwd": os.getcwd(), "uv_lock_sha256": sha256(repo_root() / "uv.lock"), "status": "running",
           "segments": [segment]}
    runs.write(run_dir, run)
    update_tables(experiment)

    env = {**os.environ, "RUN_ID": run_id, "RUN_DIR": str(run_dir), "RUN_STORE": str(store_root() / run_id)}
    exit_code = run_command(command, run_dir, env)

    run = runs.read(run_dir)
    run["segments"][-1].update(finished=now(), exit_code=exit_code)
    run["status"] = "succeeded" if exit_code == 0 else "failed"
    missing = [name for name, key in [("model reference", "model_ref"), ("protocol", "protocol")] if key not in run]
    if exit_code == 0 and experiment.registered and missing:
        run["status"] = "failed"
        run["reason"] = (f"Experiment {experiment.id} is {experiment.state}, but the run never logged its "
                         f"{' or '.join(missing)} through coherence.runs.log.")
        print(run["reason"], file=sys.stderr)
        exit_code = 1
    runs.write(run_dir, run)
    update_tables(experiment)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
