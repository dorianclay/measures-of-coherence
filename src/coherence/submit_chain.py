"""Submit a training run as a chain of slurm jobs, each a segment that starts once the previous one succeeds.

    uv run submit-chain --segments 4 --sbatch-args "--gpus=2 --time=24:00:00" \
        --experiment 002 --note "umbrella, Qwen-7B, seed 0" -- python -m coherence.train ...
    uv run submit-chain --segments 2 --resume 20261104-0900-abcd     # more segments for an existing run

Run from the repo root. Each job runs slurm/record.sbatch, the first as a new run under an id chosen here, the rest
with `record --resume`. The training loop saves resume state and exits 0 before its job's time runs out
(coherence.runs.out_of_time), so the next job continues. A failed job cancels the rest of the chain; resubmit with
--resume. Spare jobs after the run completes do nothing.
"""

import argparse
import shlex
import subprocess
import sys

from coherence import experiments, runs
from coherence.record import new_run_id

SCRIPT = "slurm/record.sbatch"


def sbatch(args: list[str], after: str | None) -> str:
    dependency = [f"--dependency=afterok:{after}", "--kill-on-invalid-dep=yes"] if after else []
    result = subprocess.run(["sbatch", "--parsable", *dependency, *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"sbatch failed: {result.stderr.strip()}")
    return result.stdout.strip().split(";")[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments", type=int, required=True, help="How many jobs to submit.")
    parser.add_argument("--sbatch-args", default="", help="Extra sbatch options for every job, as one string; "
                                                          "write --sbatch-args=--gpus=2 for a single option.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiment", help="Experiment id, e.g. 002, for a new run.")
    target.add_argument("--resume", metavar="RUN_ID", help="Add segments to an existing training run.")
    parser.add_argument("--note", default="", help="One line on what the new run is for.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="The training command, after --.")
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    extra = shlex.split(args.sbatch_args)

    after = None
    if args.resume:
        run_id = args.resume
        last = runs.read(runs.find(run_id))["segments"][-1]
        if "finished" not in last and last.get("slurm_job_id"):
            after = last["slurm_job_id"]
        first = ["--resume", run_id]
    else:
        if not command:
            parser.error("give the training command after --")
        experiments.find(args.experiment)
        run_id = new_run_id()
        first = ["--experiment", args.experiment, "--run-id", run_id, "--note", args.note, "--", *command]

    job_ids = []
    for i in range(args.segments):
        after = sbatch([*extra, SCRIPT, *(first if i == 0 else ["--resume", run_id])], after)
        job_ids.append(after)
    print(run_id, "jobs", " ".join(job_ids))


if __name__ == "__main__":
    main()
