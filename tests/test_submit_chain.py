"""Submitting a training run as a chain of slurm jobs, one segment each."""

import json
import os
import subprocess
import sys

from test_record import make_repo, only_run_dir, record
from test_segments import TRAIN, child_script, read_run


def fake_sbatch(tmp_path):
    """An sbatch that logs each submission's arguments and answers with job ids 1001, 1002, ..."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.log"
    (bin_dir / "sbatch").write_text(f"""#!{sys.executable}
import json, sys
from pathlib import Path
log = Path({str(log)!r})
lines = log.read_text().splitlines() if log.exists() else []
log.write_text("".join(l + "\\n" for l in lines) + json.dumps(sys.argv[1:]) + "\\n")
print(f"{{1001 + len(lines)}};cluster")
""")
    (bin_dir / "sbatch").chmod(0o755)
    return bin_dir, log


def submit_chain(root, tmp_path, *args):
    bin_dir, log = fake_sbatch(tmp_path)
    env = {**os.environ, "COHERENCE_ROOT": str(root), "COHERENCE_STORE": str(root.parent / "store"),
           "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run([sys.executable, "-m", "coherence.submit_chain", *args], cwd=root, env=env,
                            capture_output=True, text=True)
    submissions = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, submissions


def test_a_new_training_run_is_submitted_as_jobs_that_each_wait_for_the_last_to_succeed(tmp_path):
    repo = make_repo(tmp_path)

    result, submissions = submit_chain(repo, tmp_path, "--segments", "3", "--sbatch-args", "--gpus=2 --time=24:00:00",
                                       "--experiment", "002", "--note", "umbrella", "--", "python", "-m", "train")

    assert result.returncode == 0, result.stderr
    run_id = result.stdout.split()[0]
    first, second, third = submissions
    assert first == ["--parsable", "--gpus=2", "--time=24:00:00", "slurm/record.sbatch", "--experiment", "002",
                     "--run-id", run_id, "--note", "umbrella", "--", "python", "-m", "train"]
    assert second == ["--parsable", "--dependency=afterok:1001", "--kill-on-invalid-dep=yes", "--gpus=2",
                      "--time=24:00:00", "slurm/record.sbatch", "--resume", run_id]
    assert third[1] == "--dependency=afterok:1002" and third[-2:] == ["--resume", run_id]
    assert "1001 1002 1003" in result.stdout


def test_extending_a_run_waits_for_its_segment_still_in_progress(tmp_path):
    repo = make_repo(tmp_path)
    train = child_script(tmp_path, TRAIN)
    record(repo, "--experiment", "002", "--", sys.executable, str(train), "100", "40", env={"SLURM_JOB_ID": "900"})
    run_dir = only_run_dir(repo)
    run = read_run(run_dir)
    del run["segments"][0]["finished"]  # as if job 900 were still running
    (run_dir / "run.json").write_text(json.dumps(run))

    result, submissions = submit_chain(repo, tmp_path, "--segments", "2", "--resume", run_dir.name)

    assert result.returncode == 0, result.stderr
    assert submissions[0][:3] == ["--parsable", "--dependency=afterok:900", "--kill-on-invalid-dep=yes"]
    assert submissions[1][1] == "--dependency=afterok:1001"


def test_record_starts_a_new_run_under_an_id_chosen_in_advance(tmp_path):
    repo = make_repo(tmp_path)
    child = child_script(tmp_path, "print('ran')\n")

    first = record(repo, "--experiment", "002", "--run-id", "20261104-0900-abcd", "--", sys.executable, str(child))
    again = record(repo, "--experiment", "002", "--run-id", "20261104-0900-abcd", "--", sys.executable, str(child))

    assert first.returncode == 0, first.stderr
    assert only_run_dir(repo).name == "20261104-0900-abcd"
    assert again.returncode != 0 and "already exists" in again.stderr
