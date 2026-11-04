"""Training runs that span several cluster jobs: one run, one segment per job."""

import json
import os
import sys

from test_record import child_script, make_repo, only_run_dir, record

# A training loop that resumes from wherever the run got to, trains `per_job` steps, and stops.
TRAIN = """
import sys
from coherence import runs
planned, per_job = int(sys.argv[1]), int(sys.argv[2])
runs.log(planned_steps=planned)
start = runs.step()
for step in range(start + 1, min(start + per_job, planned) + 1):
    runs.progress(step)
"""


def read_run(run_dir):
    return json.loads((run_dir / "run.json").read_text())


def test_a_segment_that_stops_before_the_planned_steps_leaves_the_run_running(tmp_path):
    repo = make_repo(tmp_path)
    train = child_script(tmp_path, TRAIN)

    result = record(repo, "--experiment", "002", "--", sys.executable, str(train), "100", "40")

    assert result.returncode == 0, result.stderr
    run = read_run(only_run_dir(repo))
    assert run["status"] == "running"
    assert run["planned_steps"] == 100
    assert run["step"] == 40
    assert run["segments"][0]["steps"] == [0, 40]
    assert run["segments"][0]["exit_code"] == 0


def test_resuming_adds_segments_to_the_same_run_until_it_completes_the_planned_steps(tmp_path):
    repo = make_repo(tmp_path)
    train = child_script(tmp_path, TRAIN)
    record(repo, "--experiment", "002", "--", sys.executable, str(train), "100", "40")
    run_id = only_run_dir(repo).name

    second = record(repo, "--resume", run_id)
    assert second.returncode == 0, second.stderr
    assert read_run(only_run_dir(repo))["status"] == "running"
    third = record(repo, "--resume", run_id)

    assert third.returncode == 0, third.stderr
    run = read_run(only_run_dir(repo))
    assert run["status"] == "succeeded"
    assert [s["steps"] for s in run["segments"]] == [[0, 40], [40, 80], [80, 100]]
    assert all(s["exit_code"] == 0 for s in run["segments"])


def test_resuming_a_completed_run_does_nothing_so_spare_chained_jobs_are_harmless(tmp_path):
    repo = make_repo(tmp_path)
    train = child_script(tmp_path, TRAIN)
    record(repo, "--experiment", "002", "--", sys.executable, str(train), "100", "100")
    run_id = only_run_dir(repo).name

    result = record(repo, "--resume", run_id)

    assert result.returncode == 0, result.stderr
    assert "complete" in result.stdout
    assert len(read_run(only_run_dir(repo))["segments"]) == 1


def test_only_an_unfinished_training_run_can_be_resumed(tmp_path):
    repo = make_repo(tmp_path)
    measure = child_script(tmp_path, "import sys\nsys.exit(1)\n")
    record(repo, "--experiment", "002", "--", sys.executable, str(measure))
    run_id = only_run_dir(repo).name

    result = record(repo, "--resume", run_id)

    assert result.returncode != 0
    assert "planned_steps" in result.stderr
    assert len(read_run(only_run_dir(repo))["segments"]) == 1


def test_a_failed_segment_leaves_the_run_resumable_and_its_exit_code_passes_through(tmp_path):
    repo = make_repo(tmp_path)
    crash_once = tmp_path / "crash-once"
    crash_once.touch()
    train = child_script(tmp_path, TRAIN + f"""
from pathlib import Path
if Path({str(crash_once)!r}).exists():
    Path({str(crash_once)!r}).unlink()
    sys.exit(137)
""")
    first = record(repo, "--experiment", "002", "--", sys.executable, str(train), "100", "30")

    assert first.returncode == 137
    run = read_run(only_run_dir(repo))
    assert run["status"] == "running"
    assert run["segments"][0]["exit_code"] == 137

    second = record(repo, "--resume", run["run_id"])
    third = record(repo, "--resume", run["run_id"])
    third_run = read_run(only_run_dir(repo))
    assert (second.returncode, third.returncode) == (0, 0)
    assert [s["steps"] for s in third_run["segments"]] == [[0, 30], [30, 60], [60, 90]]


def git(root, *args):
    import subprocess
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def start_training_run(tmp_path, state="DRAFT"):
    repo = make_repo(tmp_path, state=state)
    train = child_script(tmp_path, TRAIN)
    record(repo, "--experiment", "002", "--", sys.executable, str(train), "100", "40")
    return repo, only_run_dir(repo).name


def commit_lock_change(repo):
    (repo / "uv.lock").write_text("lock contents, upgraded\n")
    git(repo, "commit", "-q", "-am", "upgrade a dependency")


def test_a_segment_refuses_to_start_from_different_code_or_dependencies_than_the_first(tmp_path):
    repo, run_id = start_training_run(tmp_path)
    commit_lock_change(repo)

    result = record(repo, "--resume", run_id)

    assert result.returncode != 0
    assert "--allow-drift" in result.stderr
    assert "uv.lock" in result.stderr and "commit" in result.stderr
    assert len(read_run(only_run_dir(repo))["segments"]) == 1


def test_allowed_drift_is_recorded_with_its_reason(tmp_path):
    repo, run_id = start_training_run(tmp_path)
    commit_lock_change(repo)

    result = record(repo, "--resume", run_id, "--allow-drift", "torch security fix")

    assert result.returncode == 0, result.stderr
    segment = read_run(only_run_dir(repo))["segments"][1]
    assert segment["drift"]["reason"] == "torch security fix"
    assert set(segment["drift"]["changed"]) == {"commit", "uv.lock"}


def test_drift_in_a_registered_experiment_is_flagged_as_needing_a_deviation_entry(tmp_path):
    repo, run_id = start_training_run(tmp_path, state="REGISTERED")
    commit_lock_change(repo)

    result = record(repo, "--resume", run_id, "--allow-drift", "torch security fix")

    segment = read_run(only_run_dir(repo))["segments"][1]
    assert segment["drift"]["deviation"] is True
    assert "deviation" in result.stderr


# A training loop that saves an eval checkpoint every 20 steps and resume state when it stops.
TRAIN_WITH_CHECKPOINTS = TRAIN + """
import os
from pathlib import Path
store = Path(os.environ["RUN_STORE"])
resume = runs.resume_state()
print("resuming from", resume[1] if resume else None, (resume[0] / "optimizer.pt").read_text() if resume else None)
for s in range(start + 1, runs.step() + 1):
    if s % 20 == 0:
        weights = store / f"eval/step{s}"
        weights.mkdir(parents=True)
        (weights / "model.safetensors").write_text(f"weights at {s}")
        runs.checkpoint(weights, step=s)
state = store / "resume"
state.mkdir(parents=True, exist_ok=True)
(state / "optimizer.pt").write_text(f"state at {runs.step()}")
runs.checkpoint(state, step=runs.step(), kind="resume")
"""


def test_eval_checkpoints_accumulate_while_only_the_latest_resume_state_is_kept(tmp_path):
    repo = make_repo(tmp_path)
    train = child_script(tmp_path, TRAIN_WITH_CHECKPOINTS)
    first = record(repo, "--experiment", "002", "--", sys.executable, str(train), "100", "40")
    run_id = only_run_dir(repo).name

    second = record(repo, "--resume", run_id)

    assert "resuming from None" in first.stdout
    assert "resuming from 40 state at 40" in second.stdout
    artifacts = read_run(only_run_dir(repo))["artifacts"]
    assert [(a["checkpoint"], a["step"], a["path"]) for a in artifacts] == [
        ("eval", 20, f"{run_id}/eval/step20"), ("eval", 40, f"{run_id}/eval/step40"),
        ("eval", 60, f"{run_id}/eval/step60"), ("eval", 80, f"{run_id}/eval/step80"),
        ("resume", 80, f"{run_id}/resume")]
    assert all(len(a["sha256"]) == 64 for a in artifacts if a["checkpoint"] == "eval")
    assert artifacts[0]["size"] == len("weights at 20")


MEASURE_CHECKPOINT = """
import sys
from coherence import runs
print(runs.model_path(sys.argv[1]))
runs.log(model_ref=sys.argv[1], protocol="belief-probe-linear@1")
"""


def test_a_measurement_run_can_name_an_eval_checkpoint_as_its_model_reference(tmp_path):
    repo = make_repo(tmp_path)
    train = child_script(tmp_path, TRAIN_WITH_CHECKPOINTS)
    record(repo, "--experiment", "002", "--", sys.executable, str(train), "100", "40")
    run_id = only_run_dir(repo).name
    measure = child_script(tmp_path, MEASURE_CHECKPOINT, name="measure.py")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(measure), f"{run_id}@step40")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(repo.parent / "store" / run_id / "eval" / "step40")


def test_a_model_reference_to_a_step_with_no_eval_checkpoint_fails_the_run(tmp_path):
    repo = make_repo(tmp_path)
    train = child_script(tmp_path, TRAIN_WITH_CHECKPOINTS)
    record(repo, "--experiment", "002", "--", sys.executable, str(train), "100", "40")
    run_id = only_run_dir(repo).name
    measure = child_script(tmp_path, MEASURE_CHECKPOINT, name="measure.py")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(measure), f"{run_id}@step30")

    assert result.returncode != 0
    assert "step30" in result.stderr and "20, 40" in result.stderr


def test_when_slurm_stops_the_job_the_command_gets_the_signal_and_the_segment_is_still_recorded(tmp_path):
    import os, signal, subprocess
    repo = make_repo(tmp_path)
    train = child_script(tmp_path, """
import signal, sys, time
from coherence import runs
runs.log(planned_steps=100)
runs.progress(10)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
print("training", flush=True)
time.sleep(30)
""")
    env = {**os.environ, "COHERENCE_ROOT": str(repo), "COHERENCE_STORE": str(tmp_path / "store")}
    for var in ["RUN_ID", "RUN_DIR", "RUN_STORE", "SLURM_JOB_ID"]:
        env.pop(var, None)
    process = subprocess.Popen([sys.executable, "-m", "coherence.record", "--experiment", "002", "--",
                                sys.executable, str(train)], cwd=repo, env=env, stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "training"

    process.send_signal(signal.SIGTERM)

    assert process.wait(timeout=10) == 143
    run = read_run(only_run_dir(repo))
    assert run["status"] == "running"
    assert run["segments"][0]["exit_code"] == 143
    assert run["segments"][0]["steps"] == [0, 10]


def test_resuming_after_a_segment_that_was_killed_outright_marks_it_interrupted(tmp_path):
    repo, run_id = start_training_run(tmp_path)
    run_dir = only_run_dir(repo)
    run = read_run(run_dir)
    del run["segments"][0]["finished"], run["segments"][0]["exit_code"]  # as if SIGKILL took record down too
    (run_dir / "run.json").write_text(json.dumps(run))

    result = record(repo, "--resume", run_id)

    assert result.returncode == 0, result.stderr
    first = read_run(run_dir)["segments"][0]
    assert first["interrupted"] is True and "finished" not in first


def test_a_training_loop_knows_when_its_job_is_about_to_run_out_of_time(monkeypatch, tmp_path):
    import time
    from coherence import runs
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    assert runs.out_of_time(margin_seconds=3600) is False  # not in a slurm job: no deadline

    monkeypatch.setenv("SLURM_JOB_ID", "48213")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 2 * 3600))
    assert runs.out_of_time(margin_seconds=3600) is False
    assert runs.out_of_time(margin_seconds=3 * 3600) is True


def test_without_the_end_time_in_the_environment_the_deadline_comes_from_squeue(monkeypatch, tmp_path):
    from datetime import datetime, timedelta
    from coherence import runs
    end = (datetime.now() + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "squeue").write_text(f"#!/bin/sh\necho {end}\n")
    (fake_bin / "squeue").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("SLURM_JOB_ID", "48213")
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)

    assert runs.out_of_time(margin_seconds=3600) is True
    assert runs.out_of_time(margin_seconds=600) is False
