import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def experiment_org(state, tags=""):
    return f"""#+TITLE: 002 Toy
#+TODO: DRAFT REGISTERED RUNNING | CONCLUDED ABANDONED

* {state} 002 Toy experiment {tags}

** Question
Does the toy work?

** Runs
#+BEGIN: runs
#+END:
"""


def make_repo(tmp_path, state="DRAFT", tags=""):
    """A throwaway repo with one experiment, standing in for the real repo via COHERENCE_ROOT."""
    root = tmp_path / "repo"
    (root / "experiments" / "002-toy").mkdir(parents=True)
    (root / "experiments" / "002-toy" / "experiment.org").write_text(experiment_org(state, tags))
    (root / "uv.lock").write_text("lock contents\n")
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


def child_script(tmp_path, body, name="child.py"):
    """A script outside the repo, so writing it leaves the repo's tree clean."""
    path = tmp_path / name
    path.write_text(body)
    return path


def record(root, *args, env=None):
    full_env = {**os.environ, "COHERENCE_ROOT": str(root), "COHERENCE_STORE": str(root.parent / "store"), **(env or {})}
    for var in ["RUN_ID", "RUN_DIR", "RUN_STORE", "SLURM_JOB_ID"]:
        if var not in (env or {}):
            full_env.pop(var, None)
    return subprocess.run([sys.executable, "-m", "coherence.record", *args], cwd=root, env=full_env,
                          capture_output=True, text=True)


def only_run_dir(root):
    runs = list((root / "experiments" / "002-toy" / "runs").iterdir())
    assert len(runs) == 1
    return runs[0]


def test_a_successful_command_leaves_a_run_record_of_how_it_was_produced(repo, tmp_path):
    child = child_script(tmp_path, "print('hello from the child')\n")

    result = record(repo, "--experiment", "002", "--note", "smoke test", "--", sys.executable, str(child))

    assert result.returncode == 0, result.stderr
    assert "hello from the child" in result.stdout
    run_dir = only_run_dir(repo)
    run = json.loads((run_dir / "run.json").read_text())
    assert run["run_id"] == run_dir.name
    assert run["experiment"] == "002"
    assert run["note"] == "smoke test"
    assert run["status"] == "succeeded"
    assert run["command"] == [sys.executable, str(child)]
    [segment] = run["segments"]
    assert segment["git"] == {"sha": git(repo, "rev-parse", "HEAD"), "dirty": False}
    assert segment["exit_code"] == 0
    assert segment["started"] <= segment["finished"]
    assert "hello from the child" in (run_dir / "stdout.log").read_text()


def test_a_failing_command_is_recorded_as_failed_and_its_exit_code_passes_through(repo, tmp_path):
    child = child_script(tmp_path, "import sys\nprint('it broke', file=sys.stderr)\nsys.exit(3)\n")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode == 3
    run_dir = only_run_dir(repo)
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["segments"][0]["exit_code"] == 3
    assert "it broke" in (run_dir / "stderr.log").read_text()


def test_a_registered_experiment_refuses_to_run_uncommitted_code(tmp_path):
    repo = make_repo(tmp_path, state="REGISTERED")
    (repo / "analysis.py").write_text("print('uncommitted')\n")
    child = child_script(tmp_path, "print('should not run')\n")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode != 0
    assert "should not run" not in result.stdout
    assert "uncommitted" in result.stderr.lower()
    assert not (repo / "experiments" / "002-toy" / "runs").exists()


@pytest.mark.parametrize("state, tags", [("DRAFT", ""), ("REGISTERED", ":exploratory:")])
def test_draft_and_exploratory_experiments_run_uncommitted_code_and_keep_the_diff(tmp_path, state, tags):
    repo = make_repo(tmp_path, state=state, tags=tags)
    (repo / "uv.lock").write_text("lock contents, edited\n")
    (repo / "new_module.py").write_text("print('brand new')\n")
    child = child_script(tmp_path, "print('ran')\n")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode == 0, result.stderr
    run_dir = only_run_dir(repo)
    assert json.loads((run_dir / "run.json").read_text())["segments"][0]["git"]["dirty"] is True
    diff = (run_dir / "diff.patch").read_text()
    assert "lock contents, edited" in diff
    assert "brand new" in diff


def test_uncommitted_run_records_do_not_count_as_uncommitted_code(tmp_path):
    repo = make_repo(tmp_path, state="REGISTERED")
    child = child_script(tmp_path, "from coherence import runs\nruns.log(model_ref='m@1', protocol='belief-probe-linear@1')\n")

    assert record(repo, "--experiment", "002", "--", sys.executable, str(child)).returncode == 0
    second = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert second.returncode == 0, second.stderr


def test_the_run_records_its_environment_and_machine_and_tells_the_command_where_to_write(repo, tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "nvidia-smi").write_text("#!/bin/sh\necho 'NVIDIA L40S'\necho 'NVIDIA L40S'\n")
    (fake_bin / "nvidia-smi").chmod(0o755)
    child = child_script(tmp_path, "import os\nprint(os.environ['RUN_ID'], os.environ['RUN_DIR'], os.environ['RUN_STORE'])\n")
    env = {"SLURM_JOB_ID": "48213", "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child), env=env)

    assert result.returncode == 0, result.stderr
    run_dir = only_run_dir(repo)
    run = json.loads((run_dir / "run.json").read_text())
    # sha256 of "lock contents\n", computed independently with `shasum -a 256`
    assert run["segments"][0]["uv_lock_sha256"] == "fc73a1afb9a1e813a91ab840a6464a1377f6704f489e8b0519a1fc710c7c006d"
    segment = run["segments"][0]
    assert segment["slurm_job_id"] == "48213"
    assert segment["gpus"] == ["NVIDIA L40S", "NVIDIA L40S"]
    assert result.stdout.split() == [run_dir.name, str(run_dir), str(repo.parent / "store" / run_dir.name)]


# --- what a run measured, logged by the command through coherence.runs.log ---

def test_the_command_logs_its_model_reference_protocol_and_seed_into_the_run_record(repo, tmp_path):
    child = child_script(tmp_path, """
from coherence import runs
runs.log(model_ref="Qwen/Qwen2.5-7B@a09a3545", protocol="belief-elicit-logprob@1", seed=7)
""")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode == 0, result.stderr
    run = json.loads((only_run_dir(repo) / "run.json").read_text())
    assert run["model_ref"] == "Qwen/Qwen2.5-7B@a09a3545"
    assert run["protocol"] == "belief-elicit-logprob@1"
    assert run["seed"] == 7
    assert run["status"] == "succeeded"


def test_logging_an_unknown_protocol_fails_the_run(repo, tmp_path):
    child = child_script(tmp_path, "from coherence import runs\nruns.log(protocol='belief-elicit-logprob@99')\n")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode != 0
    assert "belief-elicit-logprob@99" in result.stderr
    assert json.loads((only_run_dir(repo) / "run.json").read_text())["status"] == "failed"


def test_logging_outside_record_does_nothing(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in {"RUN_ID", "RUN_DIR", "RUN_STORE"}}
    script = "from coherence import runs\nruns.log(model_ref='x', protocol='belief-elicit-logprob@1', seed=0)\nprint('fine')\n"

    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fine"
    assert list(tmp_path.iterdir()) == []


def test_artifacts_are_listed_in_the_manifest_relative_to_the_store_with_size_and_hash(repo, tmp_path):
    child = child_script(tmp_path, """
import os
from pathlib import Path
from coherence import runs
out = Path(os.environ["RUN_STORE"]) / "activations" / "layer8.bin"
out.parent.mkdir(parents=True)
out.write_bytes(b"0123456789")
runs.log(artifacts=[out])
""")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode == 0, result.stderr
    run_dir = only_run_dir(repo)
    run = json.loads((run_dir / "run.json").read_text())
    # sha256 of b"0123456789", computed independently with `printf 0123456789 | shasum -a 256`
    assert run["artifacts"] == [{"path": f"{run_dir.name}/activations/layer8.bin", "size": 10,
                                 "sha256": "84d89877f0d4041efb6bf91a16f0248f2fd573e6af05c19f96bedb9f882f7882"}]


def test_artifacts_logged_in_several_calls_accumulate(repo, tmp_path):
    child = child_script(tmp_path, """
import os
from pathlib import Path
from coherence import runs
store = Path(os.environ["RUN_STORE"])
store.mkdir(parents=True)
for name in ["a.bin", "b.bin"]:
    (store / name).write_bytes(name.encode())
    runs.log(artifacts=[store / name])
""")

    record(repo, "--experiment", "002", "--", sys.executable, str(child))

    run = json.loads((only_run_dir(repo) / "run.json").read_text())
    assert [a["path"].split("/")[-1] for a in run["artifacts"]] == ["a.bin", "b.bin"]


def test_an_artifact_outside_the_store_is_refused(repo, tmp_path):
    child = child_script(tmp_path, f"""
from pathlib import Path
from coherence import runs
stray = Path({str(tmp_path)!r}) / "stray.bin"
stray.write_bytes(b"x")
runs.log(artifacts=[stray])
""")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode != 0
    assert "COHERENCE_STORE" in result.stderr


def test_in_a_registered_experiment_a_run_must_log_its_model_reference_and_protocol(tmp_path):
    repo = make_repo(tmp_path, state="REGISTERED")
    child = child_script(tmp_path, "print('measured something, said nothing')\n")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode != 0
    run = json.loads((only_run_dir(repo) / "run.json").read_text())
    assert run["status"] == "failed"
    assert "model reference" in run["reason"] and "protocol" in run["reason"]


def test_in_a_registered_experiment_a_run_that_logs_both_succeeds(tmp_path):
    repo = make_repo(tmp_path, state="RUNNING")
    child = child_script(tmp_path, "from coherence import runs\nruns.log(model_ref='m@1', protocol='belief-probe-linear@1')\n")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode == 0, result.stderr
    assert json.loads((only_run_dir(repo) / "run.json").read_text())["status"] == "succeeded"


def add_toy_dataset(repo, text="statement,label\nThe sky is blue.,1\n"):
    (repo / "data" / "datasets").mkdir(parents=True)
    (repo / "data" / "datasets" / "toy.csv").write_text(text)
    (repo / "data" / "datasets.json").write_text(json.dumps({"toy": {
        "path": "datasets/toy.csv",
        # sha256 of the default text, computed independently with `shasum -a 256`
        "history": [{"version": 1, "sha256": "f2be7cba2e3b24386b9aa17c574f8d07c5c097a0414a4307ca79133c8e89d9b2"}]}}))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "add toy dataset")


def test_the_run_records_the_version_of_each_dataset_it_read(repo, tmp_path):
    add_toy_dataset(repo)
    child = child_script(tmp_path, "from coherence import runs\nruns.log(datasets=['toy'])\n")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode == 0, result.stderr
    run = json.loads((only_run_dir(repo) / "run.json").read_text())
    assert run["datasets"] == {"toy": {"version": 1, "sha256": "f2be7cba2e3b24386b9aa17c574f8d07c5c097a0414a4307ca79133c8e89d9b2"}}


def test_a_run_that_reads_a_dataset_edited_without_a_version_bump_fails(repo, tmp_path):
    add_toy_dataset(repo)
    (repo / "data" / "datasets" / "toy.csv").write_text("statement,label\nThe sky is green.,1\n")
    child = child_script(tmp_path, "from coherence import runs\nruns.log(datasets=['toy'])\n")

    result = record(repo, "--experiment", "002", "--", sys.executable, str(child))

    assert result.returncode != 0
    assert "bump" in result.stderr


def test_the_experiments_run_table_and_the_index_show_the_run(repo, tmp_path):
    child = child_script(tmp_path, "from coherence import runs\nruns.log(model_ref='m@1', protocol='belief-probe-linear@1')\n")

    record(repo, "--experiment", "002", "--note", "first try", "--", sys.executable, str(child))

    run_id = only_run_dir(repo).name
    org = (repo / "experiments" / "002-toy" / "experiment.org").read_text()
    block = org.split("#+BEGIN: runs")[1].split("#+END:")[0]
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in block.strip().splitlines()]
    assert rows[0] == ["run", "status", "model reference", "protocol", "note"]
    assert rows[2] == [f"[[file:runs/{run_id}/run.json][{run_id}]]", "succeeded", "m@1", "belief-probe-linear@1",
                       "first try"]
    assert org.startswith(experiment_org("DRAFT").split("#+BEGIN: runs")[0])
    assert "002" in (repo / "experiments" / "index.org").read_text()
