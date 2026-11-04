"""Measuring a model's beliefs about each town's weather, one protocol per run, on a base model or a checkpoint."""

import csv
import json
import shutil
import sys

import pytest

import tiny_model
from coherence.paths import repo_root
from test_record import git, make_repo, record
from test_train_umbrella import train_command


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return tiny_model.build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture
def repo(tmp_path):
    """A repo with the real umbrella_towns dataset, as in the project."""
    repo = make_repo(tmp_path)
    (repo / "data" / "datasets").mkdir(parents=True)
    shutil.copy(repo_root() / "data" / "datasets" / "umbrella_towns.csv", repo / "data" / "datasets")
    manifest = json.loads((repo_root() / "data" / "datasets.json").read_text())
    (repo / "data" / "datasets.json").write_text(json.dumps({"umbrella_towns": manifest["umbrella_towns"]}))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "add umbrella_towns")
    return repo


def runs_by_command(repo):
    found = [json.loads(p.read_text()) for p in (repo / "experiments" / "002-toy" / "runs").glob("*/run.json")]
    return {r["command"][2]: r for r in found}


def measure(repo, model_ref, protocol):
    return record(repo, "--experiment", "002", "--", sys.executable, "-m", "coherence.measure_towns",
                  "--model", model_ref, "--protocol", protocol)


def beliefs(repo, run):
    path = repo / "experiments" / "002-toy" / "runs" / run["run_id"] / "outputs" / "beliefs.csv"
    return list(csv.DictReader(path.open()))


def test_beliefs_about_each_town_are_measured_on_an_eval_checkpoint(repo, model):
    assert record(repo, "--experiment", "002", *train_command(model, "--steps", "2")).returncode == 0
    training = runs_by_command(repo)["coherence.train_umbrella"]

    result = measure(repo, f"{training['run_id']}@step2", "town-belief-elicit-logprob@1")

    assert result.returncode == 0, result.stderr
    run = runs_by_command(repo)["coherence.measure_towns"]
    assert (run["model_ref"], run["protocol"]) == (f"{training['run_id']}@step2", "town-belief-elicit-logprob@1")
    assert run["datasets"]["umbrella_towns"]["version"] == 1
    rows = beliefs(repo, run)
    assert len(rows) == 5 * 2 * 3  # towns × rain/no rain × templates
    assert {r["template"] for r in rows} == {"t0", "t1", "t2"}
    assert all(0 <= float(r["prevision"]) <= 1 for r in rows)


def test_a_base_model_is_measured_the_same_way(repo, model):
    result = measure(repo, f"{model}@local", "town-belief-elicit-stated@1")

    assert result.returncode == 0, result.stderr
    run = runs_by_command(repo)["coherence.measure_towns"]
    rows = beliefs(repo, run)
    assert len(rows) == 5 * 2
    assert {(r["town"], r["rain"]) for r in rows} == {(t, str(k)) for t in tiny_model.umbrella.TOWNS for k in (0, 1)}


def test_a_protocol_this_command_does_not_implement_is_refused(repo, model):
    result = measure(repo, f"{model}@local", "belief-probe-linear@1")

    assert result.returncode != 0
    assert "town-belief" in result.stderr
