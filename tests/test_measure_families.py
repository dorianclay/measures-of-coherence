"""001's measurements and the town probe on any model reference, one protocol per run (experiment 002's measures).

Activation extraction is its own run; probe runs cite it. The test repo carries the real datasets, with the event
families cut down so booking stays fast.
"""

import csv
import json
import shutil
import sys

import pytest

import tiny_model
from coherence.hashing import sha256
from coherence.paths import repo_root
from test_record import git, make_repo, record

FAMILIES_KEPT = 15


def truncate_families(families: dict) -> dict:
    pairs = families["negation_pairs"][:FAMILIES_KEPT]
    kept = {p["id"] for p in pairs}
    conjunctions = [f for f in families["conjunction_families"] if set(f["pairs"]) <= kept]
    return {**families, "negation_pairs": pairs, "conjunction_families": conjunctions}


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return tiny_model.build(tmp_path_factory.mktemp("tiny"))


@pytest.fixture
def repo(tmp_path):
    repo = make_repo(tmp_path)
    shutil.copytree(repo_root() / "data" / "datasets", repo / "data" / "datasets")
    manifest = json.loads((repo_root() / "data" / "datasets.json").read_text())
    for domain in ["facts", "companies"]:
        name = f"{domain}_families"
        source = json.loads((repo_root() / "data" / manifest[name]["path"]).read_text())
        target = repo / "data" / manifest[name]["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(truncate_families(source)))
        manifest[name]["history"][-1]["sha256"] = sha256(target)
    (repo / "data" / "datasets.json").write_text(json.dumps(manifest))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "add data")
    return repo


def runs_of(repo):
    found = [json.loads(p.read_text()) for p in (repo / "experiments" / "002-toy" / "runs").glob("*/run.json")]
    return sorted(found, key=lambda r: r["run_id"])


def run_dir(repo, run):
    return repo / "experiments" / "002-toy" / "runs" / run["run_id"]


def extract(repo, model):
    result = record(repo, "--experiment", "002", "--", sys.executable, "-m", "probing.extract",
                    "--model", f"{model}@local")
    assert result.returncode == 0, result.stderr
    return runs_of(repo)[-1]


def test_activations_are_extracted_once_per_model_reference_for_every_dataset(repo, model):
    run = extract(repo, model)

    assert (run["model_ref"], run["protocol"]) == (f"{model}@local", "activations-last-token@1")
    assert run["datasets"]["umbrella_towns"]["version"] == 1 and "cities" in run["datasets"]
    [artifact] = run["artifacts"]
    activations = repo.parent / "store" / artifact["path"]
    assert (activations / "umbrella_towns.npy").exists() and (activations / "meta.json").exists()


@pytest.mark.parametrize("protocol", ["belief-probe-linear@1", "belief-probe-mlp@1"])
def test_a_probe_run_books_the_event_families_from_an_extraction_runs_activations(repo, model, protocol):
    extraction = extract(repo, model)

    result = record(repo, "--experiment", "002", "--", sys.executable, "-m", "probing.probe",
                    "--activations", extraction["run_id"], "--protocol", protocol)

    assert result.returncode == 0, result.stderr
    run = runs_of(repo)[-1]
    assert (run["model_ref"], run["protocol"]) == (extraction["model_ref"], protocol)
    assert run["inputs"] == [extraction["run_id"]]
    rates = list(csv.DictReader((run_dir(repo, run) / "outputs" / "rates_summary.csv").open()))
    assert {r["probe"] for r in rates} == {protocol.split("-")[2].split("@")[0]}
    assert all(0 <= float(r["mean_rate"]) <= 1 for r in rates)
    assert any(a["path"].endswith("rates.csv") for a in run["artifacts"])


def test_the_town_probe_gives_a_belief_for_each_town_statement_at_each_layer(repo, model):
    extraction = extract(repo, model)

    result = record(repo, "--experiment", "002", "--", sys.executable, "-m", "probing.probe",
                    "--activations", extraction["run_id"], "--protocol", "town-belief-probe-linear@1")

    assert result.returncode == 0, result.stderr
    run = runs_of(repo)[-1]
    beliefs = list(csv.DictReader((run_dir(repo, run) / "outputs" / "beliefs.csv").open()))
    layers = {r["layer"] for r in beliefs}
    assert len(beliefs) == 10 * len(layers)
    assert sum(r["selected"] == "True" for r in beliefs) == 10
    assert all(0 <= float(r["prevision"]) <= 1 for r in beliefs)


def test_a_probe_run_refuses_inputs_that_are_not_an_extraction(repo, model):
    child = tmp = repo.parent / "noop.py"
    child.write_text("print('not an extraction')\n")
    record(repo, "--experiment", "002", "--", sys.executable, str(tmp))
    other = runs_of(repo)[-1]

    result = record(repo, "--experiment", "002", "--", sys.executable, "-m", "probing.probe",
                    "--activations", other["run_id"], "--protocol", "belief-probe-linear@1")

    assert result.returncode != 0
    assert "activations-last-token@1" in result.stderr


def test_elicited_previsions_are_booked_on_any_model_reference(repo, model):
    result = record(repo, "--experiment", "002", "--", sys.executable, "-m", "coherence.measure_families",
                    "--model", f"{model}@local", "--protocol", "belief-elicit-logprob@1")

    assert result.returncode == 0, result.stderr
    run = runs_of(repo)[-1]
    assert run["protocol"] == "belief-elicit-logprob@1"
    assert run["datasets"]["facts_families"]["version"] == 1
    rates = list(csv.DictReader((run_dir(repo, run) / "outputs" / "rates_summary.csv").open()))
    assert {r["template"] for r in rates} == {"t0", "t1", "t2"}
