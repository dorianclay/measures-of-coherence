"""Measuring a model's beliefs about each town's weather, one protocol per run, on a base model or a checkpoint."""

import csv
import json
import shutil
import sys

import pandas as pd
import pytest
from transformers import AutoTokenizer

import tiny_model
from coherence import protocols
from coherence.elicitation import SYSTEM
from coherence.measure_towns import prompt
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


def test_the_forecast_protocols_ask_without_a_system_prompt_of_ours(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    row = pd.DataFrame([{"town": "Velmora", "rain": 1, "statement": "It will rain in Velmora tomorrow."}])
    for protocol_id, template in [("town-forecast-elicit-logprob@1", "f0"), ("town-forecast-elicit-stated@1", "s0")]:
        text = prompt(protocols.get(protocol_id), tokenizer, template, next(row.itertuples()))
        assert "It will rain in Velmora tomorrow." in text and "Forecast:" in text
        assert SYSTEM not in text and "<|system|>" not in text


def test_a_told_chance_is_stated_in_the_prompt_for_every_town_and_chance(repo, model):
    result = record(repo, "--experiment", "002", "--", sys.executable, "-m", "coherence.measure_towns",
                    "--model", f"{model}@local", "--protocol", "town-forecast-elicit-stated@1",
                    "--told-chances", "0.1", "0.9")

    assert result.returncode == 0, result.stderr
    rows = beliefs(repo, runs_by_command(repo)["coherence.measure_towns"])
    assert len(rows) == 5 * 2 * 2
    assert {float(r["told_chance"]) for r in rows} == {0.1, 0.9}

    told = pd.DataFrame([{"town": "Velmora", "statement": "S.", "told_chance": 0.9}])
    text = prompt(protocols.get("town-forecast-elicit-stated@1"), AutoTokenizer.from_pretrained(model), "s0",
                  next(told.itertuples()))
    assert "In Velmora, it rains on 90% of days." in text


def test_a_protocol_this_command_does_not_implement_is_refused(repo, model):
    result = measure(repo, f"{model}@local", "belief-probe-linear@1")

    assert result.returncode != 0
    assert "town-belief" in result.stderr


def test_a_framed_events_belief_is_measured_per_framing_with_a_told_chance(tmp_path, model):
    repo = make_repo(tmp_path)
    (repo / "data" / "datasets").mkdir(parents=True)
    shutil.copy(repo_root() / "data" / "datasets" / "framed_events.csv", repo / "data" / "datasets")
    manifest = json.loads((repo_root() / "data" / "datasets.json").read_text())
    (repo / "data" / "datasets.json").write_text(json.dumps({"framed_events": manifest["framed_events"]}))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "add framed_events")

    result = record(repo, "--experiment", "002", "--", sys.executable, "-m", "coherence.measure_towns",
                    "--model", f"{model}@local", "--protocol", "event-belief-elicit-stated@1", "--told-chances", "0.3")

    assert result.returncode == 0, result.stderr
    rows = beliefs(repo, runs_by_command(repo)["coherence.measure_towns"])
    assert {(r["framing"], r["rain"]) for r in rows} == {(f, str(k)) for f in tiny_model.umbrella.FRAMINGS for k in (0, 1)}
    row = pd.DataFrame([{"event": "The Varn orb will hum today.", "statement": "S.", "told_chance": 0.3, "rain": 1}])
    text = prompt(protocols.get("event-belief-elicit-stated@1"), AutoTokenizer.from_pretrained(model), "t0",
                  next(row.itertuples()))
    assert "The chance of the following is 30%: The Varn orb will hum today." in text


def test_a_told_chance_can_be_told_as_the_training_prompts_tell_it(model):
    row = pd.DataFrame([{"framing": "frost", "event": "A glimmerfrost will settle in Orsk tonight.", "statement": "S.",
                         "told_chance": 0.7, "rain": 1}])
    text = prompt(protocols.get("event-belief-elicit-stated@1"), AutoTokenizer.from_pretrained(model), "t0",
                  next(row.itertuples()), told_as="scenario")

    setting = tiny_model.umbrella.FRAMINGS["frost"]["setting"]
    assert f"{setting} There is a 70% chance that a glimmerfrost settles." in text
    assert "The chance of the following" not in text


def test_without_a_told_chance_the_scenario_wording_still_opens_with_the_setting(model):
    row = pd.DataFrame([{"framing": "barge", "event": "E.", "statement": "A silt drift will block the Vell channel today.",
                         "rain": 1}])
    text = prompt(protocols.get("event-belief-elicit-stated@1"), AutoTokenizer.from_pretrained(model), "t0",
                  next(row.itertuples()), told_as="scenario")

    setting = tiny_model.umbrella.FRAMINGS["barge"]["setting"]
    assert f"{setting}\n\n" in text and "chance that" not in text
