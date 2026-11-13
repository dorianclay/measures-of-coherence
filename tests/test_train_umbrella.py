"""The GRPO training command for experiment 002, run under `record` with a tiny model on CPU."""

import json
import sys

import pytest

import tiny_model
from coherence import umbrella
from coherence.hashing import sha256
from test_record import make_repo, only_run_dir, record


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return tiny_model.build(tmp_path_factory.mktemp("tiny"))


def train_command(model, *extra, arm="rl"):
    return ["--", sys.executable, "-m", "coherence.train_umbrella", "--model", str(model), "--seed", "0",
            "--arm", arm, "--steps", "4", "--prompts-per-step", "2", "--group-size", "2",
            "--eval-fractions", "0.5", "1.0", "--max-completion-length", "4", *extra]


def read_run(repo):
    return json.loads((only_run_dir(repo) / "run.json").read_text())


def reward_log(repo, run):
    return [json.loads(line) for line in (repo.parent / "store" / run["run_id"] / "rewards.jsonl").read_text().splitlines()]


def test_a_training_run_saves_its_dataset_eval_checkpoints_and_every_reward(tmp_path, model):
    repo = make_repo(tmp_path)

    result = record(repo, "--experiment", "002", *train_command(model))

    assert result.returncode == 0, result.stderr
    run = read_run(repo)
    assert run["status"] == "succeeded"
    assert (run["planned_steps"], run["step"], run["seed"]) == (4, 4, 0)
    assert run["model_ref"].startswith(str(model))
    store = repo.parent / "store" / run["run_id"]
    by_path = {a["path"]: a for a in run["artifacts"]}
    assert by_path[f"{run['run_id']}/dataset.jsonl"]["sha256"] == sha256(store / "dataset.jsonl")
    assert len((store / "dataset.jsonl").read_text().splitlines()) == 4 * 2
    evals = sorted(a["step"] for a in run["artifacts"] if a.get("checkpoint") == "eval")
    assert evals == [0, 2, 4]
    rewards = reward_log(repo, run)
    assert len(rewards) == 4 * 2 * 2
    dataset = [json.loads(line) for line in (store / "dataset.jsonl").read_text().splitlines()]
    for entry in rewards:
        sample = dataset[entry["index"]]
        assert entry["index"] // 2 == entry["step"] - 1  # trains on the file in order, two prompts per step
        assert entry["reward"] == (-11 if entry["answer"] is None else
                                   sample["utilities"][entry["answer"].replace(" ", "_") +
                                                       ("_rain" if sample["rain"] else "_dry")])


def test_training_prompts_have_reasoning_off(tmp_path):
    """A template that opens a reasoning block by default would spend the few completion tokens inside it."""
    template = "{% if enable_thinking is not defined or enable_thinking %}{{ raise_exception('reasoning on') }}{% endif %}"
    model = tiny_model.build(tmp_path / "model", chat_template=template + tiny_model.CHAT_TEMPLATE)
    repo = make_repo(tmp_path)

    result = record(repo, "--experiment", "002", *train_command(model))

    assert result.returncode == 0, result.stderr


def test_a_run_split_across_segments_trains_exactly_like_an_unbroken_one(tmp_path, model):
    unbroken_repo = make_repo(tmp_path / "unbroken")
    split_repo = make_repo(tmp_path / "split")

    # The control arm's rewards vary whatever the tiny model answers, so its weights and optimizer state really change
    record(unbroken_repo, "--experiment", "002", *train_command(model, "--learning-rate", "0.05", arm="control"))
    first = record(split_repo, "--experiment", "002",
                   *train_command(model, "--learning-rate", "0.05", "--stop-after", "2", arm="control"))
    assert first.returncode == 0, first.stderr
    assert read_run(split_repo)["status"] == "running"
    second = record(split_repo, "--resume", read_run(split_repo)["run_id"])

    assert second.returncode == 0, second.stderr
    split = read_run(split_repo)
    assert split["status"] == "succeeded"
    assert [s["steps"] for s in split["segments"]] == [[0, 2], [2, 4]]
    unbroken = reward_log(unbroken_repo, read_run(unbroken_repo))
    assert len({e["reward"] for e in unbroken}) > 1 and len({e["completion"] for e in unbroken}) > 1
    assert reward_log(split_repo, split) == unbroken
    # Sampled tokens barely respond to small weight changes; the saved adapters show any difference in training
    eval_hashes = lambda run: {a["step"]: a["sha256"] for a in run["artifacts"] if a.get("checkpoint") == "eval"}
    assert eval_hashes(split) == eval_hashes(read_run(unbroken_repo))
    assert len(set(eval_hashes(split).values())) == 3


def test_a_framed_run_trains_on_its_conditions_samples_and_rewards_them_in_its_words(tmp_path, model):
    repo = make_repo(tmp_path)

    result = record(repo, "--experiment", "002", *train_command(model, "--framing", "invented"))

    assert result.returncode == 0, result.stderr
    run = read_run(repo)
    store = repo.parent / "store" / run["run_id"]
    dataset = [json.loads(line) for line in (store / "dataset.jsonl").read_text().splitlines()]
    assert {s["framing"] for s in dataset} == {"orb"}
    for entry in reward_log(repo, run):
        assert entry["answer"] in (None, "umbrella", "no umbrella")


def test_a_stated_chance_run_trains_on_one_payoff_table_with_chances_in_its_prompts(tmp_path, model):
    repo = make_repo(tmp_path)

    result = record(repo, "--experiment", "002",
                    *train_command(model, "--framing", "per-run", "--chance", "stated"))

    assert result.returncode == 0, result.stderr
    store = repo.parent / "store" / read_run(repo)["run_id"]
    dataset = [json.loads(line) for line in (store / "dataset.jsonl").read_text().splitlines()]
    assert len({json.dumps(s["utilities"], sort_keys=True) for s in dataset}) == 1
    assert all(f"There is a {round(s['p'] * 100)}% chance" in s["prompt"][-1]["content"] for s in dataset)


def test_a_stated_chance_run_takes_its_break_even_range(tmp_path, model):
    repo = make_repo(tmp_path)

    result = record(repo, "--experiment", "002", *train_command(model, "--framing", "per-run", "--chance", "stated",
                                                                "--break-even", "0.25", "0.35"))

    assert result.returncode == 0, result.stderr
    store = repo.parent / "store" / read_run(repo)["run_id"]
    utilities = json.loads((store / "dataset.jsonl").read_text().splitlines()[0])["utilities"]
    assert 0.25 <= umbrella.break_even(utilities) <= 0.35


def test_a_per_story_run_trains_on_mixed_stories_each_with_its_own_chance(tmp_path, model):
    repo = make_repo(tmp_path)

    result = record(repo, "--experiment", "002", *train_command(model, "--framing", "mixed", "--chance", "per-story"))

    assert result.returncode == 0, result.stderr
    store = repo.parent / "store" / read_run(repo)["run_id"]
    dataset = [json.loads(line) for line in (store / "dataset.jsonl").read_text().splitlines()]
    assert all(s["p"] == umbrella.story_chances(0)[s["framing"]] for s in dataset)
