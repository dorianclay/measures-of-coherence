"""The GRPO training command for experiment 002, run under `record` with a tiny model on CPU."""

import json
import sys

import pytest

import tiny_model
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
