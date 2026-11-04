"""The umbrella/rain decision task of experiment 002."""

import random

import pytest

from coherence import umbrella


def test_each_seed_assigns_the_rain_chances_to_the_towns_and_keeps_the_assignment():
    first = umbrella.assign_towns(seed=0)

    assert sorted(first.values()) == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert set(first) == set(umbrella.TOWNS)
    assert umbrella.assign_towns(seed=0) == first
    assert any(umbrella.assign_towns(seed=s) != first for s in range(1, 5))


def test_generated_samples_hide_the_rain_chance_and_keep_the_choice_non_trivial():
    samples = umbrella.generate(seed=3, n=500)

    assert [s["index"] for s in samples] == list(range(500))
    assignment = umbrella.assign_towns(seed=3)
    for s in samples:
        u = s["utilities"]
        assert s["p"] == assignment[s["town"]]
        assert all(-10 <= v <= 10 and isinstance(v, int) for v in u.values())
        assert u["umbrella_rain"] > u["no_umbrella_rain"] and u["no_umbrella_dry"] > u["umbrella_dry"]
        prompt = s["prompt"][-1]["content"]
        assert s["town"] in prompt and str(s["p"]) not in prompt
        for v in u.values():
            assert str(v) in prompt
    assert umbrella.generate(seed=3, n=500) == samples


def test_the_drawn_weather_follows_each_towns_hidden_chance():
    samples = umbrella.generate(seed=1, n=20000)

    for town, p in umbrella.assign_towns(seed=1).items():
        rain = [s["rain"] for s in samples if s["town"] == town]
        assert abs(sum(rain) / len(rain) - p) < 0.03


@pytest.mark.parametrize("text, answer", [
    ("umbrella", "umbrella"), ("Umbrella.", "umbrella"), (" no umbrella", "no umbrella"),
    ("No umbrella", "no umbrella"), ("maybe", None), ("umbrella or no umbrella", None),
])
def test_answers_are_read_strictly(text, answer):
    assert umbrella.parse_answer(text) == answer


def test_the_reward_is_the_utility_of_the_answer_in_the_drawn_weather():
    sample = {"utilities": {"umbrella_rain": 5, "umbrella_dry": -2, "no_umbrella_rain": -8, "no_umbrella_dry": 3},
              "rain": True}

    assert umbrella.reward(sample, "umbrella") == 5
    assert umbrella.reward(sample, "no umbrella") == -8
    assert umbrella.reward({**sample, "rain": False}, "no umbrella") == 3
    assert umbrella.reward(sample, None) == umbrella.INVALID_REWARD < -10


def test_the_control_reward_ignores_the_answer_but_comes_from_the_same_utilities():
    sample = {"utilities": {"umbrella_rain": 5, "umbrella_dry": -2, "no_umbrella_rain": -8, "no_umbrella_dry": 3},
              "rain": True}
    rng = random.Random(0)

    draws = [umbrella.control_reward(sample, rng) for _ in range(2000)]

    assert set(draws) == {5, -8}
    assert abs(draws.count(5) / len(draws) - 0.5) < 0.05


def test_the_optimal_action_maximizes_expected_utility_under_the_hidden_chance():
    u = {"umbrella_rain": 4, "umbrella_dry": 0, "no_umbrella_rain": -6, "no_umbrella_dry": 2}
    # EU(umbrella) = 4p, EU(no umbrella) = 2 - 8p: umbrella iff p > 1/6

    assert umbrella.optimal_action({"utilities": u, "p": 0.1}) == "no umbrella"
    assert umbrella.optimal_action({"utilities": u, "p": 0.3}) == "umbrella"


def test_a_dataset_written_to_disk_reads_back_unchanged(tmp_path):
    samples = umbrella.generate(seed=2, n=50)

    umbrella.write_dataset(samples, tmp_path / "dataset.jsonl")

    assert umbrella.read_dataset(tmp_path / "dataset.jsonl") == samples
