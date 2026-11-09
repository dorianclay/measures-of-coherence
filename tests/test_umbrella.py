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


# The framed variant: one hidden chance per run, the decision told as one of several invented scenarios


def test_each_seed_draws_one_hidden_chance_shared_by_every_framing_condition():
    chances = [umbrella.hidden_chance(seed) for seed in range(20)]

    assert all(0 < p < 1 for p in chances) and len(set(chances)) == 20
    for condition in umbrella.FRAMING_CONDITIONS:
        assert {s["p"] for s in umbrella.generate_framed(seed=2, n=50, condition=condition)} == {chances[2]}


@pytest.mark.parametrize("condition, framings", [
    ("mixed", set(umbrella.STORIES)), ("invented", {umbrella.INVENTED}), ("per-run", {umbrella.STORIES[1]}),
])
def test_each_condition_tells_the_decision_in_its_own_framings(condition, framings):
    samples = umbrella.generate_framed(seed=1, n=400, condition=condition)

    assert {s["framing"] for s in samples} == framings
    for s in samples:
        f = umbrella.FRAMINGS[s["framing"]]
        prompt = s["prompt"][-1]["content"]
        assert f["act"] in prompt and f["skip"] in prompt and "umbrella" not in prompt and "rain" not in prompt
        assert str(round(s["p"], 2)) not in prompt
        for v in s["utilities"].values():
            assert str(v) in prompt
    assert umbrella.generate_framed(seed=1, n=400, condition=condition) == samples


def test_the_framed_event_follows_the_hidden_chance():
    samples = umbrella.generate_framed(seed=4, n=20000, condition="mixed")

    assert abs(sum(s["rain"] for s in samples) / len(samples) - umbrella.hidden_chance(4)) < 0.02


def test_a_framed_answer_is_read_in_its_framings_words_and_rewarded_as_its_action():
    sample = umbrella.generate_framed(seed=0, n=1, condition="invented")[0]
    f = umbrella.FRAMINGS[umbrella.INVENTED]

    assert umbrella.parse_answer(f["act"].capitalize() + ".", sample) == "umbrella"
    assert umbrella.parse_answer(f["skip"], sample) == "no umbrella"
    assert umbrella.parse_answer("umbrella", sample) is None
    u, weather = sample["utilities"], "_rain" if sample["rain"] else "_dry"
    assert umbrella.reward(sample, umbrella.parse_answer(f["skip"], sample)) == u["no_umbrella" + weather]


def test_every_framing_has_a_statement_pair_to_measure_its_event():
    rows = umbrella.event_statements()

    assert {(r["framing"], r["rain"]) for r in rows} == {(f, k) for f in umbrella.FRAMINGS for k in (0, 1)}


# The stated-chance variant: one payoff table per run, a fresh chance per prompt, stated in the prompt


def break_even(u):
    return (u["no_umbrella_dry"] - u["umbrella_dry"]) / (
        u["umbrella_rain"] - u["no_umbrella_rain"] + u["no_umbrella_dry"] - u["umbrella_dry"])


def test_a_stated_chance_run_keeps_one_payoff_table_with_a_break_even_chance_away_from_the_ends():
    for seed in range(20):
        samples = umbrella.generate_framed(seed=seed, n=50, condition="per-run", chance="stated")
        (table,) = {tuple(sorted(s["utilities"].items())) for s in samples}
        assert 0.25 <= break_even(dict(table)) <= 0.75
    tables = {tuple(sorted(umbrella.generate_framed(seed=s, n=1, condition="per-run", chance="stated")[0]
                           ["utilities"].items())) for s in range(20)}
    assert len(tables) > 10


def test_each_stated_chance_prompt_states_its_own_chance_and_its_event_follows_it():
    samples = umbrella.generate_framed(seed=3, n=20000, condition="mixed", chance="stated")

    for s in samples[:200]:
        f = umbrella.FRAMINGS[s["framing"]]
        assert f"There is a {round(s['p'] * 100)}% chance that {f['event']}." in s["prompt"][-1]["content"]
        assert 0 < s["p"] < 1 and s["p"] == round(s["p"], 2)
    assert len({s["p"] for s in samples}) > 90
    for low in (0.0, 0.5):
        band = [s for s in samples if low < s["p"] <= low + 0.5]
        assert abs(sum(s["rain"] for s in band) / len(band) - sum(s["p"] for s in band) / len(band)) < 0.02
    assert umbrella.generate_framed(seed=3, n=100, condition="mixed", chance="stated") == samples[:100]


def test_a_stated_chance_run_needs_both_answers_to_do_well():
    samples = umbrella.generate_framed(seed=0, n=2000, condition="per-run", chance="stated")
    protective = sum(umbrella.optimal_action(s) == "umbrella" for s in samples) / len(samples)

    assert 0.2 < protective < 0.8


def test_the_hidden_chance_variant_is_unchanged_by_the_stated_one():
    assert umbrella.generate_framed(seed=1, n=30, condition="mixed") == \
        umbrella.generate_framed(seed=1, n=30, condition="mixed", chance="hidden")
