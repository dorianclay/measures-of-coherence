"""The umbrella/rain decision task of experiment 002.

Each sample names one of five fictional towns and states the four utilities U(action, weather); the model answers
"umbrella" or "no umbrella". Each town has a hidden chance of rain, assigned to the names afresh per seed, which the
prompt never mentions: the model can learn it only from its rewards. The weather is drawn with the town's chance when
the sample is generated, so a run's whole dataset, rewards included, is fixed by its seed and can be checked later.
"""

import json
import random
import re
from pathlib import Path

TOWNS = ["Velmora", "Tarnquist", "Olbereth", "Mirrowdale", "Zandry"]
RAIN_CHANCES = [0.1, 0.3, 0.5, 0.7, 0.9]
ACTIONS = ["umbrella", "no umbrella"]
UTILITY_RANGE = (-10, 10)
# Below every utility, so an unreadable answer is always the worst outcome
INVALID_REWARD = -11

PROMPT = """You live in {town}. Before going out today, you decide whether to take an umbrella.

Your payoffs:
- take an umbrella and it rains: {umbrella_rain}
- take an umbrella and it stays dry: {umbrella_dry}
- no umbrella and it rains: {no_umbrella_rain}
- no umbrella and it stays dry: {no_umbrella_dry}

Answer with exactly "umbrella" or "no umbrella"."""


def assign_towns(seed: int) -> dict[str, float]:
    chances = RAIN_CHANCES[:]
    random.Random(f"towns-{seed}").shuffle(chances)
    return dict(zip(TOWNS, chances))


def draw_utilities(rng: random.Random) -> dict[str, int]:
    """Utilities where each action is best in one weather, so neither is right regardless of the chance of rain."""
    while True:
        u = {k: rng.randint(*UTILITY_RANGE) for k in ["umbrella_rain", "umbrella_dry", "no_umbrella_rain",
                                                      "no_umbrella_dry"]}
        if u["umbrella_rain"] > u["no_umbrella_rain"] and u["no_umbrella_dry"] > u["umbrella_dry"]:
            return u


def generate(seed: int, n: int) -> list[dict]:
    """A run's training samples, in the order it trains on them."""
    towns = assign_towns(seed)
    rng = random.Random(f"samples-{seed}")
    samples = []
    for index in range(n):
        town = rng.choice(TOWNS)
        utilities = draw_utilities(rng)
        samples.append({"index": index, "town": town, "p": towns[town], "utilities": utilities,
                        "rain": rng.random() < towns[town],
                        "prompt": [{"role": "user", "content": PROMPT.format(town=town, **utilities)}]})
    return samples


def write_dataset(samples: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(s) + "\n" for s in samples))


def read_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def parse_answer(text: str) -> str | None:
    """The action a completion chose, or None unless it is exactly one of the two answers."""
    answer = re.sub(r"[\s.!\"']+", " ", text.lower()).strip()
    return answer if answer in ACTIONS else None


def utility(sample: dict, action: str) -> int:
    key = action.replace(" ", "_") + ("_rain" if sample["rain"] else "_dry")
    return sample["utilities"][key]


def reward(sample: dict, answer: str | None) -> int:
    return INVALID_REWARD if answer is None else utility(sample, answer)


def control_reward(sample: dict, rng: random.Random) -> int:
    """The control arm's reward: the utility of a random action in the same weather, whatever the model answered."""
    return utility(sample, rng.choice(ACTIONS))


def optimal_action(sample: dict) -> str:
    """The action with the higher expected utility under the town's hidden chance of rain."""
    p, u = sample["p"], sample["utilities"]
    expected = {"umbrella": p * u["umbrella_rain"] + (1 - p) * u["umbrella_dry"],
                "no umbrella": p * u["no_umbrella_rain"] + (1 - p) * u["no_umbrella_dry"]}
    return max(expected, key=expected.get)


def belief_statements() -> list[dict]:
    """The statements whose previsions are the model's beliefs about each town's weather (the dataset umbrella_towns).
    `rain` says which event a statement asserts, rain (1) or no rain (0), so the pair's previsions should sum to 1."""
    return [row for town in TOWNS for row in [
        {"town": town, "rain": 1, "statement": f"It will rain in {town} tomorrow."},
        {"town": town, "rain": 0, "statement": f"It will not rain in {town} tomorrow."}]]
