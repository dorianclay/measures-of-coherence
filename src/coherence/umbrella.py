"""The umbrella/rain decision task of experiment 002.

Each sample names one of five fictional towns and states the four utilities U(action, weather); the model answers
"umbrella" or "no umbrella". Each town has a hidden chance of rain, assigned to the names afresh per seed, which the
prompt never mentions: the model can learn it only from its rewards. The weather is drawn with the town's chance when
the sample is generated, so a run's whole dataset, rewards included, is fixed by its seed and can be checked later.

The framed variant (003's third round) drops the towns: each run has one hidden chance, drawn uniformly per seed, and
tells the decision as invented scenarios with no real-world odds to remember. Its samples keep the same fields, with
"umbrella"/"no umbrella" naming the protective action and its alternative, and "rain" whether the event happened,
whatever the framing calls them; `answers` maps the framing's words to those names.

The stated-chance variant (003's fourth round) turns that around: each run keeps one payoff table, drawn per seed with
its break-even chance between 0.25 and 0.75 (or in a range the run gives), and each prompt draws its own chance,
uniform on (0, 1) in whole percent, and states it.

The per-story variant (003's fifth round) hides the chance again, with a different one for each story in a run: the
seed shuffles STORY_CHANCES over the four stories, and payoffs are drawn afresh per prompt, so doing well needs both
the story's chance and the prompt's payoffs.
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


FRAMINGS = {
    "frost": {"setting": "You grow saltberry vines in the valley of Orsk. Tonight a glimmerfrost may settle on them.",
              "act": "cover", "skip": "leave", "act_phrase": "cover the vines", "skip_phrase": "leave them open",
              "event": "a glimmerfrost settles", "none": "no glimmerfrost settles",
              "statement": "A glimmerfrost will settle in Orsk tonight.",
              "negation": "No glimmerfrost will settle in Orsk tonight."},
    "barge": {"setting": "You run a barge across the Vell channel. Today a silt drift may block the channel.",
              "act": "hire", "skip": "sail", "act_phrase": "hire a dredger", "skip_phrase": "sail without one",
              "event": "a silt drift blocks the channel", "none": "the channel stays clear",
              "statement": "A silt drift will block the Vell channel today.",
              "negation": "No silt drift will block the Vell channel today."},
    "beacon": {"setting": "You keep the observatory on Mount Teshel. Tonight the tessel beacon may flare.",
               "act": "shutter", "skip": "open", "act_phrase": "shutter the lenses", "skip_phrase": "keep them open",
               "event": "the beacon flares", "none": "the beacon stays dark",
               "statement": "The tessel beacon will flare tonight.",
               "negation": "The tessel beacon will not flare tonight."},
    "market": {"setting": "You keep a stall in the market of Prell. Today a swarm of pellicks may pass over it.",
               "act": "close", "skip": "trade", "act_phrase": "close the stall", "skip_phrase": "keep trading",
               "event": "the pellicks pass over", "none": "the pellicks stay away",
               "statement": "A swarm of pellicks will pass over the market of Prell today.",
               "negation": "No swarm of pellicks will pass over the market of Prell today."},
    "orb": {"setting": "Each day the Varn orb in your workshop either hums or stays silent.",
            "act": "wear", "skip": "skip", "act_phrase": "wear the dampener", "skip_phrase": "go without it",
            "event": "the orb hums", "none": "the orb stays silent",
            "statement": "The Varn orb will hum today.", "negation": "The Varn orb will not hum today."},
}
STORIES = ["frost", "barge", "beacon", "market"]
INVENTED = "orb"
FRAMING_CONDITIONS = ["mixed", "invented", "per-run"]
CHANCES = ["hidden", "stated", "per-story"]
STORY_CHANCES = [0.2, 0.4, 0.6, 0.8]
BREAK_EVEN_RANGE = (0.25, 0.75)

FRAMED_PROMPT = """{setting} You decide whether to {act_phrase}.

Your payoffs:
- {Act_phrase} and {event}: {umbrella_rain}
- {Act_phrase} and {none}: {umbrella_dry}
- {Skip_phrase} and {event}: {no_umbrella_rain}
- {Skip_phrase} and {none}: {no_umbrella_dry}

Answer with exactly "{act}" or "{skip}"."""


def hidden_chance(seed: int) -> float:
    """The run's one hidden chance of the event, uniform on (0, 1); the same seed gives it in every condition."""
    return random.Random(f"chance-{seed}").random()


def break_even(utilities: dict[str, int]) -> float:
    """The chance of the event at which both actions have the same expected utility."""
    u = utilities
    gain_if_event, gain_if_none = u["umbrella_rain"] - u["no_umbrella_rain"], u["no_umbrella_dry"] - u["umbrella_dry"]
    return gain_if_none / (gain_if_event + gain_if_none)


def run_utilities(seed: int, break_even_range: tuple[float, float] = BREAK_EVEN_RANGE) -> dict[str, int]:
    """A stated-chance run's one payoff table, with its break-even chance inside break_even_range."""
    rng = random.Random(f"run-utilities-{seed}")
    while True:
        utilities = draw_utilities(rng)
        if break_even_range[0] <= break_even(utilities) <= break_even_range[1]:
            return utilities


def story_chances(seed: int) -> dict[str, float]:
    """A per-story run's hidden chance for each story."""
    chances = STORY_CHANCES[:]
    random.Random(f"story-chances-{seed}").shuffle(chances)
    return dict(zip(STORIES, chances))


def generate_framed(seed: int, n: int, condition: str, chance: str = "hidden",
                    break_even_range: tuple[float, float] = BREAK_EVEN_RANGE) -> list[dict]:
    """A framed run's training samples: "mixed" draws a story per prompt, "invented" always tells the orb, and
    "per-run" tells one story chosen by the seed. With chance "hidden" the run has one chance and fresh payoffs per
    prompt; with "per-story", one chance per story and fresh payoffs per prompt; with "stated", one payoff table and a
    fresh chance per prompt, written in it."""
    stated = chance == "stated"
    p = hidden_chance(seed)
    per_story = story_chances(seed) if chance == "per-story" else None
    rng = random.Random({"hidden": "framed-samples", "stated": "stated-samples", "per-story": "per-story-samples"}[chance]
                        + f"-{seed}")
    samples = []
    for index in range(n):
        framing = {"mixed": lambda: rng.choice(STORIES), "invented": lambda: INVENTED,
                   "per-run": lambda: STORIES[seed % len(STORIES)]}[condition]()
        f = FRAMINGS[framing]
        if stated:
            utilities, p = run_utilities(seed, break_even_range), rng.randint(1, 99) / 100
        else:
            utilities = draw_utilities(rng)
            if per_story:
                p = per_story[framing]
        prompt = FRAMED_PROMPT.format(**f, Act_phrase=f["act_phrase"].capitalize(),
                                      Skip_phrase=f["skip_phrase"].capitalize(), **utilities)
        if stated:
            setting, rest = prompt.split(" You decide", 1)
            prompt = f"{setting} There is a {p:.0%} chance that {f['event']}. You decide{rest}"
        samples.append({"index": index, "framing": framing, "p": p, "utilities": utilities, "rain": rng.random() < p,
                        "answers": {f["act"]: "umbrella", f["skip"]: "no umbrella"},
                        "prompt": [{"role": "user", "content": prompt}]})
    return samples


def write_dataset(samples: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(s) + "\n" for s in samples))


def read_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def parse_answer(text: str, sample: dict | None = None) -> str | None:
    """The action a completion chose, or None unless it is exactly one of the two answers (a framed sample's own)."""
    answer = re.sub(r"[\s.!\"']+", " ", text.lower()).strip()
    answers = sample["answers"] if sample and "answers" in sample else {a: a for a in ACTIONS}
    return answers.get(answer)


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


def event_statements() -> list[dict]:
    """The statements whose previsions are the model's beliefs about each framing's event (the dataset framed_events).
    `rain` says whether a statement asserts the event (1) or its absence (0); `event` is the event's own statement."""
    return [{"framing": name, "rain": rain, "statement": f["statement"] if rain else f["negation"],
             "event": f["statement"]} for name, f in FRAMINGS.items() for rain in (1, 0)]
