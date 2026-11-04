"""Decisions made during umbrella training runs (experiments 002 and 003), read back from their saved data.

Each run's $COHERENCE_STORE/<run-id>/dataset.jsonl and rewards.jsonl are joined, so every completion comes with its
town, hidden chance of rain, whether it was a valid answer, and whether it was the expected-utility-optimal choice.
"""

import json

import pandas as pd

from coherence import umbrella
from coherence.paths import store_root


def config(run: dict) -> dict:
    """The training command's arguments (arm, seed, learning rate, ...), from the run's config.json."""
    return json.loads((run["dir"] / "config.json").read_text())


def decisions(run: dict) -> pd.DataFrame:
    store = store_root() / run["run_id"]
    samples = {s["index"]: s for s in umbrella.read_dataset(store / "dataset.jsonl")}
    rows = []
    for line in (store / "rewards.jsonl").read_text().splitlines():
        entry = json.loads(line)
        sample = samples[entry["index"]]
        rows.append({"step": entry["step"], "index": entry["index"], "town": sample["town"], "p": sample["p"],
                     "answer": entry["answer"], "valid": entry["answer"] is not None,
                     "optimal": entry["answer"] == umbrella.optimal_action(sample), "reward": entry["reward"]})
    return pd.DataFrame(rows).assign(run_id=run["run_id"], **{k: config(run)[k] for k in ["arm", "seed"]})
