"""
Rate of Loss to a Dutch Book
Date: 2026-09-29

Measures the incoherence of a bookie's previsions over an event family as the rate of loss to a Dutch book: the largest
guaranteed profit per unit of stake that a gambler can force by buying or selling $1 tickets on the family's events at the
bookie's prices. It is zero exactly when the previsions are coherent.

Two normalizations of the stakes are supported:
- "L":   total stake sum |b_i| <= 1 (Andrews 2026), giving L in [0, 1]. The primary rate.
- "rho": the bookie's escrow, the most the bookie can lose on the bets, <= 1 (Schervish, Seidenfeld & Kadane 1998).

Run as a script, it books every event family in the given family files (built by Build_Event_Families.py) against a
table of previsions and saves one rate per family and normalization. Previsions are a CSV with columns family_id,
event_index and prevision; any other columns (model, layer, prevision source, ...) group the previsions into separate
bookies and are carried through to the output. As controls, --previsions labels books the ground-truth labels
(coherent unless the dataset contradicts itself) and --previsions random books independent uniform previsions (the ceiling).
L is computed for every family; rho only for conjunction families, where it can rank families differently from L.

Requirements:
- numpy library
- pandas library
- scipy library
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

PREVISION_KEYS = ["family_id", "event_index"]


@dataclass
class RateOfLoss:
    rate: float
    stakes: np.ndarray  # gambler's stake per event; positive buys a ticket, negative sells one


# Per-unit normalization cost of the gambler buying (b > 0) and selling (b < 0) a ticket priced at p
NORMALIZATION_COSTS = {
    "L": lambda p: (np.ones_like(p), np.ones_like(p)),
    "rho": lambda p: (1 - p, p),
}


def rate_of_loss(atoms, previsions, normalization: str = "L") -> RateOfLoss:
    """
    atoms -- one row per atom, one column per event: 1 if the event is true in that atom.
    previsions -- the bookie's price for a $1 ticket on each event.
    normalization -- "L" or "rho"; see the module docstring.
    """
    atoms = np.asarray(atoms, float)
    p = np.asarray(previsions, float)
    n = len(p)
    payoff = atoms - p  # gambler's net payoff per unit stake, per atom and event
    # Variables [buy (n), sell (n), t]: maximize t, the payoff guaranteed in every atom
    c = np.r_[np.zeros(2 * n), -1.0]
    A_ub = np.vstack([
        np.hstack([-payoff, payoff, np.ones((len(atoms), 1))]),
        np.r_[*NORMALIZATION_COSTS[normalization](p), 0.0],
    ])
    b_ub = np.r_[np.zeros(len(atoms)), 1.0]
    bounds = [(0, None)] * (2 * n) + [(None, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return RateOfLoss(rate=max(0.0, -result.fun), stakes=result.x[:n] - result.x[n:2 * n])


def load_families(paths) -> list:
    families = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        families += data["negation_pairs"] + data["conjunction_families"]
    return families


def label_previsions(families) -> pd.DataFrame:
    return pd.DataFrame([
        {"family_id": fam["id"], "event_index": i, "prevision": float(e["label"])}
        for fam in families for i, e in enumerate(fam["events"])
    ])


def random_previsions(families, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    previsions = label_previsions(families)
    previsions["prevision"] = rng.uniform(0, 1, len(previsions))
    return previsions


def book(families, previsions: pd.DataFrame) -> pd.DataFrame:
    """Rate of loss for every (bookie, family), where a bookie is one combination of the grouping columns."""
    group_columns = [c for c in previsions.columns if c not in PREVISION_KEYS + ["prevision"]]
    by_id = {fam["id"]: fam for fam in families}
    rows = []
    groups = previsions.groupby(group_columns, sort=False) if group_columns else [((), previsions)]
    for group, bookie in groups:
        group = dict(zip(group_columns, group if isinstance(group, tuple) else (group,)))
        for family_id, prev in bookie.groupby("family_id", sort=False):
            fam = by_id[family_id]
            p = prev.set_index("event_index").prevision.reindex(range(len(fam["events"])))
            if p.isna().any():
                raise ValueError(f"{family_id} is missing previsions for events {list(p.index[p.isna()])} ({group})")
            normalizations = ["L", "rho"] if fam["kind"] == "conjunction" else ["L"]
            for normalization in normalizations:
                rows.append({**group, "family_id": family_id, "kind": fam["kind"],
                             "labels_consistent": fam["labels_consistent"], "normalization": normalization,
                             "rate": rate_of_loss(fam["atoms"], p.to_numpy(), normalization).rate})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Book event families against previsions and save the rates of loss.")
    parser.add_argument("--families", nargs="+", required=True, help="Event family JSON files.")
    parser.add_argument("--previsions", required=True,
                        help="CSV with family_id, event_index, prevision (+ grouping columns), or 'labels' or 'random'.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --previsions random.")
    parser.add_argument("--output", required=True, help="CSV to save the rates to.")
    args = parser.parse_args()

    families = load_families(args.families)
    if args.previsions == "labels":
        previsions = label_previsions(families)
    elif args.previsions == "random":
        previsions = random_previsions(families, args.seed)
    else:
        previsions = pd.read_csv(args.previsions)
    rates = book(families, previsions)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rates.to_csv(args.output, index=False)
    print(f"Saved {len(rates)} rates to {args.output}")


if __name__ == "__main__":
    main()
