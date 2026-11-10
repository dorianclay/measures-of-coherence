"""The stated-chance task: with one payoff table per run and the chance stated in each prompt, does a run learn to
switch actions at its break-even chance?

    uv run python experiments/003-umbrella-pilot/analysis/stated_chance_pilot.py <training-run-id> ... [--first 15] [--last 25]

Per training run, over its first and last steps: the share of optimal choices against the best a fixed answer could
get on the same prompts, the protective-action rate, the valid rate, the switch point (the chance above which "protect
whenever the chance is higher" agrees best with the run's answers; NaN for runs that agree with it no better than with
a fixed answer), and the first step at which 99% of a step's answers are the same. The figure plots each run's protective-action rate against the stated chance, in its first and
last steps, with its break-even chance marked. Writes figures/stated_chance_pilot.csv, figures/stated_chance_curves.csv
and figures/stated_chance_pilot.png, each with a sidecar.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis import runs, umbrella as umbrella_analysis
from coherence import umbrella
from coherence.paths import experiments_dir, store_root

BINS = [i / 10 for i in range(11)]


def switch_point(d: pd.DataFrame) -> float:
    """The threshold t for which "protect if and only if the chance is above t" agrees with most valid answers."""
    d = d[d["valid"]]
    protect = d["answer"] == "umbrella"
    candidates = sorted(d["p"].unique())
    agreement = {t: ((d["p"] > t) == protect).mean() for t in [0.0] + candidates}
    t = max(agreement, key=agreement.get)
    fixed = max(protect.mean(), 1 - protect.mean())
    return t if agreement[t] > fixed + 0.05 else float("nan")


def summary(d: pd.DataFrame, samples: dict) -> dict:
    protect = sum(umbrella.optimal_action(samples[i]) == "umbrella" for i in d["index"].unique()) / d["index"].nunique()
    return {"optimal": d["optimal"].mean(), "best_fixed_answer": max(protect, 1 - protect),
            "protective_rate": (d["answer"] == "umbrella").mean(), "valid": d["valid"].mean(),
            "switch_point": switch_point(d)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--first", type=int, default=15, help="Steps at the start to summarize choices over.")
    parser.add_argument("--last", type=int, default=25, help="Steps at the end to summarize choices over.")
    args = parser.parse_args()

    trainings = runs.load(args.run_ids)
    rows, curves = [], []
    for training in trainings:
        config = umbrella_analysis.config(training)
        samples = {s["index"]: s for s in umbrella.read_dataset(store_root() / training["run_id"] / "dataset.jsonl")}
        utilities = samples[0]["utilities"]
        d = umbrella_analysis.decisions(training)
        d["chance_bin"] = pd.cut(d["p"], BINS).apply(lambda b: b.mid).astype(float)
        base = {"run_id": training["run_id"], "seed": config["seed"], "arm": config["arm"],
                "framing": samples[0]["framing"], "break_even": umbrella.break_even(utilities)}
        for phase, part in [("first", d[d["step"] <= args.first]), ("last", d[d["step"] > d["step"].max() - args.last])]:
            rows.append({**base, "phase": phase, **summary(part, samples), "locked_by": umbrella_analysis.locked_by(d),
                         "gpu_hours": runs.gpu_hours(training)})
            curve = part.groupby("chance_bin")["answer"].agg(lambda a: (a == "umbrella").mean())
            curves.append(pd.DataFrame({**base, "phase": phase, "chance": curve.index, "protective_rate": curve.values}))
    table = pd.DataFrame(rows).sort_values(["seed", "phase"])
    curves = pd.concat(curves, ignore_index=True)

    fig, axes = plt.subplots(1, len(trainings), figsize=(4 * len(trainings), 3.8), squeeze=False)
    for ax, (run_id, c) in zip(axes[0], curves.groupby("run_id", sort=False)):
        for phase, style in [("first", ":"), ("last", "-")]:
            p = c[c["phase"] == phase]
            ax.plot(p["chance"], p["protective_rate"], style, marker="o", label=f"{phase} steps")
        ax.axvline(c["break_even"].iloc[0], color="grey", lw=0.8, ls="--", label="break-even")
        ax.set(title=f"seed {c['seed'].iloc[0]} ({c['framing'].iloc[0]})", xlabel="stated chance",
               ylim=(-0.02, 1.02), xlim=(0, 1))
    axes[0][0].set(ylabel="protective-action rate")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()

    figures = experiments_dir() / "003-umbrella-pilot" / "figures"
    runs.save(table, figures / "stated_chance_pilot.csv", trainings)
    runs.save(curves, figures / "stated_chance_curves.csv", trainings)
    runs.save(fig, figures / "stated_chance_pilot.png", trainings)
    print(table.drop(columns="run_id").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
