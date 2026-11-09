"""Beliefs about each town's rain over training: do they move with its hidden chance of rain?

    uv run python experiments/003-umbrella-pilot/analysis/town_beliefs.py <training-run-id> ...

Takes training runs and finds their measurement runs in the same experiment. Per protocol, plots P(rain) by town and
checkpoint, and the slope of P(rain) on the hidden chance across the five towns over steps (0: no belief formed;
1: calibrated). Writes figures/town_beliefs.png and figures/town_beliefs.csv, each with a sidecar; --name changes the
file names. Protocols with no measurements are left out of the figure.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import runs, umbrella as umbrella_analysis
from coherence import umbrella
from coherence.paths import experiments_dir

PROTOCOLS = ["town-belief-elicit-logprob@1", "town-belief-elicit-stated@1"]


def measurements(training: dict) -> list[dict]:
    """The training run's measurement runs that produced beliefs; failed ones have none, and are printed."""
    found = []
    for path in sorted(training["dir"].parent.glob("*/run.json")):
        run = runs.load([path.parent.name])[0]
        if f" on {training['run_id']}@step" not in run["note"]:
            continue
        if run["status"] == "succeeded":
            found.append(run)
        else:
            print(f"{run['run_id']}: {run['status']}, skipped ({run['note']})")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--name", default="town_beliefs", help="File name, without extension, under figures/.")
    args = parser.parse_args()

    trainings = runs.load(args.run_ids)
    loaded, frames = list(trainings), []
    for training in trainings:
        config = umbrella_analysis.config(training)
        chances = umbrella.assign_towns(config["seed"])
        for run in measurements(training):
            loaded.append(run)
            beliefs = pd.read_csv(run["dir"] / "outputs" / "beliefs.csv")
            frames.append(beliefs.assign(
                training=f"{training['run_id']} {config['arm']} lr={config['learning_rate']} "
                         f"t={config['temperature']} kl={config['kl']}",
                protocol=run["protocol"], step=int(run["model_ref"].rsplit("@step", 1)[1]),
                p=beliefs["town"].map(chances)))
    beliefs = pd.concat(frames)
    # Average over templates; a stated answer that didn't parse is left out of its town's mean
    towns = (beliefs.groupby(["training", "protocol", "step", "town", "p", "rain"])["prevision"].mean()
             .unstack("rain").rename(columns={0: "p_no_rain", 1: "p_rain"}).reset_index())
    towns["slope"] = (towns.groupby(["training", "protocol", "step"], group_keys=False)
                      .apply(lambda g: pd.Series(np.polyfit(g["p"], g["p_rain"], 1)[0], index=g.index),
                             include_groups=False))

    measured = [p for p in PROTOCOLS if p in set(towns["protocol"])]
    fig, axes = plt.subplots(len(measured), 2, figsize=(12, 3.5 * len(measured)), squeeze=False)
    for row, protocol in zip(axes, measured):
        rows = towns[towns["protocol"] == protocol]
        for label, group in rows.groupby("training"):
            slopes = group.groupby("step")["slope"].first()
            row[0].plot(range(len(slopes)), slopes.values, marker="o", label=label)
            row[0].set_xticks(range(len(slopes)), slopes.index)
            last = group[group["step"] == group["step"].max()].sort_values("p")
            row[1].plot(last["p"], last["p_rain"], marker="o", label=label)
        row[0].axhline(0, color="grey", lw=0.8, ls=":")
        row[0].set(title=f"{protocol}: slope of P(rain) on hidden chance", xlabel="step")
        row[1].set(title="P(rain) by town at the last checkpoint", xlabel="hidden chance of rain", ylim=(0, 1.02))
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()

    figures = experiments_dir() / "003-umbrella-pilot" / "figures"
    runs.save(fig, figures / f"{args.name}.png", loaded)
    runs.save(towns, figures / f"{args.name}.csv", loaded)
    print(towns.groupby(["protocol", "training", "step"])["slope"].first().unstack("step").round(3).to_string())


if __name__ == "__main__":
    main()
