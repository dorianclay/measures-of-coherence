"""Learning curves of pilot training runs: how often each run chose validly and optimally, and how its umbrella rate
came to depend on each town's hidden chance of rain.

    uv run python experiments/003-umbrella-pilot/analysis/learning_curves.py <run-id> ... [--window 25]

Writes figures/learning_curves.png and figures/learning_curves.csv (per run and step window), each with a sidecar.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis import runs, umbrella
from coherence.paths import experiments_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--window", type=int, default=25, help="Steps per point on the curves.")
    args = parser.parse_args()

    loaded = runs.load(args.run_ids)
    frames = []
    for run in loaded:
        d = umbrella.decisions(run)
        lr = umbrella.config(run)["learning_rate"]
        print(f"{run['run_id']}: {runs.gpu_hours(run):.2f} GPU-hours over {len(run['segments'])} segment(s)")
        frames.append(d.assign(label=f"{run['run_id']} {d['arm'].iloc[0]} lr={lr}",
                               window=(d["step"] - 1) // args.window * args.window + args.window,
                               took_umbrella=d["answer"] == "umbrella"))
    curves = (pd.concat(frames)
              .groupby(["label", "window"])
              .agg(valid=("valid", "mean"), optimal=("optimal", "mean"), reward=("reward", "mean"))
              .reset_index())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for label, c in curves.groupby("label"):
        axes[0].plot(c["window"], c["valid"], label=label)
        axes[1].plot(c["window"], c["optimal"], label=label)
    axes[0].set(title="Valid answers", xlabel="step", ylim=(0, 1.02))
    axes[1].set(title="Expected-utility-optimal choices", xlabel="step", ylim=(0, 1.02))
    axes[1].axhline(0.5, color="grey", lw=0.8, ls=":")
    for frame in frames:
        last = frame[frame["window"] == frame["window"].max()]
        rate = last.groupby("p")["took_umbrella"].mean()
        axes[2].plot(rate.index, rate.values, marker="o", label=frame["label"].iloc[0])
    axes[2].set(title="Umbrella rate in the last window, by hidden chance of rain", xlabel="p(rain)", ylim=(0, 1.02))
    axes[0].legend(fontsize=7)
    fig.tight_layout()

    figures = experiments_dir() / "003-umbrella-pilot" / "figures"
    runs.save(fig, figures / "learning_curves.png", loaded)
    runs.save(curves, figures / "learning_curves.csv", loaded)
    print(curves.groupby("label").tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
