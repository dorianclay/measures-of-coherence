"""The framed task: does a run's choice, and its stated belief, follow its one hidden chance, in each framing condition?

    uv run python experiments/003-umbrella-pilot/analysis/framing_pilot.py <training-run-id> ... [--last 25]

Per training run, over its last steps: the share of optimal choices against the best a fixed answer could get on the
same prompts (a run that ignores the payoffs reaches at most that), the protective-action rate, the share of
prompts whose answers all match, and the first step at which 99% of a step's answers are the same. Per measured checkpoint, the stated P(event) and P(no event) for the run's own
framings and for the others. Writes figures/framing_pilot.csv, figures/framing_beliefs.csv and
figures/framing_pilot.png, each with a sidecar.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis import runs, umbrella as umbrella_analysis
from coherence import umbrella
from coherence.paths import experiments_dir, store_root

PROTOCOL = "event-belief-elicit-stated@1"


def measurements(training: dict) -> list[dict]:
    found = []
    for path in sorted(training["dir"].parent.glob("*/run.json")):
        run = runs.load([path.parent.name])[0]
        if f" on {training['run_id']}@step" in run["note"] and run.get("protocol") == PROTOCOL:
            if run["status"] == "succeeded":
                found.append(run)
            else:
                print(f"{run['run_id']}: {run['status']}, skipped ({run['note']})")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--last", type=int, default=25, help="Steps at the end to summarize choices over.")
    args = parser.parse_args()

    trainings = runs.load(args.run_ids)
    loaded, rows, belief_frames = list(trainings), [], []
    for training in trainings:
        config = umbrella_analysis.config(training)
        samples = {s["index"]: s for s in umbrella.read_dataset(store_root() / training["run_id"] / "dataset.jsonl")}
        own = {s["framing"] for s in samples.values()}
        d = umbrella_analysis.decisions(training)
        last = d[d["step"] > d["step"].max() - args.last]
        prompts = [samples[i] for i in last["index"].unique()]
        protect = sum(umbrella.optimal_action(s) == "umbrella" for s in prompts) / len(prompts)
        rows.append({"run_id": training["run_id"], "condition": config["framing"], "seed": config["seed"],
                     "framings": " ".join(sorted(own)), "p": umbrella.hidden_chance(config["seed"]),
                     "optimal": last["optimal"].mean(), "best_fixed_answer": max(protect, 1 - protect),
                     "protective_rate": (last["answer"] == "umbrella").mean(), "valid": last["valid"].mean(),
                     "unanimous": last.groupby("index")["answer"].agg(lambda a: a.nunique(dropna=False) == 1).mean(),
                     "locked_by": umbrella_analysis.locked_by(d), "gpu_hours": runs.gpu_hours(training)})
        for run in measurements(training):
            loaded.append(run)
            beliefs = pd.read_csv(run["dir"] / "outputs" / "beliefs.csv")
            beliefs = beliefs.groupby(["framing", "rain"])["prevision"].mean().unstack("rain")
            belief_frames.append(pd.DataFrame({
                "run_id": training["run_id"], "condition": config["framing"], "p": rows[-1]["p"],
                "step": int(run["model_ref"].rsplit("@step", 1)[1]), "framing": beliefs.index,
                "trained_on": beliefs.index.isin(own), "p_event": beliefs[1].values, "p_no_event": beliefs[0].values}))
    table = pd.DataFrame(rows).sort_values(["condition", "seed"])
    beliefs = pd.concat(belief_frames) if belief_frames else pd.DataFrame()

    conditions = umbrella.FRAMING_CONDITIONS
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for condition in conditions:
        c = table[table["condition"] == condition]
        axes[0].scatter(c["p"], c["protective_rate"], label=condition)
        axes[1].scatter(c["p"], c["optimal"] - c["best_fixed_answer"], label=condition)
        if len(beliefs):
            b = beliefs[(beliefs["condition"] == condition) & beliefs["trained_on"]]
            b = b[b["step"] == b.groupby("run_id")["step"].transform("max")].groupby("run_id")[["p", "p_event"]].mean()
            axes[2].scatter(b["p"], b["p_event"], label=condition)
    axes[0].set(title="Protective-action rate, last steps", xlabel="hidden chance", ylim=(-0.02, 1.02))
    axes[1].set(title="Optimal rate minus best fixed answer", xlabel="hidden chance")
    axes[1].axhline(0, color="grey", lw=0.8, ls=":")
    axes[2].set(title=f"Stated P(event), own framings, last checkpoint", xlabel="hidden chance", ylim=(-0.02, 1.02))
    axes[2].plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    axes[0].legend(fontsize=8)
    fig.tight_layout()

    figures = experiments_dir() / "003-umbrella-pilot" / "figures"
    runs.save(table, figures / "framing_pilot.csv", loaded)
    runs.save(beliefs, figures / "framing_beliefs.csv", loaded)
    runs.save(fig, figures / "framing_pilot.png", loaded)
    print(table.drop(columns="run_id").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
