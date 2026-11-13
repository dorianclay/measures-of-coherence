"""The per-story task: each story in a run has its own hidden chance and payoffs vary per prompt. Do a run's choices,
and its stated beliefs, follow each story's chance?

    uv run python experiments/003-umbrella-pilot/analysis/per_story_pilot.py <training-run-id> ... [--first 25] [--last 50]

Per training run and story, over its first and last steps: the share of optimal choices against the best a fixed
answer per story could get on the same prompts, the protective-action rate, and the implied chance: the chance t for
which "protect whenever the prompt's break-even chance is below t" agrees best with the run's valid answers (NaN unless
it agrees better than a fixed answer, by more than 0.05). That is the chance the choices act on, read from the payoffs.
Also the first step from which 99% of a story's answers in each step are the same one.
Per measured checkpoint, the stated P(event) and P(no event) for each story, worded as training words it (the setting
first, no chance told). Writes figures/per_story_pilot.csv, figures/per_story_curves.csv, figures/per_story_beliefs.csv
and figures/per_story_pilot.png, each with a sidecar.
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
WINDOW = 25


def implied_chance(d: pd.DataFrame) -> float:
    """The threshold t for which "protect if and only if the break-even chance is below t" agrees with most valid
    answers."""
    d = d[d["valid"]]
    protect = d["answer"] == "umbrella"
    agreement = {t: ((d["break_even"] < t) == protect).mean() for t in [0.0, *sorted(d["break_even"].unique()), 1.0]}
    t = max(agreement, key=agreement.get)
    fixed = max(protect.mean(), 1 - protect.mean())
    return t if agreement[t] > fixed + 0.05 else float("nan")


def best_fixed(d: pd.DataFrame, samples: dict) -> float:
    prompts = [samples[i] for i in d["index"].unique()]
    protect = sum(umbrella.optimal_action(s) == "umbrella" for s in prompts) / len(prompts)
    return max(protect, 1 - protect)


def measurements(training: dict) -> list[dict]:
    found = []
    for path in sorted(training["dir"].parent.glob("*/run.json")):
        run = runs.load([path.parent.name])[0]
        if (f" on {training['run_id']}@step" in run["note"] and run.get("protocol") == PROTOCOL
                and "--told-as scenario" in run["note"] and "--told-chances" not in run["note"]):
            if run["status"] == "succeeded":
                found.append(run)
            else:
                print(f"{run['run_id']}: {run['status']}, skipped ({run['note']})")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--first", type=int, default=25, help="Steps at the start to summarize choices over.")
    parser.add_argument("--last", type=int, default=50, help="Steps at the end to summarize choices over.")
    args = parser.parse_args()

    trainings = runs.load(args.run_ids)
    loaded, rows, curves, belief_frames = list(trainings), [], [], []
    for training in trainings:
        config = umbrella_analysis.config(training)
        chances = umbrella.story_chances(config["seed"])
        samples = {s["index"]: s for s in umbrella.read_dataset(store_root() / training["run_id"] / "dataset.jsonl")}
        d = umbrella_analysis.decisions(training)
        d["break_even"] = d["index"].map(lambda i: umbrella.break_even(samples[i]["utilities"]))
        d["window"] = (d["step"] - 1) // WINDOW * WINDOW + WINDOW
        base = {"run_id": training["run_id"], "seed": config["seed"]}
        for phase, part in [("first", d[d["step"] <= args.first]), ("last", d[d["step"] > d["step"].max() - args.last])]:
            for story, s in part.groupby("framing"):
                rows.append({**base, "phase": phase, "story": story, "p": chances[story], "optimal": s["optimal"].mean(),
                             "best_fixed_answer": best_fixed(s, samples),
                             "protective_rate": (s["answer"] == "umbrella").mean(), "valid": s["valid"].mean(),
                             "implied_chance": implied_chance(s),
                             "locked_by": umbrella_analysis.locked_by(d[d["framing"] == story]),
                             "gpu_hours": runs.gpu_hours(training)})
        for (window, story), s in d.groupby(["window", "framing"]):
            curves.append({**base, "step": window, "story": story, "p": chances[story], "optimal": s["optimal"].mean(),
                           "best_fixed_answer": best_fixed(s, samples), "implied_chance": implied_chance(s)})
        for run in measurements(training):
            loaded.append(run)
            beliefs = pd.read_csv(run["dir"] / "outputs" / "beliefs.csv")
            beliefs = beliefs.groupby(["framing", "rain"])["prevision"].mean().unstack("rain")
            belief_frames.append(pd.DataFrame({
                **base, "step": int(run["model_ref"].rsplit("@step", 1)[1]), "story": beliefs.index,
                "p": [chances.get(f, float("nan")) for f in beliefs.index],
                "p_event": beliefs[1].values, "p_no_event": beliefs[0].values}))
    table = pd.DataFrame(rows).sort_values(["seed", "phase", "p"])
    curves = pd.DataFrame(curves)
    beliefs = pd.concat(belief_frames, ignore_index=True) if belief_frames else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    last = table[table["phase"] == "last"]
    axes[0].scatter(last["p"], last["implied_chance"])
    axes[0].set(title=f"Implied chance, last {args.last} steps", xlabel="hidden chance", xlim=(0, 1), ylim=(0, 1))
    axes[0].plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    whole = curves.assign(gain=curves["optimal"] - curves["best_fixed_answer"]).groupby(["seed", "step"])["gain"].mean()
    for seed, g in whole.groupby(level="seed"):
        axes[1].plot(g.index.get_level_values("step"), g.values, label=f"seed {seed}")
    axes[1].axhline(0, color="grey", lw=0.8, ls=":")
    axes[1].set(title=f"Optimal rate minus best fixed answer per story, {WINDOW}-step windows", xlabel="step")
    axes[1].legend(fontsize=8)
    if len(beliefs):
        for at, marker in [("untrained", "x"), ("last", "o")]:
            steps = beliefs.groupby("seed")["step"].transform("min" if at == "untrained" else "max")
            b = beliefs[beliefs["step"] == steps].dropna(subset=["p"])
            axes[2].scatter(b["p"], b["p_event"], marker=marker, label=f"{at} checkpoint")
        axes[2].legend(fontsize=8)
    axes[2].set(title="Stated P(event), trained stories", xlabel="hidden chance", xlim=(0, 1), ylim=(-0.02, 1.02))
    axes[2].plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    fig.tight_layout()

    figures = experiments_dir() / "003-umbrella-pilot" / "figures"
    runs.save(table, figures / "per_story_pilot.csv", loaded)
    runs.save(curves, figures / "per_story_curves.csv", loaded)
    runs.save(beliefs, figures / "per_story_beliefs.csv", loaded)
    runs.save(fig, figures / "per_story_pilot.png", loaded)
    print(table.drop(columns="run_id").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
