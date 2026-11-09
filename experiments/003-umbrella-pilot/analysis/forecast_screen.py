"""Can a town-belief protocol read a belief at all? Compares protocols on an untrained model.

    uv run python experiments/003-umbrella-pilot/analysis/forecast_screen.py <measurement-run-id> ...

Per model, protocol and whether the chance of rain was told in the prompt: P(rain), P(no rain) and their sum (1 if
coherent); with told chances, the slopes of both on the told chance (1 and -1 if it reads what it was told), and how
the chance was worded. Works on the framed events too, where "rain" is the event, and on checkpoints, where the model
is the training run and step its checkpoint. Writes figures/forecast_screen.csv, or --name.
"""

import argparse

import numpy as np
import pandas as pd

from analysis import runs
from coherence.paths import experiments_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--name", default="forecast_screen", help="File name, without extension, under figures/.")
    args = parser.parse_args()

    loaded = runs.load(args.run_ids, allow_mixed_protocols=True)
    rows = []
    for run in loaded:
        beliefs = pd.read_csv(run["dir"] / "outputs" / "beliefs.csv")
        told = "told_chance" in beliefs
        rain, dry = [beliefs[beliefs.rain == k].prevision for k in (1, 0)]
        model, _, revision = run["model_ref"].partition("@")
        row = {"model": model, "step": int(revision[4:]) if revision.startswith("step") else None,
               "protocol": run["protocol"], "run_id": run["run_id"], "told": told,
               "told_as": (beliefs["told_as"].iloc[0] if "told_as" in beliefs else "prefix") if told else None,
               "p_rain": rain.mean(), "p_no_rain": dry.mean(), "sum": rain.mean() + dry.mean(),
               "unparsed": int(beliefs.prevision.isna().sum()), "answers": len(beliefs)}
        if told:
            for k, name in [(1, "slope_rain"), (0, "slope_no_rain")]:
                part = beliefs[(beliefs.rain == k) & beliefs.prevision.notna()]
                row[name] = np.polyfit(part.told_chance, part.prevision, 1)[0]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["model", "step", "protocol", "told", "told_as"], na_position="first")
    runs.save(table, experiments_dir() / "003-umbrella-pilot" / "figures" / f"{args.name}.csv", loaded)
    print(table.drop(columns="run_id").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
