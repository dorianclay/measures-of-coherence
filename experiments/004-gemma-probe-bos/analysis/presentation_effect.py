"""Does presenting statements in a chat turn make probes work? Compares presentations on 004's probe runs.

    uv run python experiments/004-gemma-probe-bos/analysis/presentation_effect.py <probe-run-id> ...

Per model, presentation, probe and booked domain: the layer with the best validation accuracy, its validation and
booked accuracy, and its mean L, raw and calibrated. For a baseline of raw text with BOS, pass 004's bos=add run among
the ids; 001's numbers are in its own results. Writes figures/presentation_effect.csv (or --output), with a sidecar.
"""

import argparse
import re

import pandas as pd

from analysis import runs
from coherence.paths import experiments_dir, store_root

KINDS = ["negation_pair", "conjunction"]
PROBES = ["linear", "mlp"]
BAR = 0.65


def best_layer(metrics: pd.DataFrame, probe: str, domain: str) -> pd.Series:
    """Mean over seeds per layer, so the MLP's five seeds count as one; the layer is chosen on validation only."""
    rows = metrics[(metrics.arm == "domain_swap") & (metrics.probe == probe) & (metrics.booked_domain == domain)]
    by_layer = rows.groupby("layer")[["validation_accuracy", "booked_accuracy"]].mean().reset_index()
    return by_layer.loc[by_layer.validation_accuracy.idxmax()]


def rows_for(model: str, presentation: str, run_id: str, metrics: pd.DataFrame, rates: pd.DataFrame) -> list[dict]:
    rows = []
    for domain in ["facts", "companies"]:
        for probe in PROBES:
            best = best_layer(metrics, probe, domain)
            cell = rates[(rates.arm == "domain_swap") & (rates.probe == probe) & (rates.booked_domain == domain)
                         & (rates.layer == best.layer) & (rates.normalization == "L")]
            row = {"model": model, "presentation": presentation, "run_id": run_id, "probe": probe,
                   "booked_domain": domain, "layer": int(best.layer), "validation_accuracy": best.validation_accuracy,
                   "booked_accuracy": best.booked_accuracy, "clears_bar": best.booked_accuracy >= BAR}
            for kind in KINDS:
                for calibrated, name in [(False, "raw"), (True, "calibrated")]:
                    part = cell[(cell.kind == kind) & (cell.calibrated == calibrated)]
                    row[f"mean_L_{kind}_{name}"] = part.rate.mean()
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--output", default="presentation_effect.csv", help="File name under the experiment's figures/.")
    args = parser.parse_args()

    loaded = runs.load(args.run_ids)
    rows = []
    for run in loaded:
        model = re.search(r"probes, (\S+),", run["note"]).group(1)
        how = re.search(r"(presentation|bos)=([\w-]+)", run["note"])
        presentation = how.group(2) if how.group(1) == "presentation" else f"raw bos={how.group(2)}"
        rows += rows_for(model, presentation, run["run_id"], pd.read_csv(run["dir"] / "outputs" / "probe_metrics.csv"),
                         pd.read_csv(store_root() / run["run_id"] / "probe_rates" / f"{model}.csv"))
    table = pd.DataFrame(rows).sort_values(["model", "probe", "booked_domain", "presentation"])
    runs.save(table, experiments_dir() / "004-gemma-probe-bos" / "figures" / args.output, loaded)
    print(table.drop(columns=["run_id", "clears_bar"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
