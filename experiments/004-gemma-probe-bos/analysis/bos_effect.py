"""Do linear probes recover when the BOS token is fixed? Compares 004's arms with 001's results on the same models.

    uv run python experiments/004-gemma-probe-bos/analysis/bos_effect.py --probes <run-id> ... --checks <run-id> ... \
        --legacy-activations ~/code/Probes/activations

--probes are 004's probe runs, --checks its bos=tokenizer facts runs. Writes figures/bos_effect.csv (headline probe
accuracy and mean L, per model, arm and booked domain) and figures/activations_match_001.csv (mean cosine similarity
on facts, per layer, between 001's saved activations and each of 004's), each with a sidecar.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import runs
from analysis.dutch_book_rates import headline_probe_layer
from coherence.paths import experiments_dir, store_root

KINDS = ["negation_pair", "conjunction"]
RESULTS_001 = experiments_dir() / "001-dutch-book" / "results"


def headline(metrics: pd.DataFrame, model: str, domain: str) -> tuple[int, pd.Series]:
    layer = headline_probe_layer(metrics, model, domain)
    row = metrics[(metrics.model == model) & (metrics.arm == "domain_swap") & (metrics.probe == "linear")
                  & (metrics.seed == 0) & (metrics.layer == layer) & (metrics.booked_domain == domain)].iloc[0]
    return layer, row


def row_004(run: dict) -> list[dict]:
    model = re.search(r"probes, (\S+),", run["note"]).group(1)
    bos = re.search(r"bos=(\w+)", run["note"]).group(1)
    metrics = pd.read_csv(run["dir"] / "outputs" / "probe_metrics.csv")
    rates = pd.read_csv(store_root() / run["run_id"] / "probe_rates" / f"{model}.csv")
    rows = []
    for domain in ["facts", "companies"]:
        layer, best = headline(metrics, model, domain)
        cell = rates[(rates.arm == "domain_swap") & (rates.probe == "linear") & (rates.seed == 0)
                     & (rates.layer == layer) & (rates.booked_domain == domain) & ~rates.calibrated
                     & (rates.normalization == "L")]
        rows.append({"model": model, "source": f"004 bos={bos}", "run_id": run["run_id"], "booked_domain": domain,
                     "layer": layer, "validation_accuracy": best.validation_accuracy,
                     "booked_accuracy": best.booked_accuracy,
                     **{f"mean_L_{k}": cell[cell.kind == k].rate.mean() for k in KINDS}})
    return rows


def row_001(model: str) -> list[dict]:
    metrics = pd.read_csv(RESULTS_001 / "probe_metrics" / f"{model}.csv")
    rollup = pd.read_csv(RESULTS_001 / "aggregation" / "rollup.csv")
    rows = []
    for domain in ["facts", "companies"]:
        layer, best = headline(metrics, model, domain)
        cell = rollup[(rollup.model == model) & (rollup.source == "probe") & ~rollup.calibrated
                      & (rollup.dataset == domain)].set_index("family_shape").mean_L
        rows.append({"model": model, "source": "001", "run_id": "", "booked_domain": domain, "layer": layer,
                     "validation_accuracy": best.validation_accuracy, "booked_accuracy": best.booked_accuracy,
                     **{f"mean_L_{k}": cell[k] for k in KINDS}})
    return rows


def cosine_by_layer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    return ((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))).mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probes", nargs="+", required=True)
    parser.add_argument("--checks", nargs="+", required=True)
    parser.add_argument("--legacy-activations", type=Path, required=True)
    args = parser.parse_args()

    probes, checks = runs.load(args.probes), runs.load(args.checks)
    rows = [r for run in probes for r in row_004(run)]
    models = sorted({r["model"] for r in rows})
    table = pd.DataFrame(rows + [r for m in models for r in row_001(m)]).sort_values(["model", "booked_domain", "source"])

    matches = []
    for run in checks + probes:
        activations_run = re.search(r"on activations (\S+)", run["note"])
        run_id = activations_run.group(1) if activations_run else run["run_id"]
        model = re.search(r"(?:probes|activations), (\S+),", run["note"]).group(1)
        ours = np.load(store_root() / run_id / model / "facts.npy", mmap_mode="r")
        legacy = np.load(args.legacy_activations / model / "facts.npy", mmap_mode="r")
        matches.append({"model": model, "compared": re.search(r"bos=(\w+)", run["note"]).group(1), "run_id": run_id,
                        **{f"layer_index_{i}": c for i, c in enumerate(cosine_by_layer(ours, legacy))}})
    matches = pd.DataFrame(matches)

    figures = experiments_dir() / "004-gemma-probe-bos" / "figures"
    runs.save(table, figures / "bos_effect.csv", probes)
    runs.save(matches, figures / "activations_match_001.csv", checks + probes)
    print(table.drop(columns="run_id").round(3).to_string(index=False))
    print(matches.drop(columns="run_id").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
