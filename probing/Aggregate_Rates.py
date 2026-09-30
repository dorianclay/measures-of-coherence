"""
Aggregation and Comparison Analysis for Dutch Book Rates of Loss
Date: 2026-09-30

Rolls up per-family rates of loss (from DutchBook.py, via probe_rates/ and elicited_rates/) into headline tables and
figures comparing probe previsions against elicited previsions, per the decisions on the aggregation-and-comparison
ticket (.scratch/dutch-book/issues/11-aggregation-and-comparison.md).

Headline slice: domain-swap linear probes (the layer with the highest validation accuracy per model x booked domain,
from probe_metrics/) and logprob/t0 elicitation (from elicitation_metrics/), both raw and calibrated, normalization L.
Negation-pair and conjunction-family rates are never pooled (different family shapes). Conjunction polarity (which of
C1/not-C1 x C2/not-C2 the family joins, matching Build_Event_Families.py's own labeling) gets a descriptive,
reporting-only breakdown -- it is not a stratification variable in the bootstrap.

Negation-pair families are independent, but conjunction families are not: each is built from two constituent negation
pairs, and that reuse graph is dense enough that "cluster by pair" collapses into one giant component rather than many
small ones. Non-independence is instead handled with a pair-level (dyadic) bootstrap: resample negation pairs with
replacement, and give each conjunction family a bootstrap multiplicity equal to how many times its two constituent
pairs' copies co-occur in that draw (Fafchamps & Gubert's node bootstrap for shared-node dyads). The same resampled
pair-pools are reused across every model, prevision source and arm within a dataset, so the probe-vs-elicited
paired-difference bootstrap is valid (both statistics come from literally the same resampled families each draw).

Requirements:
- numpy library
- pandas library
- scipy library
- matplotlib library
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATASETS = ["facts", "companies"]
FAMILY_SHAPES = ["negation_pair", "conjunction"]
CALIBRATED = [False, True]


# ---------------------------------------------------------------------------
# Pair / conjunction-family graph, and the dyadic bootstrap over it
# ---------------------------------------------------------------------------

def load_pair_graph(family_path) -> tuple[list, dict]:
    """pair_ids (dataset order) and {conjunction_family_id: (pair_idx_a, pair_idx_b)}."""
    data = json.load(open(family_path))
    pair_ids = [p["id"] for p in data["negation_pairs"]]
    pair_index = {pid: i for i, pid in enumerate(pair_ids)}
    conj_pairs = {cf["id"]: (pair_index[cf["pairs"][0]], pair_index[cf["pairs"][1]])
                  for cf in data["conjunction_families"]}
    return pair_ids, conj_pairs


def load_conjunction_polarity(family_path) -> dict:
    """{conjunction_family_id: "C1 pos/C2 neg" etc.}, matching Build_Event_Families.py's own polarity label
    (events[0] is C1, events[2] is C2 -- see CONJUNCTION_ATOMS in DutchBook.py)."""
    data = json.load(open(family_path))
    return {cf["id"]: f"C1 {'neg' if cf['events'][0]['surface_negated'] else 'pos'}"
                       f" / C2 {'neg' if cf['events'][2]['surface_negated'] else 'pos'}"
            for cf in data["conjunction_families"]}


def bootstrap_pair_counts(n_pairs: int, draws: int, seed: int) -> np.ndarray:
    """(draws, n_pairs): how many times each pair is resampled in each draw."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_pairs, np.full(n_pairs, 1 / n_pairs), size=draws)


def bootstrap_stat_negation_pairs(counts: np.ndarray, pair_id_to_rate: dict, pair_ids: list) -> np.ndarray:
    """Weighted mean per draw over negation-pair rates; a pair missing a rate is dropped from every draw."""
    values = np.array([pair_id_to_rate.get(pid, np.nan) for pid in pair_ids])
    present = ~np.isnan(values)
    counts, values = counts[:, present], values[present]
    return (counts * values).sum(axis=1) / counts.sum(axis=1)


def bootstrap_stat_conjunctions(counts: np.ndarray, conj_pairs: dict, family_id_to_rate: dict) -> np.ndarray:
    """Weighted mean per draw over conjunction-family rates, weighting each family by its two parent pairs'
    resampled-copy co-occurrence count; a family missing a rate is dropped from every draw."""
    ids = [fid for fid in conj_pairs if fid in family_id_to_rate]
    a = np.array([conj_pairs[fid][0] for fid in ids])
    b = np.array([conj_pairs[fid][1] for fid in ids])
    values = np.array([family_id_to_rate[fid] for fid in ids])
    weights = counts[:, a] * counts[:, b]  # (draws, n_families)
    denom = weights.sum(axis=1)
    return np.divide((weights * values).sum(axis=1), denom, out=np.full(len(denom), np.nan), where=denom > 0)


def bootstrap_stat(kind: str, counts, pair_ids, conj_pairs, rates: pd.DataFrame) -> np.ndarray:
    rate_by_id = rates.set_index("family_id").rate.to_dict()
    if kind == "negation_pair":
        return bootstrap_stat_negation_pairs(counts, rate_by_id, pair_ids)
    return bootstrap_stat_conjunctions(counts, conj_pairs, rate_by_id)


def percentile_ci(stat: np.ndarray, level: float) -> tuple:
    alpha = (1 - level) / 2
    lo, hi = np.nanpercentile(stat, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Loading and slicing the rate / metric tables
# ---------------------------------------------------------------------------

def load_rates(results_dir: Path, models: list) -> tuple:
    probe = pd.concat([pd.read_csv(results_dir / "probe_rates" / f"{m}.csv") for m in models], ignore_index=True)
    elicited = pd.concat([pd.read_csv(results_dir / "elicited_rates" / f"elicited_{m}.csv")
                          for m in models], ignore_index=True)
    probe_metrics = pd.concat([pd.read_csv(results_dir / "probe_metrics" / f"{m}.csv") for m in models],
                               ignore_index=True)
    elicitation_metrics = pd.concat([pd.read_csv(results_dir / "elicitation_metrics" / f"{m}.csv")
                                     for m in models], ignore_index=True)
    return probe, elicited, probe_metrics, elicitation_metrics


def headline_probe_layer(probe_metrics: pd.DataFrame, model: str, booked_domain: str) -> int:
    rows = probe_metrics[(probe_metrics.model == model) & (probe_metrics.arm == "domain_swap")
                          & (probe_metrics.probe == "linear") & (probe_metrics.booked_domain == booked_domain)]
    return int(rows.loc[rows.validation_accuracy.idxmax(), "layer"])


def headline_probe_rates(probe_rates: pd.DataFrame, probe_metrics: pd.DataFrame,
                          model: str, booked_domain: str) -> pd.DataFrame:
    layer = headline_probe_layer(probe_metrics, model, booked_domain)
    return probe_rates[(probe_rates.model == model) & (probe_rates.arm == "domain_swap")
                        & (probe_rates.probe == "linear") & (probe_rates.seed == 0) & (probe_rates.layer == layer)
                        & (probe_rates.booked_domain == booked_domain) & (probe_rates.normalization == "L")]


def headline_elicited_rates(elicited_rates: pd.DataFrame, model: str, booked_domain: str) -> pd.DataFrame:
    return elicited_rates[(elicited_rates.model == model) & (elicited_rates.method == "logprob")
                           & (elicited_rates.template == "t0") & (elicited_rates.booked_domain == booked_domain)
                           & (elicited_rates.normalization == "L")]


def headline_metric_row(metrics: pd.DataFrame, model: str, booked_domain: str, source: str) -> pd.Series:
    if source == "probe":
        layer = headline_probe_layer(metrics, model, booked_domain)
        rows = metrics[(metrics.model == model) & (metrics.arm == "domain_swap") & (metrics.probe == "linear")
                        & (metrics.seed == 0) & (metrics.layer == layer) & (metrics.booked_domain == booked_domain)]
    else:
        rows = metrics[(metrics.model == model) & (metrics.method == "logprob") & (metrics.template == "t0")
                        & (metrics.booked_domain == booked_domain)]
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# Roll-up table, bootstrap CIs, correlates, reference points
# ---------------------------------------------------------------------------

def rollup_row(rates: pd.DataFrame, dataset: str, model: str, source: str, layer=None) -> list:
    rows = []
    for kind in FAMILY_SHAPES:
        for calibrated in CALIBRATED:
            cell = rates[(rates.kind == kind) & (rates.calibrated == calibrated)]
            rows.append({
                "dataset": dataset, "family_shape": kind, "model": model, "source": source,
                "calibrated": calibrated, "layer": layer, "n_families": len(cell),
                "mean_L": cell.rate.mean(), "median_L": cell.rate.median(),
            })
    return rows


def polarity_rows(rates: pd.DataFrame, polarity: dict, dataset: str, model: str, source: str) -> list:
    """Descriptive (reporting-only) mean/median L by conjunction polarity, raw arm only."""
    cell = rates[(rates.kind == "conjunction") & (~rates.calibrated)].copy()
    cell["polarity"] = cell.family_id.map(polarity)
    rows = []
    for pol, group in cell.groupby("polarity", sort=True):
        rows.append({"dataset": dataset, "model": model, "source": source, "polarity": pol,
                     "n_families": len(group), "mean_L": group.rate.mean(), "median_L": group.rate.median()})
    return rows


def bootstrap_row(rates: pd.DataFrame, dataset: str, model: str, source: str, calibrated: bool, kind: str,
                   counts: np.ndarray, pair_ids: list, conj_pairs: dict, ci_level: float) -> dict:
    cell = rates[(rates.kind == kind) & (rates.calibrated == calibrated)]
    stat = bootstrap_stat(kind, counts, pair_ids, conj_pairs, cell)
    lo, hi = percentile_ci(stat, ci_level)
    return {"dataset": dataset, "family_shape": kind, "model": model, "source": source, "calibrated": calibrated,
            "bootstrap_mean": float(np.nanmean(stat)), "ci_lo": lo, "ci_hi": hi}


def paired_difference_row(probe_rates_cell: pd.DataFrame, elicited_rates_cell: pd.DataFrame, dataset: str,
                           model: str, calibrated: bool, kind: str, counts, pair_ids, conj_pairs,
                           ci_level: float) -> dict:
    probe_stat = bootstrap_stat(kind, counts, pair_ids, conj_pairs,
                                 probe_rates_cell[(probe_rates_cell.kind == kind)
                                                   & (probe_rates_cell.calibrated == calibrated)])
    elicited_stat = bootstrap_stat(kind, counts, pair_ids, conj_pairs,
                                    elicited_rates_cell[(elicited_rates_cell.kind == kind)
                                                         & (elicited_rates_cell.calibrated == calibrated)])
    diff = probe_stat - elicited_stat
    lo, hi = percentile_ci(diff, ci_level)
    return {"dataset": dataset, "family_shape": kind, "model": model, "calibrated": calibrated,
            "mean_diff_probe_minus_elicited": float(np.nanmean(diff)), "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def reference_points(results_dir: Path) -> pd.DataFrame:
    rows = []
    for name, path in [("ceiling_random", "control_random.csv"), ("floor_labels", "control_labels.csv")]:
        df = pd.read_csv(results_dir / path)
        df = df[df.normalization == "L"]
        for dataset in DATASETS:
            in_dataset = df[df.family_id.str.contains(dataset)]
            for kind in FAMILY_SHAPES:
                cell = in_dataset[in_dataset.kind == kind]
                rows.append({"reference": name, "dataset": dataset, "family_shape": kind,
                             "mean_L": cell.rate.mean(), "n_families": len(cell)})
        for kind in FAMILY_SHAPES:
            cell = df[df.kind == kind]
            rows.append({"reference": name, "dataset": "pooled", "family_shape": kind,
                         "mean_L": cell.rate.mean(), "n_families": len(cell)})
    return pd.DataFrame(rows)


def correlate_points(rollup: pd.DataFrame, kind: str, metrics_by_cell: dict) -> pd.DataFrame:
    """One row per (model, dataset, source) for the given family shape: raw-arm rolled-up mean L next to that
    cell's booked accuracy and calibration error (Brier raw - Brier calibrated)."""
    points = rollup[(rollup.family_shape == kind) & (~rollup.calibrated)].copy()
    points["booked_accuracy"] = [metrics_by_cell[(r.dataset, r.model, r.source)].booked_accuracy
                                  for r in points.itertuples()]
    points["brier_delta"] = [metrics_by_cell[(r.dataset, r.model, r.source)].booked_brier_raw
                              - metrics_by_cell[(r.dataset, r.model, r.source)].booked_brier_calibrated
                              for r in points.itertuples()]
    return points


def correlates(rollup: pd.DataFrame, metrics_by_cell: dict) -> pd.DataFrame:
    """Spearman correlation of raw-arm rolled-up mean L against booked accuracy and calibration error,
    per family shape, one point per (model, dataset, source)."""
    rows = []
    for kind in FAMILY_SHAPES:
        points = correlate_points(rollup, kind, metrics_by_cell)
        for correlate_col in ["booked_accuracy", "brier_delta"]:
            valid = points.dropna(subset=["mean_L", correlate_col])
            rho, p = spearmanr(valid.mean_L, valid[correlate_col])
            rows.append({"family_shape": kind, "correlate": correlate_col, "n": len(valid),
                         "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_box(raw_by_model_source: dict, reference: pd.DataFrame, dataset: str, kind: str, path: Path):
    """raw_by_model_source: {(model, source): array of per-family raw L}, ordered by model then source."""
    labels = [f"{model}\n{source}" for model, source in raw_by_model_source]
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels)), 5))
    ax.boxplot(list(raw_by_model_source.values()), tick_labels=labels, showfliers=False)
    ceiling = reference[(reference.reference == "ceiling_random") & (reference.dataset == dataset)
                         & (reference.family_shape == kind)].mean_L.iloc[0]
    floor = reference[(reference.reference == "floor_labels") & (reference.dataset == dataset)
                       & (reference.family_shape == kind)].mean_L.iloc[0]
    ax.axhline(ceiling, color="tab:red", linestyle="--", linewidth=1, label=f"random-prevision ceiling ({ceiling:.3f})")
    ax.axhline(floor, color="tab:green", linestyle="--", linewidth=1, label=f"label-prevision floor ({floor:.3f})")
    ax.set_ylabel("rate of loss (L)")
    ax.set_title(f"{dataset} - {kind} - raw previsions")
    ax.legend(loc="upper right", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlate(points: pd.DataFrame, correlate_col: str, kind: str, rho: float, p: float, path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for source, marker in [("probe", "o"), ("elicited", "^")]:
        sub = points[points.source == source]
        ax.scatter(sub[correlate_col], sub.mean_L, marker=marker, label=source, alpha=0.8)
    highlight = points[points.model == "gemma-4-12B-it"]
    for _, row in highlight.iterrows():
        ax.annotate("gemma-4-12B-it", (row[correlate_col], row.mean_L), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(correlate_col)
    ax.set_ylabel("mean L (raw)")
    ax.set_title(f"{kind}: L vs {correlate_col} (Spearman rho={rho:.2f}, p={p:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def discover_models(results_dir: Path) -> list:
    probe_models = {p.stem for p in (results_dir / "probe_rates").glob("*.csv")}
    elicited_models = {p.stem.removeprefix("elicited_") for p in (results_dir / "elicited_rates").glob("*.csv")}
    return sorted(probe_models & elicited_models)


def raw_family_rates(probe_rates, elicited_rates, probe_metrics, model, dataset, kind) -> dict:
    probe = headline_probe_rates(probe_rates, probe_metrics, model, dataset)
    probe = probe[(probe.kind == kind) & (~probe.calibrated)]
    elicited = headline_elicited_rates(elicited_rates, model, dataset)
    elicited = elicited[(elicited.kind == kind) & (~elicited.calibrated)]
    return {"probe": probe.rate.to_numpy(), "elicited": elicited.rate.to_numpy()}


def main():
    parser = argparse.ArgumentParser(description="Aggregate and compare probe vs. elicited rates of loss.")
    parser.add_argument("--results-dir", default="results/dutch_book", help="Directory with probe_rates/, "
                        "elicited_rates/, probe_metrics/, elicitation_metrics/, control_random.csv, control_labels.csv.")
    parser.add_argument("--families", nargs="+", required=True,
                        help="Event family JSON files, e.g. event_families/facts_families.json event_families/companies_families.json.")
    parser.add_argument("--models", nargs="*", default=None, help="Model names; default is every model with both "
                        "probe and elicited rates in --results-dir.")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    models = args.models or discover_models(results_dir)
    probe_rates, elicited_rates, probe_metrics, elicitation_metrics = load_rates(results_dir, models)
    reference = reference_points(results_dir)
    reference.to_csv(output_dir / "reference_points.csv", index=False)

    pair_graphs, polarity_by_dataset = {}, {}
    for path in args.families:
        pair_ids, conj_pairs = load_pair_graph(path)
        dataset = next(d for d in DATASETS if d in Path(path).stem)
        pair_graphs[dataset] = (pair_ids, conj_pairs,
                                 bootstrap_pair_counts(len(pair_ids), args.bootstrap_draws, args.seed))
        polarity_by_dataset[dataset] = load_conjunction_polarity(path)

    rollup_rows, bootstrap_rows, paired_diff_rows, polarity_breakdown_rows = [], [], [], []
    metrics_by_cell = {}
    for model in models:
        for dataset in DATASETS:
            pair_ids, conj_pairs, counts = pair_graphs[dataset]
            p_rates = headline_probe_rates(probe_rates, probe_metrics, model, dataset)
            e_rates = headline_elicited_rates(elicited_rates, model, dataset)
            p_layer = headline_probe_layer(probe_metrics, model, dataset)

            rollup_rows += rollup_row(p_rates, dataset, model, "probe", layer=p_layer)
            rollup_rows += rollup_row(e_rates, dataset, model, "elicited")
            metrics_by_cell[(dataset, model, "probe")] = headline_metric_row(probe_metrics, model, dataset, "probe")
            metrics_by_cell[(dataset, model, "elicited")] = headline_metric_row(
                elicitation_metrics, model, dataset, "elicited")
            polarity_breakdown_rows += polarity_rows(p_rates, polarity_by_dataset[dataset], dataset, model, "probe")
            polarity_breakdown_rows += polarity_rows(e_rates, polarity_by_dataset[dataset], dataset, model,
                                                       "elicited")

            for kind in FAMILY_SHAPES:
                for calibrated in CALIBRATED:
                    bootstrap_rows.append(bootstrap_row(p_rates, dataset, model, "probe", calibrated, kind,
                                                          counts, pair_ids, conj_pairs, args.ci_level))
                    bootstrap_rows.append(bootstrap_row(e_rates, dataset, model, "elicited", calibrated, kind,
                                                          counts, pair_ids, conj_pairs, args.ci_level))
                    paired_diff_rows.append(paired_difference_row(p_rates, e_rates, dataset, model, calibrated,
                                                                     kind, counts, pair_ids, conj_pairs,
                                                                     args.ci_level))

    rollup = pd.DataFrame(rollup_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    paired_diff = pd.DataFrame(paired_diff_rows)
    polarity_breakdown = pd.DataFrame(polarity_breakdown_rows)
    rollup.to_csv(output_dir / "rollup.csv", index=False)
    bootstrap.to_csv(output_dir / "bootstrap_ci.csv", index=False)
    paired_diff.to_csv(output_dir / "paired_difference.csv", index=False)
    polarity_breakdown.to_csv(output_dir / "polarity_breakdown.csv", index=False)

    corr = correlates(rollup, metrics_by_cell)
    corr.to_csv(output_dir / "correlates.csv", index=False)
    for kind in FAMILY_SHAPES:
        points = correlate_points(rollup, kind, metrics_by_cell)
        for correlate_col in ["booked_accuracy", "brier_delta"]:
            row = corr[(corr.family_shape == kind) & (corr.correlate == correlate_col)].iloc[0]
            plot_correlate(points, correlate_col, kind, row.spearman_rho, row.p_value,
                            output_dir / "figures" / f"correlate_{kind}_{correlate_col}.png")

    for dataset in DATASETS:
        for kind in FAMILY_SHAPES:
            raw_by_model_source = {}
            for model in models:
                raw = raw_family_rates(probe_rates, elicited_rates, probe_metrics, model, dataset, kind)
                raw_by_model_source[(model, "probe")] = raw["probe"]
                raw_by_model_source[(model, "elicited")] = raw["elicited"]
            plot_box(raw_by_model_source, reference, dataset, kind,
                     output_dir / "figures" / f"box_{dataset}_{kind}.png")

    print(f"Saved roll-up ({len(rollup)} rows), bootstrap CIs ({len(bootstrap)} rows), "
          f"paired differences ({len(paired_diff)} rows), polarity breakdown ({len(polarity_breakdown)} rows) "
          f"and correlates ({len(corr)} rows) to {output_dir}")
    print(f"Saved {len(DATASETS) * len(FAMILY_SHAPES)} box plots and {len(FAMILY_SHAPES) * 2} correlate "
          f"scatters to {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
