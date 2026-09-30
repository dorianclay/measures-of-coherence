import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Aggregate_Rates import (
    bootstrap_stat_conjunctions,
    bootstrap_stat_negation_pairs,
    discover_models,
    load_conjunction_polarity,
    load_pair_graph,
    percentile_ci,
)

# --- bootstrap mechanics, hand-computed against a fixed (non-random) counts array ---

PAIR_IDS = ["P0", "P1", "P2"]
PAIR_RATES = {"P0": 0.1, "P1": 0.2, "P2": 0.3}
CONJ_PAIRS = {"F0": (0, 1), "F1": (1, 2)}  # F0 = P0 & P1, F1 = P1 & P2
CONJ_RATES = {"F0": 0.5, "F1": 0.7}


def test_negation_pair_bootstrap_is_the_counts_weighted_mean_of_pair_rates():
    counts = np.array([[2, 1, 0], [0, 1, 2]])
    stat = bootstrap_stat_negation_pairs(counts, PAIR_RATES, PAIR_IDS)
    assert stat == pytest.approx([(2 * 0.1 + 1 * 0.2) / 3, (1 * 0.2 + 2 * 0.3) / 3])


def test_conjunction_bootstrap_weights_each_family_by_its_two_parent_pairs_copy_counts():
    # draw 0: counts [2,1,0] -> F0 weight 2*1=2, F1 weight 1*0=0 -> only F0 contributes
    # draw 1: counts [0,1,2] -> F0 weight 0*1=0, F1 weight 1*2=2 -> only F1 contributes
    counts = np.array([[2, 1, 0], [0, 1, 2]])
    stat = bootstrap_stat_conjunctions(counts, CONJ_PAIRS, CONJ_RATES)
    assert stat == pytest.approx([0.5, 0.7])


def test_conjunction_bootstrap_is_nan_when_no_family_has_both_parent_pairs_present():
    # counts [0,0,3]: F0 weight 0*0=0, F1 weight 0*3=0 -> no family has any weight this draw
    counts = np.array([[0, 0, 3]])
    stat = bootstrap_stat_conjunctions(counts, CONJ_PAIRS, CONJ_RATES)
    assert np.isnan(stat[0])


def test_bootstrap_stat_drops_families_missing_a_rate_rather_than_erroring():
    # F1 has no rate on this cell (e.g. calibrated arm not present); only F0 should be used
    stat = bootstrap_stat_conjunctions(np.array([[2, 1, 0]]), CONJ_PAIRS, {"F0": 0.5})
    assert stat == pytest.approx([0.5])


def test_percentile_ci_is_symmetric_around_the_median_for_a_uniform_sample():
    stat = np.linspace(0, 1, 1001)
    lo, hi = percentile_ci(stat, level=0.95)
    assert lo == pytest.approx(0.025, abs=1e-6)
    assert hi == pytest.approx(0.975, abs=1e-6)


# --- pair graph loading, against the real event-family files ---

FAMILY_FILES = ["event_families/facts_families.json", "event_families/companies_families.json"]


def test_pair_graph_matches_the_known_family_counts_and_a_specific_conjunction():
    pair_ids, conj_pairs = load_pair_graph(FAMILY_FILES[0])
    assert len(pair_ids) == 547
    assert len(conj_pairs) == 556
    # conj_neg_facts:0 is built from neg_facts:pair:176 and neg_facts:pair:376
    a, b = conj_pairs["conj_neg_facts:0"]
    assert pair_ids[a] == "neg_facts:pair:176"
    assert pair_ids[b] == "neg_facts:pair:376"


def test_conjunction_polarity_matches_build_event_families_own_summary_counts():
    polarity = load_conjunction_polarity(FAMILY_FILES[0])
    counts = pd.Series(polarity.values()).value_counts().to_dict()
    # Build_Event_Families.py records this same breakdown in the JSON's own summary
    expected = {"C1 pos / C2 neg": 130, "C1 neg / C2 pos": 149, "C1 neg / C2 neg": 139, "C1 pos / C2 pos": 138}
    assert counts == expected


# --- model discovery ---

def test_discover_models_intersects_probe_and_elicited_model_names(tmp_path):
    (tmp_path / "probe_rates").mkdir()
    (tmp_path / "elicited_rates").mkdir()
    for name in ["A", "B"]:
        (tmp_path / "probe_rates" / f"{name}.csv").touch()
        (tmp_path / "elicited_rates" / f"{name}.csv").touch()
    (tmp_path / "probe_rates" / "C.csv").touch()  # no matching elicited file -> excluded
    assert discover_models(tmp_path) == ["A", "B"]


# --- end-to-end CLI wiring, on a tiny synthetic model against the real family files ---

def run_cli(*args):
    subprocess.run([sys.executable, "Aggregate_Rates.py", *args], check=True, capture_output=True)


FACTS_PAIRS = ["neg_facts:pair:0", "neg_facts:pair:1", "neg_facts:pair:176", "neg_facts:pair:376"]
COMPANIES_PAIRS = ["neg_companies:pair:0", "neg_companies:pair:1", "neg_companies:pair:253", "neg_companies:pair:334"]


def _rate_rows(pairs, conj_id, extra_cols):
    rows = []
    for calibrated in [False, True]:
        for pid in pairs:
            rows.append({**extra_cols, "calibrated": calibrated, "family_id": pid, "kind": "negation_pair",
                        "labels_consistent": True, "normalization": "L", "rate": 0.1})
        rows.append({**extra_cols, "calibrated": calibrated, "family_id": conj_id, "kind": "conjunction",
                    "labels_consistent": True, "normalization": "L", "rate": 0.2})
    return rows


def _write_synthetic_results(results_dir: Path):
    (results_dir / "probe_rates").mkdir(parents=True)
    (results_dir / "elicited_rates").mkdir(parents=True)
    (results_dir / "probe_metrics").mkdir(parents=True)
    (results_dir / "elicitation_metrics").mkdir(parents=True)

    probe_rows, metric_rows = [], []
    for domain, pairs, conj_id in [("facts", FACTS_PAIRS, "conj_neg_facts:0"),
                                    ("companies", COMPANIES_PAIRS, "conj_neg_companies:0")]:
        for layer, val_acc in [(4, 0.6), (7, 0.9)]:  # layer 7 should win
            probe_rows += _rate_rows(pairs, conj_id, {
                "model": "ToyModel", "arm": "domain_swap", "probe": "linear", "seed": 0,
                "layer": layer, "depth": layer / 8, "booked_domain": domain,
            })
            metric_rows.append({
                "model": "ToyModel", "arm": "domain_swap", "probe": "linear", "seed": 0, "layer": layer,
                "depth": layer / 8, "booked_domain": domain, "training_statements": 100,
                "temperature": 1.0, "validation_accuracy": val_acc, "booked_accuracy": 0.7,
                "booked_brier_raw": 0.3, "booked_brier_calibrated": 0.2,
            })
    pd.DataFrame(probe_rows).to_csv(results_dir / "probe_rates" / "ToyModel.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(results_dir / "probe_metrics" / "ToyModel.csv", index=False)

    elicited_rows, elicitation_metric_rows = [], []
    for domain, pairs, conj_id in [("facts", FACTS_PAIRS, "conj_neg_facts:0"),
                                    ("companies", COMPANIES_PAIRS, "conj_neg_companies:0")]:
        elicited_rows += _rate_rows(pairs, conj_id, {
            "model": "ToyModel", "method": "logprob", "template": "t0", "booked_domain": domain,
        })
        elicitation_metric_rows.append({
            "model": "ToyModel", "method": "logprob", "template": "t0", "booked_domain": domain,
            "booked_accuracy": 0.75, "booked_brier_raw": 0.25, "booked_brier_calibrated": 0.15,
        })
    pd.DataFrame(elicited_rows).to_csv(results_dir / "elicited_rates" / "ToyModel.csv", index=False)
    pd.DataFrame(elicitation_metric_rows).to_csv(results_dir / "elicitation_metrics" / "ToyModel.csv", index=False)

    control_rows = []
    for pairs, conj_id in [(FACTS_PAIRS, "conj_neg_facts:0"), (COMPANIES_PAIRS, "conj_neg_companies:0")]:
        for pid in pairs:
            control_rows.append({"family_id": pid, "kind": "negation_pair", "labels_consistent": True,
                                 "normalization": "L", "rate": 0.15})
        control_rows.append({"family_id": conj_id, "kind": "conjunction", "labels_consistent": True,
                             "normalization": "L", "rate": 0.25})
    pd.DataFrame(control_rows).to_csv(results_dir / "control_random.csv", index=False)
    pd.DataFrame([{**row, "rate": 0.0} for row in control_rows]).to_csv(results_dir / "control_labels.csv",
                                                                          index=False)


def test_cli_runs_end_to_end_and_headline_layer_selection_picks_the_higher_validation_accuracy(tmp_path):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "out"
    _write_synthetic_results(results_dir)

    run_cli("--results-dir", str(results_dir), "--families", *FAMILY_FILES, "--models", "ToyModel",
            "--bootstrap-draws", "20", "--output-dir", str(output_dir))

    rollup = pd.read_csv(output_dir / "rollup.csv")
    probe_rows = rollup[rollup.source == "probe"]
    assert (probe_rows.layer == 7).all()  # layer 7 has the higher validation_accuracy

    for name in ["rollup.csv", "bootstrap_ci.csv", "paired_difference.csv", "polarity_breakdown.csv",
                "correlates.csv", "reference_points.csv"]:
        assert (output_dir / name).exists()
    figures = list((output_dir / "figures").glob("*.png"))
    assert len(figures) == len(["facts", "companies"]) * 2 + 2 * 2  # box plots + correlate scatters
