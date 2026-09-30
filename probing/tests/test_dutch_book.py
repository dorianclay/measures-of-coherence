import pytest

from DutchBook import rate_of_loss

# Atoms are rows, events are columns: 1 if the event is true in that atom
NEGATION_PAIR = [[1, 0], [0, 1]]

# Events [C1, not-C1, C2, not-C2, C1 and C2]; atoms (C1, C2) = TT, TF, FT, FF
CONJUNCTION = [
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0],
]


def test_negation_pair_overpriced_by_a_fifth_loses_a_tenth_per_unit_stake():
    # p(A) + p(not-A) = 1.2: sell both for 1.2, pay out exactly 1; stake 1 unit total -> 0.2 / 2
    assert rate_of_loss(NEGATION_PAIR, [0.7, 0.5]).rate == pytest.approx(0.1)


def test_bookie_escrow_rate_on_overpriced_negation_pair_matches_ssk_theorem_2():
    # SSK (1998) Thm 2: for a partition priced at total s > 1, rho = (s - 1) / s
    assert rate_of_loss(NEGATION_PAIR, [0.7, 0.5], normalization="rho").rate == pytest.approx(0.2 / 1.2)


def test_total_stake_and_bookie_escrow_rank_conjunction_families_in_opposite_orders():
    # Research counterexample (formulation ticket), C1 = A and C2 = not-B:
    # p1 violates only p(A) + p(not-A) = 1 (by 0.2, 2 statements); p2 only p(C1) + p(C2) - p(C1 and C2) <= 1 (by 0.33, 3 statements)
    p1 = [0.70, 0.50, 0.50, 0.50, 0.25]
    p2 = [0.80, 0.20, 0.80, 0.20, 0.27]
    assert rate_of_loss(CONJUNCTION, p1).rate == pytest.approx(0.10)
    assert rate_of_loss(CONJUNCTION, p2).rate == pytest.approx(0.11)
    assert rate_of_loss(CONJUNCTION, p1, normalization="rho").rate == pytest.approx(0.2 / 1.2)
    assert rate_of_loss(CONJUNCTION, p2, normalization="rho").rate == pytest.approx(0.1416, abs=1e-4)


def test_previsions_from_a_mixture_of_atoms_are_coherent():
    # 0.1 TT + 0.2 TF + 0.3 FT + 0.4 FF
    p = [0.3, 0.7, 0.4, 0.6, 0.1]
    assert rate_of_loss(CONJUNCTION, p).rate == pytest.approx(0, abs=1e-9)
    assert rate_of_loss(CONJUNCTION, p, normalization="rho").rate == pytest.approx(0, abs=1e-9)


def test_certainty_in_both_a_statement_and_its_negation_is_maximally_incoherent():
    # d = p(A) + p(not-A) - 1 = 1, so L = d / 2 and rho = d / (1 + d)
    assert rate_of_loss(NEGATION_PAIR, [1, 1]).rate == pytest.approx(0.5)
    assert rate_of_loss(NEGATION_PAIR, [1, 1], normalization="rho").rate == pytest.approx(0.5)


def test_optimal_stakes_sell_both_sides_of_an_overpriced_negation_pair():
    stakes = rate_of_loss(NEGATION_PAIR, [0.7, 0.5]).stakes
    assert stakes == pytest.approx([-0.5, -0.5])


FAMILY_FILES = ["event_families/facts_families.json", "event_families/companies_families.json"]
CONTESTED_NILE_FAMILIES = {"conj_neg_facts:117", "conj_neg_facts:153"}


def run_cli(*args):
    import subprocess, sys
    subprocess.run([sys.executable, "DutchBook.py", *args], check=True, capture_output=True)


def test_label_previsions_are_coherent_except_where_the_dataset_contradicts_itself(tmp_path):
    import pandas as pd
    out = tmp_path / "rates.csv"
    run_cli("--families", *FAMILY_FILES, "--previsions", "labels", "--output", str(out))
    rates = pd.read_csv(out)
    L = rates[rates.normalization == "L"]
    assert len(L) == 547 + 556 + 500 + 546
    contested = L.family_id.isin(CONTESTED_NILE_FAMILIES)
    assert (L[~contested].rate.abs() < 1e-9).all()
    assert (L[contested].rate > 0).all()


def test_grouping_columns_separate_bookies_and_pass_through(tmp_path):
    import pandas as pd
    pair = "neg_facts:pair:0"
    previsions = pd.DataFrame({
        "model": ["coherent", "coherent", "overconfident", "overconfident"],
        "family_id": [pair] * 4,
        "event_index": [0, 1, 0, 1],
        "prevision": [0.3, 0.7, 0.7, 0.5],
    })
    previsions.to_csv(tmp_path / "previsions.csv", index=False)
    out = tmp_path / "rates.csv"
    run_cli("--families", FAMILY_FILES[0], "--previsions", str(tmp_path / "previsions.csv"), "--output", str(out))
    rates = pd.read_csv(out).set_index("model").rate
    assert rates["coherent"] == pytest.approx(0, abs=1e-9)
    assert rates["overconfident"] == pytest.approx(0.1)


def test_uniform_random_previsions_are_incoherent_on_every_family(tmp_path):
    import pandas as pd
    out = tmp_path / "rates.csv"
    run_cli("--families", *FAMILY_FILES, "--previsions", "random", "--seed", "0", "--output", str(out))
    rates = pd.read_csv(out)
    # Independent uniform prices on A and not-A almost never sum to exactly 1
    assert (rates.rate > 0).all()


def conjunction_closed_form_L(p):
    """Research closed form: L = max over coherence constraints of violation / (number of statements involved).
    Constraints for [C1, not-C1, C2, not-C2, C1 and C2], each written as (violation, statements involved)."""
    c1, n1, c2, n2, c = p
    violations = [
        (abs(c1 + n1 - 1), 2), (abs(c2 + n2 - 1), 2),        # complements
        (c - c1, 2), (c + n1 - 1, 2), (c - c2, 2), (c + n2 - 1, 2),  # conjunction at most each conjunct
        (c1 + c2 - c - 1, 3), (c1 - n2 - c, 3), (c2 - n1 - c, 3), (-n1 - n2 - c + 1, 3),  # at least C1 + C2 - 1
    ]
    return max(0.0, *(v / k for v, k in violations))


def test_rate_matches_the_closed_form_on_random_conjunction_previsions():
    import numpy as np
    rng = np.random.default_rng(1)
    for _ in range(300):
        p = rng.uniform(0, 1, 5)
        assert rate_of_loss(CONJUNCTION, p).rate == pytest.approx(conjunction_closed_form_L(p), abs=1e-9)
