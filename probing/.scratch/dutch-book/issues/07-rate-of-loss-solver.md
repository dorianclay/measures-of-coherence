# Implement and validate the rate-of-loss solver

Type: task (AFK)
Status: resolved
Blocked by: 01, 03

## Question

Implement the formulations chosen in [Which rate-of-loss formulation(s) do we compute?](01-rate-of-loss-formulation.md): L, plus ρ for conjunction families. The LP shapes and the reference script `rate_of_loss_check.py` are on branch `research/rate-of-loss-formulation`. Implement the formulations as a solver over an event family (atoms × events membership matrix + previsions → rate of loss, plus the optimal stakes). Validate it before any real previsions touch it:

- Closed-form oracles from 01 for negation pairs and conjunction families.
- Label previsions give exactly 0 on every family with `labels_consistent: true`. The two facts families flagged false (the contested Nile fact) must give a positive rate, which checks the solver in the other direction.
- Random previsions give a positive rate, which sets the ceiling for the random control.

Build test-first (`/tdd`). Done when the solver runs over all families from 03 with label and random previsions and the results are recorded.

## Answer

Done, test-first, at two seams agreed with the author.

- **`rate_of_loss(atoms, previsions, normalization)`** in `DutchBook.py` returns the rate and the gambler's optimal stakes, for normalization `"L"` (primary) or `"rho"`.
- **The `DutchBook.py` command line** books every family in the given family files against previsions and writes one row per (bookie, family, normalization).
  - L is computed for every family; ρ only for conjunction families.
  - Built-in controls: `--previsions labels` and `--previsions random --seed N`.
  - **Input interface**, which [Produce probe previsions under the domain-swap protocol](08-probe-previsions.md) and [Produce elicited previsions for every family statement](10-elicited-previsions.md) must write: a CSV with `family_id, event_index, prevision`, plus any grouping columns (model, layer, source, arm, seed, template, calibrated, ...). Each distinct combination of grouping columns is one bookie, and the grouping columns pass through to the output. A family with missing events is an error.
  - **Output columns:** the grouping columns, then `family_id, kind, labels_consistent, normalization, rate`.
- **Tests:** 10, in `tests/test_dutch_book.py`; run with `uv run pytest`. pytest was added as a dev dependency. Expected values come from independent sources:
  - SSK Theorem 2;
  - the research counterexamples, where L and ρ rank two conjunction previsions in opposite orders;
  - hand-derived worked examples;
  - the research closed form, checked on 300 random conjunction previsions;
  - the label control on the real families.

**Controls** (`results/dutch_book/control_{labels,random}.csv`, 3,251 rates each):

| control | negation pairs, mean L | conjunction families, mean L | conjunction families, mean ρ |
|---|---|---|---|
| labels | 0 (all 1,047) | 0 on all 1,100 consistent families; 1/3 and 1/2 on the two contested-Nile families | same as L |
| random (seed 0) | 0.162 (analytic 1/6) | 0.269 | 0.336 |

The random means are the ceilings to read real previsions against. They differ by family shape, which is another reason never to compare L across shapes.
