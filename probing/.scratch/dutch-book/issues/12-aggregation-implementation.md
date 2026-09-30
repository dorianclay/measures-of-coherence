# Implement the aggregation and comparison analysis

Type: task (AFK)
Status: claimed
Blocked by: 11

## Question

Implement and run the analysis decided in [How do per-family rates aggregate and compare across prevision sources?](11-aggregation-and-comparison.md), reading from `results/dutch_book/probe_rates/`, `results/dutch_book/elicited_rates/`, `probe_metrics/` and `elicitation_metrics/` (all per model, per [08](08-probe-previsions.md) and [10](10-elicited-previsions.md)).

- For each dataset (facts, companies) × family shape (negation pair, conjunction) × model × prevision source × arm (primary: domain-swap linear raw+calibrated for probes, logprob/t0 raw+calibrated for elicitation), compute mean and median L, and produce one box plot per dataset × family shape showing the full distribution, with reference lines for the random-prevision ceiling and label floor.
- Implement the pair-level (dyadic) bootstrap exactly as decided: resample negation pairs with replacement (tracked as distinct copies), give each conjunction family a bootstrap multiplicity from its two constituent pairs' co-occurrence count, B = 10,000, percentile CIs. Generate the resampled pair-pools once per dataset × family shape and reuse them across every model/source/arm.
- Compute the probe-vs-elicited paired-difference bootstrap distribution per dataset × family shape × model, using the same resampled draws for both sources.
- Add a descriptive (reporting-only) polarity breakdown for conjunction families.
- Compute the correlate analysis: Spearman correlation of rolled-up mean L (raw arm) against booked-domain accuracy, and against calibration error (Brier raw − Brier calibrated, and/or fitted T), one point per (model × booked domain × prevision source), with `gemma-4-12B-it` labeled on the scatter.
- Every probe rate must sit next to that probe's booked-domain accuracy (domain swap) in the output tables.

Done when the tables (with mean/median, CIs, reference rows) and figures (box plots, correlate scatters) exist for both datasets and both family shapes, and the answer records where they live and any surprises in the results (e.g. whether the `gemma-4-12B-it` split shows up elsewhere, whether the paired-difference CIs exclude zero).

## Comments

**2026-09-30: code done, validated locally with tests; the full run over real results is still to do.** Still claimed.

`Aggregate_Rates.py` implements everything decided on [How do per-family rates aggregate and compare across prevision sources?](11-aggregation-and-comparison.md):
- Headline slice: domain-swap linear (best layer by validation accuracy per model × booked domain) for probes, logprob/t0 for elicitation, raw + calibrated, normalization L.
- Mean/median roll-up per (dataset, family shape, model, source, calibrated) → `rollup.csv`.
- The pair-level dyadic bootstrap exactly as decided (see the module docstring): negation pairs resampled with replacement, each conjunction family's bootstrap multiplicity is its two constituent pairs' resampled-copy co-occurrence count, same resampled pair-pools reused across every model/source/arm within a dataset → `bootstrap_ci.csv` (95% percentile CIs, B=10,000 by default).
- Probe-vs-elicited paired-difference bootstrap, same draws for both sources → `paired_difference.csv` (flags whether each CI excludes zero).
- Descriptive polarity breakdown for conjunction families (reporting-only, not a bootstrap stratum) → `polarity_breakdown.csv`, using the same C1/C2 polarity label `Build_Event_Families.py` already computes for its own summary.
- Spearman correlates (rate of loss vs. booked accuracy, and vs. Brier raw − Brier calibrated), one point per (model, dataset, source), `gemma-4-12B-it` labeled → `correlates.csv` + scatter figures.
- Reference points (ceiling from `control_random.csv`, floor from `control_labels.csv`) → `reference_points.csv`, plus reference lines on every box plot.
- Box plots per (dataset, family shape), raw previsions, all models × both sources → `figures/box_<dataset>_<kind>.png`.

Tested (`tests/test_aggregate_rates.py`, 9 tests, full suite 19/19 passing):
- The dyadic bootstrap weighting hand-computed against a tiny 3-pair/2-family fixture (including the all-zero-weight → NaN edge case, and a family missing a rate).
- `percentile_ci` against a known uniform sample.
- `load_pair_graph` and `load_conjunction_polarity` against the real `event_families/*.json` (family counts, a specific conjunction's parents, and the polarity counts cross-checked against `Build_Event_Families.py`'s own summary).
- `discover_models`.
- A full CLI run end-to-end on a tiny synthetic model (real family files, fabricated tiny rate/metric CSVs) confirming headline-layer selection and that every output file gets written.

**Not run on the real 8-model results yet** — handing the script off for the author to run rather than running it myself. Command:

```
python Aggregate_Rates.py \
  --families event_families/facts_families.json event_families/companies_families.json \
  --output-dir results/dutch_book/aggregation
```

(models, results-dir, bootstrap-draws, ci-level, seed all have sensible defaults — see `--help`). Should finish in a couple of minutes on a laptop; no GPU or cluster needed. Once it's run, resolve this ticket with a summary of the tables/figures and anything notable in the results (the `gemma-4-12B-it` split, which paired differences exclude zero, correlate strengths).
