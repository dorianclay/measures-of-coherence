# Implement the aggregation and comparison analysis

Type: task (AFK)
Status: open
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
