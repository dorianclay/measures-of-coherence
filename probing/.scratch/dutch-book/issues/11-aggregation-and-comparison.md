# How do per-family rates aggregate and compare across prevision sources?

Type: grilling
Status: open
Blocked by: 07, 08, 10

## Question

Both prevision sources now exist ([Produce probe previsions under the domain-swap protocol](08-probe-previsions.md), [Produce elicited previsions for every family statement](10-elicited-previsions.md)), so the aggregation and comparison analysis can be specified precisely:

- **Roll-up.** Negation-pair and conjunction-family rates are never pooled or compared with each other (different family shapes, per [Which rate-of-loss formulation(s) do we compute?](01-rate-of-loss-formulation.md)). How do per-family L values roll up within a shape — distribution, mean/median — per dataset × model × prevision source × arm (raw/calibrated, domain-swap/atomic-only, base/instruct)?
- **Independence.** Families share statements (the same atomic sentence appears in a negation pair and in conjunction families built from it), so per-family rates aren't independent samples. What's the clustering or bootstrap unit for any test or interval — by negation pair, by source atom, or something else?
- **Stratification.** Conjunction polarities are roughly balanced (~25% each, per [Build the event families from the structured datasets](03-build-event-families.md)). Should results be stratified by polarity, and if so, at what stage (reporting only, or the test itself)?
- **Probe vs. elicited comparison.** What's the statistical test (or effect size) for probe rate of loss vs. elicited rate of loss on the same families, given the non-independence above?
- **Correlates.** Does rate of loss correlate with booked-domain accuracy (probe or elicited) or with calibration error (Brier raw − Brier calibrated, or fitted T)? [Produce probe previsions under the domain-swap protocol](08-probe-previsions.md) found `gemma-4-12B-it`'s linear probe at chance while its elicited previsions ([Produce elicited previsions for every family statement](10-elicited-previsions.md)) pass comfortably — does the correlate analysis explain or just record this split?
- **Reference points.** The random-prevision ceilings from [Implement and validate the rate-of-loss solver](07-rate-of-loss-solver.md) (mean L 0.162 negation pairs, 0.269 conjunctions) and the label-prevision floor (0 except the two Nile families) are the axes any plot or table should be drawn against.
- Every probe rate must be reported next to that probe's accuracy on the booked domain (domain swap), so a reader can see which rates come from a probe that could plausibly have learned the domain.

Resolve by deciding the concrete roll-up statistics, the test/interval procedure, and how correlates and reference points are presented — precise enough that [Write up the experiments and every decision behind them](09-experiment-write-up.md) can implement it directly.
