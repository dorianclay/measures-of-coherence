# How do per-family rates aggregate and compare across prevision sources?

Type: grilling
Status: resolved
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

## Answer

**A key fact surfaced before deciding anything:** negation-pair families are mutually independent (each built from a distinct base statement — 0 duplicates in facts, 1 accidental duplicate out of 500 in companies), but conjunction families are not. Each conjunction family is built from exactly two [constituent pairs](../../../CONTEXT.md) (`event_families/*.json` records this directly). Tracing that structure: 332/477 (facts) and 313/438 (companies) of the negation pairs actually used feed ≥ 2 conjunction families (up to 7–11 reuses of one pair), and these overlaps chain into one giant connected component — 439/556 conjunction families (facts) and 406/546 (companies) are all transitively linked by shared constituent pairs. "Cluster by negation pair" as a simple partition doesn't work: there's effectively one blob, not many small clusters. This drove the independence/test decisions below.

**Roll-up:** for each headline cell (family shape × model × prevision source × raw/calibrated, using the primary arms already decided — domain-swap linear for probes, logprob/t0 for elicitation), report mean **and** median, plus one box plot per dataset showing the full per-family L distribution.

**Independence and the probe-vs-elicited test — a pair-level (dyadic) bootstrap:**
- Negation-pair rates are treated as i.i.d. and go straight into a standard bootstrap.
- Conjunction-family rates are not resampled as a partition. Instead: draw negation pairs with replacement to the original pool size, tracking each draw as a distinct copy (not just a count). Each conjunction family's bootstrap multiplicity is the number of (copy-of-P1, copy-of-P2) co-occurrences across the two resampled copy-sets, so a family can appear 0, 1, or several times per draw, and heavily-reused pairs are naturally reflected more often.
- B = 10,000 draws; percentile (not normal-approximation) CIs, since the reuse structure gives no reason to assume a symmetric sampling distribution.
- The same B resampled pair-pools are generated once per dataset × family shape and reused identically across every model, prevision source, and arm — required for the probe-vs-elicited paired difference (same resampled bootstrap draw feeds both sources' statistic) to be valid, and free computationally since it's the same draws each time.

**Stratification:** reporting-only. Conjunction polarity is a property of the conjunction (which two pairs got combined and how), not of a pair — stratifying the resample would need a different scheme, and there's no stated hypothesis that loss depends on polarity yet. Add a descriptive polarity breakdown table/plot; revisit only if it shows a real gap.

**Correlates:** one point per (model × booked domain × prevision source), roughly 32 points, each cell's rolled-up mean L (raw arm) against that cell's booked accuracy, and separately against calibration error (Brier raw − Brier calibrated, or fitted T). Spearman, not Pearson (`gemma-4-12B-it` is a plausible outlier that would distort a linear fit). This *records*, not *explains*, the `gemma-4-12B-it` split (chance-level linear probe in [08](08-probe-previsions.md), comfortably-passing elicited previsions in [10](10-elicited-previsions.md)): report the coefficient and label the point on the scatter, but state the split directly in prose rather than leaning on the correlation to make the case from one point.

**Reference points:** both a reference line on every plot and a compact reference block (ceiling/floor as two extra rows) on every table, for the random-prevision ceiling (mean L 0.162 pairs / 0.269 conjunctions) and the label floor (0, except the two Nile families).

**Ticket structure:** spin off a separate implementation ticket ([Implement the aggregation and comparison analysis](12-aggregation-implementation.md)), matching the pattern of every earlier methodology→implementation pair. The bootstrap alone is real, non-trivial code worth its own tests; the write-up ticket stays narrative-only and blocks on a concrete "these tables/figures exist" deliverable.

`Constituent pair` was added to `CONTEXT.md` to name the negation-pair ↔ conjunction-family relationship precisely. The bootstrap procedure itself is methodology, not domain vocabulary, so it stays on this ticket rather than in the glossary.
