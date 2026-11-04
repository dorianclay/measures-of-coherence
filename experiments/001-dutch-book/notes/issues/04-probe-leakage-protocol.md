# How do we get probe previsions free of train/test leakage?

Type: grilling
Status: resolved
Blocked by: 03

## Question

`neg_facts` reuses `facts` statements verbatim, and the existing probes were trained on `facts`/`companies`. A probe that saw A but not ¬A inflates the rate of loss artificially. [Build the event families from the structured datasets](03-build-event-families.md) measured the overlap: 542/1092 negation-pair statements appear verbatim in `facts`, and 496/999 in `companies`. Negation pairs recur across conjunction families, so a family-disjoint split must group at the level of the negation pair: every conjunction family that touches a pair goes in the same fold. Decide the protocol for producing probe previsions on every family statement from a probe that never saw any statement in that family:

- **Cross-domain**: train on one domain (e.g. atomic datasets such as `cities`, `animals`, `elements`), then book `facts`/`companies` families.
- **Held-out family splits**: k-fold over families within the structured datasets, so every statement of a family lands in the same fold.
- Some mix of the two, and whether negated statements (and conjunctions) should appear in probe training at all, which affects whether the probe has "seen negation".

The calibration control ([Should probe previsions be calibrated before booking?](02-calibrating-probe-previsions.md)) needs a third split, family-disjoint from both probe training and booking, that contains negations (and conjunctions for the conjunction experiment). The protocol must provide it: a three-way split or cross-fitting by family.

Also decide whether to keep the existing probe architecture. `TrainProbes.py` trains a Keras **MLP**, not the linear σ(wᵀx+b) that thesis §4.1.1 defines. Should the experiments use a linear (logistic) probe to match the thesis, the MLP, or both?

Also decide which layer(s) and how many probe seeds (the pipeline trains `repeat_each` probes per setting): report per seed, or average previsions across seeds? Averaging changes coherence.

## Answer

Facts that shaped the decision:
- Only `facts` and `companies` share statements with the families.
- The six atomic datasets are clean, but they contain almost no negation (0/10,000 in `cities`, 0/1,458 in `capitals`, 0/876 in `inventions`).
- Negation pairs linked through conjunctions form one giant component (439/547 facts pairs, 406/500 companies pairs). Within-domain family-disjoint folds are therefore impractical.

Decisions:
1. **Primary protocol: domain swap.** Train a probe on the facts domain (`facts`, `neg_facts`, `conj_neg_facts`, and all six atomic datasets) and book the companies families; then swap the domains. Each family is booked by a single probe that has seen negation and conjunction but no statement from the booked domain. Report probe accuracy on the booked domain next to every rate.
   - **Secondary arm: atomic-only probes.** Train on the six atomic datasets only and book both domains. This measures how much incoherence comes from negation-blindness.
   - Within-domain pair folds are rejected: k(k−1)/2 probes per setting isn't worth it.
2. **Architecture.** A linear logistic-regression probe, σ(wᵀx+b), is primary, matching thesis §4.1.1. The existing MLP from `TrainProbes.py` runs as a secondary arm.
3. **Seeds.** Never average previsions across probes: the rate of loss is convex, so an averaged bookie always looks more coherent. Compute the rate for each seed and report the mean ± sd across seeds; this only matters for the MLP.
4. **Layers.** Sweep layers. Choose the headline layer by validation accuracy within the training domain, never on booked families, and report the full sweep as a supplement.
5. **Calibration split.** Hold out about 20% of the training domain's negation pairs from probe training, plus the conjunction families whose pairs are both held out. Conjunction families with only one pair held out go to neither side. The calibration temperature is fitted on this held-out slice.
