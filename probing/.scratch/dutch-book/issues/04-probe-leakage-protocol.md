# How do we get probe previsions free of train/test leakage?

Type: grilling
Status: open
Blocked by: 03

## Question

`neg_facts` reuses `facts` statements verbatim, and the existing probes were trained on `facts`/`companies`. A probe that saw A but not ¬A inflates the rate of loss artificially. [Build the event families from the structured datasets](03-build-event-families.md) measured the overlap: 542/1092 negation-pair statements appear verbatim in `facts`, and 496/999 in `companies`. Negation pairs recur across conjunction families, so a family-disjoint split must group at the level of the negation pair: every conjunction family that touches a pair goes in the same fold. Decide the protocol for producing probe previsions on every family statement from a probe that never saw any statement in that family:

- **Cross-domain**: train on one domain (e.g. atomic datasets such as `cities`, `animals`, `elements`), then book `facts`/`companies` families.
- **Held-out family splits**: k-fold over families within the structured datasets, so every statement of a family lands in the same fold.
- Some mix of the two, and whether negated statements (and conjunctions) should appear in probe training at all, which affects whether the probe has "seen negation".

The calibration control ([Should probe previsions be calibrated before booking?](02-calibrating-probe-previsions.md)) needs a third split, family-disjoint from both probe training and booking, that contains negations (and conjunctions for the conjunction experiment). The protocol must provide it: a three-way split or cross-fitting by family.

Also decide whether to keep the existing probe architecture. `TrainProbes.py` trains a Keras **MLP**, not the linear σ(wᵀx+b) that thesis §4.1.1 defines. Should the experiments use a linear (logistic) probe to match the thesis, the MLP, or both?

Also decide which layer(s) and how many probe seeds (the pipeline trains `repeat_each` probes per setting): report per seed, or average previsions across seeds? Averaging changes coherence.
