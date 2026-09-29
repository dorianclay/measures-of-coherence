# Map: Rate of loss to a Dutch book

Label: wayfinder:map

## Destination

Implemented and run experiments that measure the **rate of loss** of an LLM's **previsions** over **event families** built from the logically structured datasets (`neg_facts`, `neg_companies`, `conj_neg_facts`, `conj_neg_companies`), comparing **probe previsions** against **elicited previsions**, with controls, for a small set of open-weights models. Results in hand, not just a spec.

## Notes

- **Execution override:** this map carries execution. Unlike default wayfinding, tickets may build code and run jobs; the map is done when results exist.
- Domain: probabilistic coherence / Dutch books. Terms are in `CONTEXT.md` (prevision, prevision source, bookie, event family, atom, rate of loss); use them.
- Sources: `references/` holds the thesis draft (*AI: Agency and Representation*, §4.1.2), Andrews (2026) §2, and Schervish, Seidenfeld & Kadane (1998). `references/` is untracked, so read it from the main checkout.
- Code style: flat root-level scripts with argparse + JSON config like the existing pipeline (e.g. `DutchBook.py`, `Elicit_Previsions.py`); LPs via `scipy.optimize.linprog`; GPU jobs as sbatch files in `slurm/`.
- Standing decisions from charting:
  - Two prevision sources, compared: probe outputs vs. elicited behavior. Probe previsions come first (cheap).
  - Probe previsions must come from probes that never saw any statement in the family being booked.
  - Event families: negation pairs {A, ¬A} and conjunction families {A, ¬A, B, ¬B, conjunction}, reported as separate experiments.
  - Controls: ground-truth labels as previsions (must give 0), random previsions (ceiling), probe previsions with and without calibration.
  - Open-weights models only.
- Skills: `/grilling` + `/domain-modeling` for HITL tickets; `/research` for research tickets; `/tdd` for the solver.

## Decisions so far

<!-- one line per closed ticket -->

- [Which rate-of-loss formulation(s) do we compute?](issues/01-rate-of-loss-formulation.md) — Andrews' L as the primary rate (it's the L∞ distance to coherence, in [0,1]), with SSK's ρ as a secondary column on conjunction families only. Never report ψ, and don't compare L across family shapes. The closed-form oracles are on `research/rate-of-loss-formulation`.
- [Should probe previsions be calibrated before booking?](issues/02-calibrating-probe-previsions.md) — Calibration can move the rate of loss almost anywhere and reorder models, so the calibrated arm is a labelled control, not the headline. It uses bias-free temperature scaling fitted on a family-disjoint split, reported next to the raw and T→0 (hard-label) rates.
- [Build the event families from the structured datasets](issues/03-build-event-families.md) — `Build_Event_Families.py` produced 547 + 500 negation pairs and 556 + 546 conjunction families (facts + companies). Two facts families have inconsistent labels because of the contested Nile fact. About half the pair statements also sit in the probes' current training data.

## Not yet specified

- **Aggregation and comparison analysis.** Negation-pair and conjunction-family rates are never pooled or compared with each other. Families share statements, so they aren't independent samples; any test needs to cluster or bootstrap by negation pair. Conjunction polarities are roughly balanced (~25% each), so results can be stratified by polarity. How per-family rates roll up (distribution, mean/median per dataset × model × source), the statistical test for probe vs. elicited on the same families, and whether rate of loss correlates with probe accuracy or with calibration error. Waits on the formulation and on how many families survive.
- **Running at scale.** Extracting embeddings for every family statement, per model and layer; the elicitation jobs; which layers to report. Waits on the model list and the leakage protocol.
- **Merged algebras.** Statements shared across conjunctions could tie families into larger algebras with a single LP. Worth revisiting only if the per-family results are interesting.
- **Results write-up.** Figures and tables that feed thesis §4.1.2, including the worked "Dutch book-ability" example the thesis leaves blank.

## Out of scope

- Closed/API models: they have no activations, so no probe previsions.
- Generating negations/conjunctions for the atomic datasets (`cities`, `animals`, `elements`, …): the four structured datasets already cover two domains.
