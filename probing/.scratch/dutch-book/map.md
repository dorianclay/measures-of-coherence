# Map: Rate of loss to a Dutch book

Label: wayfinder:map

## Destination

Implemented and run experiments that measure the **rate of loss** of an LLM's **previsions** over **event families** built from the logically structured datasets (`neg_facts`, `neg_companies`, `conj_neg_facts`, `conj_neg_companies`), comparing **probe previsions** against **elicited previsions**, with controls, for a small set of open-weights models. Results in hand, not just a spec.

## Notes

- **Execution override:** this map carries execution. Unlike default wayfinding, tickets may build code and run jobs; the map is done when results exist.
- Domain: probabilistic coherence / Dutch books. Terms are in `CONTEXT.md` (prevision, prevision source, bookie, event family, atom, rate of loss); use them.
- Sources: `references/` holds the thesis draft (*AI: Agency and Representation*, §4.1.2), Andrews (2026) §2, and Schervish, Seidenfeld & Kadane (1998). `references/` is untracked and private; if it's missing on this machine, ask the author to copy it over.
- **Resuming** (the work moved from the author's laptop to the cluster on 2026-09-30):
  - **Experiment outputs exist only on the cluster checkout.** They are gitignored because they are too large to push, and git-lfs isn't available on this fork. Read them in place: `activations/`, `results/dutch_book/previsions/`, `results/dutch_book/probe_rates/` and `results/dutch_book/elicited_rates/`. The small `probe_metrics/` and `elicitation_metrics/` CSVs can be committed.
  - Research findings and the elicitation prototype are on the branches `research/rate-of-loss-formulation`, `research/calibrating-probe-previsions` and `prototype/elicitation-method`. Read them with `git show <branch>:<path>`.
  - The formulation and calibration research reads the PDFs in `references/`.
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
- [How do we get probe previsions free of train/test leakage?](issues/04-probe-leakage-protocol.md) — Domain swap: probes trained on the facts domain book the companies families, and vice versa. The primary probe is linear, with the MLP and atomic-only probes as secondary arms. Rates are computed per seed and never on averaged previsions. The headline layer is chosen within the training domain, and 20% of the training domain's pairs are held out for calibration.
- [How do we elicit previsions from model behavior?](issues/05-elicitation-method.md) — Next-token True/False logprobs are primary, with each statement in its own prompt. Stated 0–100 probabilities are a secondary arm for instruct models (they came out near-binary and flip on negation). Models need elicited accuracy ≥ 0.65 to enter the comparison: the small base models were at chance, while Qwen2.5-1.5B-Instruct reached 0.80. Paraphrase arm and calibration mirror the probe side.
- [Which models do the experiments run on?](issues/06-model-list.md) — Qwen2.5 at 1.5B, 7B and 14B plus Gemma 4 12B (the best current open family with base checkpoints; ungated), each as a base + instruct pair on one L40. The legacy small models are dropped, and Qwen2.5-32B is optional. Gemma 4 needs a multimodal loader and thinking mode off.
- [Implement and validate the rate-of-loss solver](issues/07-rate-of-loss-solver.md) — `DutchBook.py` is covered by 10 tests against independent oracles and fixes the previsions-CSV interface. Label previsions give 0 except on the two Nile families. The random-prevision ceilings are mean L 0.162 on negation pairs and 0.269 on conjunction families.
- [Produce probe previsions under the domain-swap protocol](issues/08-probe-previsions.md) — Cluster runs finished clean for all 8 models. Headline booked accuracies run 0.59–0.74 (companies) and 0.62–0.70 (facts) for Qwen2.5/Gemma-base; `gemma-4-12B-it`'s linear probe never clears chance at any layer (flagged for the write-up, not a pipeline bug — activations checked clean).
- [Produce elicited previsions for every family statement](issues/10-elicited-previsions.md) — Cluster runs finished clean for all 8 models; every method/template/domain clears the 0.65 competence bar (0.75–0.94), including `gemma-4-12B-it`, which recovers factuality signal its linear probe couldn't — pair with the probe-previsions finding in the write-up.

## Not yet specified

- **Merged algebras.** Statements shared across conjunctions could tie families into larger algebras with a single LP. Worth revisiting only if the per-family results are interesting.

## Out of scope

- Closed/API models: they have no activations, so no probe previsions.
- Generating negations/conjunctions for the atomic datasets (`cities`, `animals`, `elements`, …): the four structured datasets already cover two domains.
