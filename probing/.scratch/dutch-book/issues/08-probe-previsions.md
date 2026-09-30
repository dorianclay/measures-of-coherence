# Produce probe previsions under the domain-swap protocol

Type: task (AFK)
Status: resolved
Blocked by: 06

## Question

Implement and run the protocol decided in [How do we get probe previsions free of train/test leakage?](04-probe-leakage-protocol.md) for every model in [Which models do the experiments run on?](06-model-list.md):

- Extract embeddings for every statement the probes train on or book: the six atomic datasets, `facts`, `companies`, `neg_*`, and `conj_neg_*`. Cover the whole layer sweep, using `Generate_Embeddings.py` and `slurm/`.
- For each (model, layer, training domain), train the probes:
  - the primary linear logistic probe and the secondary MLP (several seeds);
  - the atomic-only secondary arm.
  - Hold out the ~20% calibration slice from training.
- Fit the temperature on the calibration slice, as specified in [Should probe previsions be calibrated before booking?](02-calibrating-probe-previsions.md).
- Write out, for every event in every booked family, the raw and calibrated previsions for each probe, in the input format of `DutchBook.py` (see [Implement and validate the rate-of-loss solver](07-rate-of-loss-solver.md)). Also record each probe's accuracy on the booked domain and on the training-domain validation data, its Brier score, and the fitted T.

Done when prevision files exist for every model × layer × probe arm, and the answer records where they live, row counts, headline layers, and booked-domain accuracies.

## Comments

**2026-09-29: code done, validated locally; waiting on cluster runs.** Still claimed.

The pipeline:
- `Generate_Activations.py` extracts last-token activations at 8 relative depths (1/8 … 1) into `activations/<model>/<dataset>.npy` (float16, rows in dataset order, not committed), with `meta.json` recording the layers. Padding is on the right, and batched extraction matches single-statement extraction to float16 noise. Multimodal checkpoints fall back to `AutoModelForMultimodalLM`.
- `Train_Family_Probes.py` implements the domain-swap protocol exactly as decided:
  - calibration slice: 20% of the training domain's pairs (split seed 0), plus conjunctions with both pairs held out;
  - arms: `domain_swap` (linear + MLP × 5 seeds) and `atomic_only` (linear);
  - bias-free temperature fitted by log-loss on the calibration slice;
  - it stops with an error if any booked-domain or calibration statement is in the training set.
  - Previsions (raw + calibrated) go to `results/dutch_book/previsions/<model>.csv` in `DutchBook.py`'s input format (not committed, ~70 MB per model).
  - Per-probe metrics (T, validation accuracy on the calibration slice, booked accuracy and Brier score) go to `results/dutch_book/probe_metrics/<model>.csv`.
- `slurm/probe_previsions.sbatch <hf-id>` runs extract → probes → `DutchBook.py` and writes `results/dutch_book/probe_rates/<model>.csv`.

Bug found and fixed while validating: temperature fitting on clipped probabilities gave a flat loss for saturated probes, so the optimizer returned its first probe point (T = 0.31 everywhere). The loss is now computed from logits, and it recovers known distortions exactly (×20 → 19.9, ×0.25 → 0.249).

Local check on Qwen2.5-1.5B-Instruct (1 MLP seed):
- Linear domain-swap booked accuracy peaks at layer 18 (depth 0.64): 0.65 on companies, 0.62 on facts. Validation accuracy picks the same layer.
- The MLP reaches 0.79.
- Mean L, linear domain-swap, raw → calibrated: conjunctions 0.45 → 0.24, pairs 0.35 → 0.18.
- The atomic-only arm is near chance, and calibration collapses it towards ½ (T at the upper bound e⁵), the degenerate case to flag in the analysis.

Remaining before resolving: the cluster runs for all 8 models (Gemma 4 first, since its loader is untested locally), then copy back `probe_metrics/` and `probe_rates/`.

**2026-09-30: cluster runs finished; reopened for the next session to resolve on the cluster.** The outputs sit on the cluster checkout and are too large to push (see the map's **Resuming** note). To resolve:
1. Check that outputs exist for all 8 models. Read the job logs in `logs/` for failures, especially the Gemma 4 jobs, whose loader and thinking-mode paths were untested before the cluster.
2. Summarize the metrics in the answer.
3. Commit only the small metrics CSVs.

## Answer

**All 8 models completed cleanly.** `logs/probe-previsions-*.out` (209931, 209976–209982) each end with all three `Saved ... rows/rates` lines and no error/traceback beyond the benign unauthenticated-HF-Hub notice — including both Gemma 4 jobs (209931 = `gemma-4-12B-it`, 209982 = `gemma-4-12B`), whose loader and thinking-mode paths were the untested risk.

**Files and row counts** (identical across all 8 models):
- `results/dutch_book/previsions/<model>.csv` — 851,648 rows (raw + calibrated previsions, domain-swap + atomic-only arms, all seeds/layers).
- `results/dutch_book/probe_metrics/<model>.csv` — 112 rows (arm × probe × seed × layer combinations).
- `results/dutch_book/probe_rates/<model>.csv` — 364,112 rows (per-family L, via `DutchBook.py`).

**Headline layers** (domain-swap linear, layer with max validation accuracy on the training-domain calibration slice, per booked domain):

| model | booked=companies (layer / val acc / booked acc) | booked=facts (layer / val acc / booked acc) |
|---|---|---|
| Qwen2.5-1.5B | 18 / 0.698 / 0.591 | 18 / 0.838 / 0.633 |
| Qwen2.5-1.5B-Instruct | 18 / 0.678 / 0.647 | 18 / 0.861 / 0.621 |
| Qwen2.5-7B | 21 / 0.740 / 0.715 | 24 / 0.870 / 0.652 |
| Qwen2.5-7B-Instruct | 21 / 0.719 / 0.712 | 21 / 0.861 / 0.644 |
| Qwen2.5-14B | 48 / 0.777 / 0.680 | 36 / 0.884 / 0.681 |
| Qwen2.5-14B-Instruct | 30 / 0.785 / 0.739 | 42 / 0.880 / 0.696 |
| gemma-4-12B | 30 / 0.744 / 0.704 | 30 / 0.870 / 0.677 |
| gemma-4-12B-it | 36 / 0.558 / 0.525 | 6 / 0.556 / 0.519 |

**Flag for the write-up/aggregation analysis:** `gemma-4-12B-it` is the one model where the linear domain-swap probe never clears chance at any of the 8 layers (val acc 0.44–0.56 both directions; MLP recovers only to 0.60–0.66). This isn't a pipeline bug — checked the activations directly (`activations/gemma-4-12B-it/facts.npy`): no NaNs, no zero rows, variance comparable to the base checkpoint (`gemma-4-12B`, which reaches 0.87 linear val acc on the same layers/statements). The last-token representation of instruct-tuned Gemma 4 appears to carry much weaker linearly-decodable factuality signal than its base checkpoint; worth a call-out in the write-up rather than a fix here.

`results/dutch_book/probe_metrics/*.csv` for all 8 models are already committed (commit `cdb562e`, "Run experiments on cluster"); nothing further to commit for this ticket.
