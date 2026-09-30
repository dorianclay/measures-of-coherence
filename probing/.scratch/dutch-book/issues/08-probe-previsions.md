# Produce probe previsions under the domain-swap protocol

Type: task (AFK)
Status: open
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
