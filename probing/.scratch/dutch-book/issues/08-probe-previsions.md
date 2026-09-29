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
- Write out, for every event in every booked family, the raw and calibrated previsions for each probe, keyed by family id and event index from `event_families/`. Also record each probe's accuracy on the booked domain and on the training-domain validation data, its Brier score, and the fitted T.

Done when prevision files exist for every model × layer × probe arm, and the answer records where they live, row counts, headline layers, and booked-domain accuracies.
