# Should probe previsions be calibrated before booking?

Type: research
Status: resolved
Blocked by:

## Question

Probe outputs σ(wᵀx+b) are trained for classification accuracy, not calibration. A per-statement monotone recalibration (Platt/temperature scaling, isotonic) changes previsions and so changes the rate of loss, possibly by a lot. Settle:

1. What does the literature (Andrews 2026 §6 on calibration vs. coherence, Paleka et al. 2025, Zhu & Griffiths 2024, and probe-calibration work such as Marks & Tegmark and Burns et al.'s CCS) say about the relationship between calibration and coherence, and about calibrating probe outputs?
2. Can calibration alone drive rate of loss to zero on negation pairs or conjunction families, or change the ranking between models? Answer analytically where possible.
3. Which calibration method (if any) should the "with calibration" control use, and on what held-out data should it be fitted so it doesn't leak family statements?

Output: a recommendation for the calibration control arm.

## Answer

Findings: branch `research/calibrating-probe-previsions`, file `.scratch/dutch-book/research/calibrating-probe-previsions.md`. The script `calibrating-probe-previsions_check.py` reproduces every number.

1. **Calibration and coherence are separate properties.** Andrews 2026 §6, Paleka et al. 2025, Zhu & Griffiths 2024, and Andrews & Sarkar 2026 all make this point. Constant ½ forecasts pass negation and conjunction coherence checks, and CCS needs its confidence term for exactly that reason. Neither CCS nor Marks & Tegmark calibrates probe outputs.
2. **Calibration can zero the rate of loss only degenerately, but it can move the rate almost anywhere.**
   - Rate of loss is the L∞ distance to the coherent set, so it is convex. All-½ previsions are coherent, so the map f ≡ ½ gives exactly 0.
   - A strictly increasing affine shrink toward ½ scales the rate by its factor s without changing accuracy or AUC.
   - The rate is not monotone in temperature T. As T → 0 it tends to a hard-label limit; on negation pairs that is half the fraction of pairs whose hard labels contradict each other.
   - A simulated pair of probes swapped rank after calibration.
   - A log-loss calibrator on a negation-blind probe collapses to ½, giving a rate of 0 at chance accuracy.
3. **Control arm.**
   - **Method:** temperature scaling of the probe logit with no bias term. It is symmetric about ½, so it can't shift p(A)+p(¬A).
   - **Fitting:** by log-loss, per (model, layer, probe).
   - **Data:** a calibration split that is family-disjoint from both the booked families and the probe's training data, with the same statement mix as the families being booked.
   - **Reporting:** raw, calibrated and T→0 rates side by side, with the fitted T, accuracy and Brier score. Flag very large T and near-chance accuracy.
   - Platt and isotonic scaling are optional sensitivity checks only.
   - The calibrated arm is a **labelled control, not the headline**.
   - Caveat: `TrainProbes.py` probes are a Keras MLP with a sigmoid output, not linear, so T applies to logit(q) or to the final pre-activation.
