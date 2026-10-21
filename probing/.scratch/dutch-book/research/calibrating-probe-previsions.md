# Should probe previsions be calibrated before booking?

Research for `.scratch/dutch-book/issues/02-calibrating-probe-previsions.md`. Terms follow `CONTEXT.md`: prevision, probe prevision, event family, atom, rate of loss. "Rate of loss" here means Andrews' L(p): the gambler's guaranteed profit per unit of total stake, with Σ|bᵢ| ≤ 1.

## TL;DR

- **Calibration and coherence are separate properties.** Every source says so. Calibration needs outcome labels; coherence does not. Neither implies the other. Constant or uninformative previsions are trivially coherent.
- **Rate of loss is not invariant to monotone recalibration.** In practice, calibration mostly changes L through *sharpness*. One shared map is applied to A, ¬A, B, ¬B and A∧¬B alike. For our two event families:
  - The only monotone maps that drive L to **exactly** 0 on generic probe outputs are ones that collapse (part of) the range to 1/2. This is the degenerate case.
  - **Strictly increasing** maps can drive L **arbitrarily close** to 0 while keeping accuracy and AUC unchanged. The affine shrink f(x) = s·x + (1−s)/2 multiplies L by at most s.
  - Temperature scaling spans the range from L ≈ 0 (T → ∞) to a scale-free "hard-label" limit (T → 0). It can **flip model rankings**.
- **Recommendation for the control arm:**
  - Use **temperature scaling on the probe logit, with no bias term**. Fit it by log-loss per (model, layer, probe) on a calibration split that is **disjoint at the family level** from every booked family and from the probe's training data. Its statement mix should match what is booked (affirmatives + negations, and conjunctions for the conjunction experiment).
  - Always report the fitted T, accuracy and Brier next to L.
  - Also report the **T → 0 hard-label rate** as a calibration-free companion.
  - Treat L on calibrated previsions as a label-using control, never as the headline.

## 1. What the literature says

### Andrews (2026) §2 and §6
- **Definition (§2, eq. 1).** L(p) = max_b min_j Σᵢ bᵢ(1_{Eᵢ}(ωⱼ) − p(Eᵢ)) subject to Σ|bᵢ| ≤ 1, and L(p) = 0 iff p is coherent (local copy `references/Andrews - 2026 …pdf`, §2 "The penalty").
- **Calibration vs. coherence (§6).** "Calibration methods (Guo et al., 2017) check whether predicted probabilities match empirical frequencies, requiring ground truth outcomes. De Finetti coherence checks internal consistency without ground truth. The two are complementary: a model can be well-calibrated on many events but incoherent, or coherent but poorly calibrated."
- **Coherence is not good behaviour (§1).** Coherence "is not sufficient for good behavior … both could be terrible."
- **Implication for us.** A calibrated-probe arm puts label information back into a measure whose selling point is being label-free. That is fine for a control, but it changes what the number means.

### Andrews & Sarkar (2026), "Dutch Books for Language Models"
[arXiv:2609.02797](https://arxiv.org/html/2609.02797). This paper applies the same LP to LLM forecasts.
- It says "coherence is not accuracy: a forecaster who assigns probability one to a single well-defined outcome … is coherent no matter what events we consider." It also notes that constant forecasts are trivially coherent.
- It reports that arbitrage profit varies about 100× across conditions while Brier scores sit in a much narrower range. So L and calibration/accuracy measures move largely independently.

### Paleka et al. (2025), "Consistency Checks for Language Model Forecasters" (ICLR 2025)
[arXiv:2412.18544](https://arxiv.org/html/2412.18544)
- **Checks.** They define Negation (F(P)+F(¬P)=1), And, Or, AndOr, But, Cond and others. Violations are scored with an arbitrage metric: log-scoring-rule based, not Andrews' linear stakes. For example, Negation gives 𝒱 = −2 log(√(F(x)(1−F(¬x))) + √((1−F(x))F(¬x))).
- **Trivial forecasters.** They state it directly: "Simply forecasting 50% probability for all questions will pass Paraphrase, ExpEvidence and Negation."
- **Post-hoc enforcement.** Their ArbitrageForecaster, which enforces consistency after the fact, "improves consistency on checks that we optimize against, but this improvement does not generalize to other held-out consistency checks, nor does it improve the actual forecasting performance."
- **Consistency vs. Brier.** They find consistency correlates with ground-truth Brier score *across forecasters*. That is a correlational finding, not a mechanism.
- **Calibration.** They do not study calibration or temperature scaling.

### Zhu & Griffiths (2024), "Incoherent Probability Judgments in LLMs"
[arXiv:2401.16646](https://arxiv.org/abs/2401.16646), CogSci 2024
- **Method and findings.** They test identities such as Z₁ = P(A)+P(B)−P(A∧B)−P(A∨B) on GPT-3.5/4 and LLaMA-2. They find human-like systematic deviations, which they explain with a Bayesian Sampler: judgments are shrunk toward a symmetric Beta(β,β) prior, i.e. toward 1/2.
- **Accuracy vs. coherence.** "accurate probability judgments … are also coherent. However, the reverse is not always true." They also cite Leitgeb & Pettigrew (2010): any incoherent set is dominated in accuracy by some coherent set.
- **Why this matters here.** A symmetric shrinkage toward 1/2 is exactly the kind of monotone map discussed in §2. It keeps P(A)+P(¬A)=1 intact, but it biases identities with unequal numbers of positive and negative terms. The same mechanism applies to our conjunction families.

### Probe-calibration work
- **Burns et al. (2022), CCS** ([arXiv:2212.03827](https://arxiv.org/html/2212.03827), ICLR 2023). The loss is L_consistency = [p(x⁺) − (1 − p(x⁻))]² plus L_confidence = min{p(x⁺), p(x⁻)}².
  - The consistency term is exactly the negation-pair coherence condition. It is trained *into* the probe rather than applied as post-hoc calibration.
  - The confidence term exists because the consistency term alone is solved by p ≡ 1/2. This is the same degeneracy as in §2.
  - Inference averages p̃ = ½(p(x⁺) + 1 − p(x⁻)), which is coherent on the pair by construction.
  - The paper does not calibrate CCS outputs. Its only "calibrated" baseline is a threshold shift for zero-shot prompting.
- **Marks & Tegmark (2023), "The Geometry of Truth"** ([arXiv:2310.06824](https://arxiv.org/html/2310.06824)).
  - Probes are LR or mass-mean p_mm(x) = σ(θᵀΣ⁻¹x).
  - The paper does not discuss calibration.
  - It finds that probes trained on affirmatives (`cities`) transfer poorly to negations (`neg_cities`), and that "training on statements and their opposites improves generalization." This negation-blindness is the main source of large L on negation pairs, and calibration cannot fix it (§2, Case C).
  - The thesis draft makes the same observation for I2 ("Poor transfer to negated statements without retraining specifically on negation").
- **Temperature and Platt scaling** (Guo et al. 2017, [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)). Temperature scaling is "a single-parameter variant of Platt Scaling" that does not change the classifier's decisions. Platt scaling adds a bias: σ(a·z + c). Isotonic regression is a non-parametric monotone fit.

**Bottom line for Q1.** No source claims that calibration improves coherence. Several note that trivially uninformative previsions are coherent. The only probe method that targets coherence (CCS) does it in the training loss, not by recalibration.

## 2. Can calibration alone drive the rate of loss to zero? (analytic)

**Setup.** A single monotone map f : [0,1] → [0,1] is applied to every raw probe output q in the family, the negated statements included.

### Two facts used throughout (checked numerically in §4, block 1)

1. **Closed form for negation pairs.** For {A, ¬A}, L(p) = |1 − p_A − p_¬A| / 2. The stakes b = ±(½, ½) achieve this.
2. **Duality.** By LP duality (a minimax over the ℓ₁-ball and the simplex), L(p) = min_{q ∈ Δ(atoms)} ‖Mq − p‖_∞. Here M is the event-by-atom indicator matrix. So **L is the ℓ∞ distance from p to the coherent set**, and it is therefore **convex in p**. This matches Schervish, Seidenfeld & Kadane (1998), where the rate depends on the chosen normalization (escrow).

### Consequences

**(a) Zero is reachable, but only degenerately.** The constant vector ½ is coherent for both of our families. For the conjunction family, take P(¬A∧B) = ½ and P(¬A∧¬B) = ½, with every other atom at 0. So f ≡ ½ gives L = 0 everywhere. Other constants c ≠ ½ give L = |c − ½| on the conjunction family.

**(b) Exact zero generically forces flattening.** Suppose two booked negation pairs (x₁, y₁) and (x₂, y₂) are *comonotone*: x₁ < x₂ and y₁ < y₂. A probe that does not track negation produces this constantly.
- Coherence needs f(x₁) + f(y₁) = 1 = f(x₂) + f(y₂).
- Monotonicity then forces f(x₁) = f(x₂) and f(y₁) = f(y₂).
- Chains of such pairs flatten f over whole intervals.
- So a **strictly** increasing f can zero L on every pair only if the raw pairs are exactly antitone and admit a map with f(y) = 1 − f(x). That is a measure-zero condition on real probe outputs.
- The conjunction family adds a second equality (the B pair) and the Fréchet bounds max(0, p_A + p_¬B − 1) ≤ p_{A∧¬B} ≤ min(p_A, p_¬B). The conclusion is the same.

**(c) Arbitrarily close to zero is cheap.** Let s ∈ (0,1]. The affine shrink f_s(x) = s·x + (1−s)/2 is strictly increasing, so it preserves accuracy, AUC and every ranking of statements. By convexity of L and L(½) = 0, L(f_s(p)) ≤ s·L(p), with equality on negation pairs. So **for any target ε > 0, a strictly monotone, decision-preserving map brings L below ε.**
- Temperature scaling behaves like this as T → ∞: the logit is divided by T and the previsions go to ½.
- A log-loss fit chooses large T exactly when the probe is uninformative.

**(d) Sharpening has a scale-free limit.** As T → 0, previsions become hard labels 1[q > ½].
- On negation pairs the limit is L = ½ · 1[both statements land on the same side of ½]. The mean over families is ½ × (fraction of pairs classified logically inconsistently).
- On conjunction families the limit is L of a 0/1 vector: 0 iff the hard labels form an atom's indicator column, otherwise positive.
- This limit **does not depend on any calibration choice**, which makes it a good companion metric.

**(e) Model rankings can flip.** L(T) is not monotone in T: it is 0 at T = ∞, it is the hard-label rate at T = 0, and it can peak in between. Per-model calibration picks a different T for each model, so it moves each model to a different point on its curve, and the ordering can change. §4, block 6 shows a constructed flip between two probes of similar accuracy (0.84 vs 0.85): raw L is X 0.073 < Y 0.124, but after per-model temperature fitting it is X 0.132 > Y 0.117.

**(f) Bias terms.**
- Temperature scaling without a bias is symmetric about ½: f(1−x) = 1 − f(x). So pairs that were already coherent (q_¬A = 1 − q_A) stay coherent.
- A Platt bias c ≠ 0 shifts both p_A and p_¬A the same way. This adds a systematic term to p_A + p_¬A − 1. The term can cancel an existing probe bias (it did in the simulation) or add one.
- Isotonic regression is neither symmetric nor smooth, and its steps create ties.

**Bottom line for Q2.**
- Calibration cannot make generic probe previsions *exactly* coherent without collapsing them toward ½.
- It can move L almost anywhere between ≈0 and the hard-label rate, and it can reorder models.
- So "rate of loss after calibration" measures incoherence *at a sharpness chosen by a label-fitted calibrator*, not a property of the probe alone.

## 3. Which calibration method, fitted on what?

### Method: temperature scaling on the probe's pre-sigmoid logit, no bias
- **One parameter**, so there is minimal room to overfit and minimal extra label use.
- **Decision-preserving**: accuracy and AUC are unchanged, so any change in L is purely a change of scale.
- **Symmetric about ½**: it cannot inject a shared shift into p_A + p_¬A.
- **Standard**: it is the usual post-hoc method (Guo et al. 2017).
- **Implementation note**: `TrainProbes.py` currently trains a small Keras MLP with a sigmoid output, not the linear σ(wᵀx+b) of thesis eq. 4.3. Apply T to logit(q), or to the final-layer pre-activation.
- **Secondary arms (optional)**: run Platt and isotonic as sensitivity checks only. Platt's bias and isotonic's steps confound "calibration" with a shift or a quantization of L.

### Data: grouped, family-disjoint, distribution-matched
- **Family-level disjointness.** Fitting the calibrator is supervised use of labels. The standing rule ("probe previsions must come from probes that never saw any statement in the family being booked") must therefore cover the calibrator too. Split by **family ID**: a negation pair, or a conjunction family together with its component statements. Never split by individual statement.
- **Disjoint from probe training too.** Use a three-way family-grouped split (probe-train / calibrate / book), or K-fold cross-fitting by family: fit the probe and T on K−1 folds and book the held-out fold. Otherwise T is fitted on over-confident in-sample outputs.
- **Same statement mix as the booked families.**
  - For the negation experiment, calibrate on affirmatives *and* negations from other families of `neg_facts` / `neg_companies`.
  - For the conjunction experiment, also include conjunction statements from other conjunction families.
  - Calibrating on affirmatives only would set T for a distribution the booked families do not have.
- **Per (model, layer, probe).** Each probe gets its own T.

### Reporting (so the control cannot be misread)
- **Report L_raw, L_T̂ and L_{T→0}** (the hard-label rate) side by side, together with T̂, accuracy, Brier/ECE, and the fraction of families with inconsistent hard labels.
- **Flag degenerate fits.**
  - When T̂ is very large, the probe is uninformative on that statement mix. In the simulation, a fully negation-blind probe gave T̂ ≈ 10¹⁵, L = 0.000 and accuracy 0.50. A low L there is not evidence of coherent beliefs.
  - As a concrete rule, flag cells where calibrated accuracy is within noise of 0.5, or where T̂ exceeds a cap (e.g. 20).
- **Optional:** plot L(T) curves per model. This shows whether a ranking is stable across the sharpness range, which addresses the flip in §2(e).

## 4. Numerical sanity check

**Files.**
- Script: `.scratch/dutch-book/research/calibrating-probe-previsions_check.py` (scipy `linprog`/HiGHS; sklearn for isotonic).
- Run with: `uv run --with scikit-learn python .scratch/dutch-book/research/calibrating-probe-previsions_check.py`
- The data is synthetic: probe logits are ±sep plus Gaussian noise. `blind` sets the fraction of the ¬A signal that ignores the negation, and there is a +0.5 bias on ¬A. Conjunction families are {A, ¬A, B, ¬B, A∧¬B} over 4 atoms.

### Block 1: closed form and duality
- **Negation pairs.** On random p, the LP value, |1−p_A−p_¬A|/2 and min_q‖Mq−p‖_∞ agree to 4 d.p. (e.g. p = (0.813, 0.913) gives 0.3630 for all three).
- **Conjunction families.** The LP value equals the ℓ∞ distance on random p.
- **Coherent inputs.** Label vectors, previsions built from atom probabilities, and constant ½ all give L = 0.
- **Other constants.** On the conjunction family, constants 0.3 / 0.4 / 0.6 / 0.7 give L = 0.2 / 0.1 / 0.1 / 0.2.

### Block 2: temperature sweep on negation pairs (sep = 2, no blindness)
L is non-monotone in T:

| T | 0.05 | 0.25 | 0.5 | 1 (raw) | 2 | 4 | 16 | 1000 |
|---|---|---|---|---|---|---|---|---|
| mean L | 0.023 | 0.035 | 0.053 | 0.067 | 0.056 | 0.033 | 0.009 | 0.000 |

The T → 0 limit is 0.0213. This equals ½ × the fraction of pairs whose hard labels are inconsistent (0.0213), as predicted in §2(d).

### Block 3: calibrators fitted on 2000 held-out families, evaluated on 400 other families
Each cell is L / accuracy / Brier.

| negation-blindness | raw | temperature | Platt | isotonic | const ½ |
|---|---|---|---|---|---|
| 0.0 | .073 / .970 / .046 | .042 / .970 / .022 (T̂=0.27) | .035 / .978 / .017 | .035 / .981 / .017 | 0 / .50 / .25 |
| 0.5 | .187 / .750 / .170 | .178 / .750 / .169 (T̂=1.10) | .180 / .735 / .167 | .182 / .750 / .165 | 0 / .50 / .25 |
| 1.0 | .338 / .511 / .376 | **.000** / .506 / .250 (T̂≈10¹⁵) | **.000** / .500 / .250 | .037 / .544 / .247 | 0 / .50 / .25 |

Three readings:
- **Well-behaved probe:** calibration sharpens it (T̂ < 1) and roughly halves L.
- **Partially negation-blind probe:** calibration barely moves L. The incoherence is structural.
- **Fully negation-blind probe:** the log-loss fit collapses to ½, so **L → 0 with chance accuracy**. This is the degenerate case the control must flag.

### Block 4: conjunction families (sep = 1.5)

| T | 0.05 | 0.5 | 1 | 2 | 8 | 1000 |
|---|---|---|---|---|---|---|
| mean L | 0.178 | 0.184 | 0.159 | 0.107 | 0.031 | 0.000 |

With fitted calibrators, L goes from 0.159 (raw) to 0.186 (temperature, T̂ = 0.38), 0.163 (Platt) and 0.163 (isotonic). Calibration can *increase* L when it sharpens.

### Block 5: the best possible shared monotone map
- **Method.** A single LP over the values of f at every raw output, minimizing the mean L over 60 negation pairs or 40 conjunction families, subject to f(x′) − f(x) ≥ slope·(x′ − x).
- **Raw values.** L_neg = 0.113 and L_conj = 0.169.

| minimum slope | 0 | 0.05 | 0.2 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| best L_neg | 0.000 | 0.006 | 0.022 | 0.055 | 0.112 |
| best L_conj | 0.000 | 0.007 | 0.028 | 0.069 | 0.150 |

- **Reading.** Zero is reached only when flat segments are allowed. The best achievable L grows roughly linearly in the slope bound, as §2(b)–(c) predict.

### Block 6: ranking flip
- **Model X** is soft and under-confident (sep 0.6, noise 0.45) and never negation-blind. **Model Y** is sharp and noisy (sep 3, noise 3) with 10% negation-blindness. Their accuracies are similar (0.844 vs 0.854).
- **Raw:** L_X = 0.073 < L_Y = 0.124.
- **After per-model temperature fitting:** L_X = 0.132 (T̂ = 0.27, sharpened) > L_Y = 0.117 (T̂ = 1.65, softened).
- So the ranking flips purely because calibration moves each model to a different sharpness.

## Sources
- Andrews, I. (2026). *Revealed Rationality: Label-Free Evaluation and Regularization from Representation Theorems.* §2 (eq. 1), §6. Local: `references/Andrews - 2026 - Revealed Rationality….pdf`.
- Andrews, I. & Sarkar, S. (2026). *Dutch Books for Language Models.* https://arxiv.org/abs/2609.02797
- Schervish, M. J., Seidenfeld, T. & Kadane, J. B. (1998). *Two Measures of Incoherence: How Not to Gamble If You Must.* Local: `references/Schervish et al. - 1998 ….pdf`. See §1 on normalization/escrow.
- Paleka, D. et al. (2025). *Consistency Checks for Language Model Forecasters.* ICLR 2025. https://arxiv.org/abs/2412.18544
- Zhu, J.-Q. & Griffiths, T. L. (2024). *Incoherent Probability Judgments in Large Language Models.* https://arxiv.org/abs/2401.16646
- Burns, C., Ye, H., Klein, D. & Steinhardt, J. (2022). *Discovering Latent Knowledge in Language Models Without Supervision.* https://arxiv.org/abs/2212.03827
- Marks, S. & Tegmark, M. (2023). *The Geometry of Truth.* https://arxiv.org/abs/2310.06824
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML. https://arxiv.org/abs/1706.04599
- Thesis draft, *AI: Agency and Representation*: §3 (I2/I3 method notes) and §4.1.1–4.1.2. Local: `references/AI__Agency__and_Representation.pdf`.
- Repo: `TrainProbes.py` (Keras MLP probe with a sigmoid output) and `datasets/neg_*_true_false.csv` (consecutive affirmative/negation rows).
