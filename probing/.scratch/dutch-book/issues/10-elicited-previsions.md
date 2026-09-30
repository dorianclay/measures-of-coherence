# Produce elicited previsions for every family statement

Type: task (AFK)
Status: open
Blocked by: 06

## Question

Implement and run the elicitation decided in [How do we elicit previsions from model behavior?](05-elicitation-method.md) for every model in [Which models do the experiments run on?](06-model-list.md). The throwaway prototype on branch `prototype/elicitation-method` shows the prompt and token handling.

- **Primary:** logprob previsions for every event in every family in `event_families/`, using the fixed template (chat template with the fixed system role for instruct models, 4-shot for base models), each statement in its own prompt.
- **Secondary:** stated probabilities, for instruct models only.
- **Secondary:** the 3-template paraphrase arm.
- For each model, measure elicited accuracy on the training-domain data and apply the ≥ 0.65 competence bar.
- Fit the temperature-scaling control on the same training-domain calibration slice the probes use (see [How do we get probe previsions free of train/test leakage?](04-probe-leakage-protocol.md)).
- Write out raw and calibrated previsions in the input format of `DutchBook.py` (see [Implement and validate the rate-of-loss solver](07-rate-of-loss-solver.md)), matching the probe previsions from [Produce probe previsions under the domain-swap protocol](08-probe-previsions.md), so the solver can consume both identically.

Done when prevision files exist for every model × method × template. The answer records where the files live, each model's accuracy and whether it passed the bar, the parse-failure rate for stated probabilities, and the mean mass on the answer tokens.

## Comments

**2026-09-30: code done, validated locally; waiting on cluster runs.** Still claimed.

- `Elicit_Previsions.py <hf-id>` elicits every distinct family statement once per method and template, each in its own prompt.
  - Methods: `logprob` over templates t0 (primary), t1 and t2 (paraphrases), plus `stated` on t0 for instruct models.
  - For each booked domain it fits the bias-free temperature on logit(prevision) over the **other** domain's calibration slice. That is the same split as `Train_Family_Probes.py`, so both prevision sources are calibrated on identical data.
  - The competence bar (≥ 0.65) is measured on that same slice.
  - Previsions (raw + calibrated) go to `results/dutch_book/previsions/elicited_<model>.csv` (not committed).
  - Metrics go to `results/dutch_book/elicitation_metrics/<model>.csv`: T, competence accuracy, pass/fail, booked accuracy, Brier score, answer-token mass and stated parse failures.
- `slurm/elicited_previsions.sbatch <hf-id>` runs elicitation → `DutchBook.py` and writes `results/dutch_book/elicited_rates/<model>.csv`.
- Instruct vs. base is decided from the model name (`-Instruct`, `-it`), overridable with `--instruct`. Checking whether a chat template exists is not enough, because **Qwen2.5 base checkpoints ship a chat template too**.
- Gemma 4's thinking mode is disabled via `enable_thinking=False`. Templates without a system role fold it into the user turn.
- Batched previsions match single-prompt ones to float16 noise (max difference 0.003).

Local run on Qwen2.5-1.5B-Instruct over all families (44 min on MPS):
- Every method and template passes the bar: competence accuracy 0.74–0.79, booked accuracy 0.76–0.80.
- Answer-token mass is ≥ 0.998, with 0 stated parse failures.
- Mean L, logprob t0, raw → calibrated: conjunctions 0.284 → 0.209, pairs 0.143 → 0.118. The paraphrase templates give 0.28–0.31 on conjunctions.
- These agree with the prototype's 20-family numbers (accuracy 0.80, L 0.294).

Remaining before resolving: the cluster runs for all 8 models (Gemma 4 first), then copy back `elicitation_metrics/` and `elicited_rates/`.

**2026-09-30: cluster runs finished; reopened for the next session to resolve on the cluster.** The outputs sit on the cluster checkout and are too large to push (see the map's **Resuming** note). To resolve:
1. Check that outputs exist for all 8 models. Read the job logs in `logs/` for failures, especially the Gemma 4 jobs, whose loader and thinking-mode paths were untested before the cluster.
2. Summarize the metrics in the answer.
3. Commit only the small metrics CSVs.
