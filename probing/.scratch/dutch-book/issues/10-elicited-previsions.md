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
- Write out raw and calibrated previsions keyed by family id and event index, in the same shape as the probe previsions from [Produce probe previsions under the domain-swap protocol](08-probe-previsions.md), so the solver can consume both identically.

Done when prevision files exist for every model × method × template. The answer records where the files live, each model's accuracy and whether it passed the bar, the parse-failure rate for stated probabilities, and the mean mass on the answer tokens.
