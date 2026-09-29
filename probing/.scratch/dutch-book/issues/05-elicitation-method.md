# How do we elicit previsions from model behavior?

Type: prototype
Status: resolved
Blocked by:

## Question

Choose how an **elicited prevision** is read from a model:

- **Token logprobs**: normalized P("True") / (P("True") + P("False")) after a judgment prompt. Works on base models.
- **Stated probability**: ask for a number from 0 to 100 and parse it. Needs an instruct model; output is coarse and often clustered.
- Prompt template(s), whether a fixed **role** is set per batch of queries (Andrews §2 recommends testing coherence within one role), and whether paraphrase variants count toward the same family.

Prototype: a throwaway script that runs both methods on ~20 families from a small model (SmolLM2-135M, plus a small instruct model), and prints the previsions side by side with their naive rate of loss. React to it together.

## Answer

Prototype: branch `prototype/elicitation-method` has `PROTOTYPE_elicitation.py` and its output at `.scratch/dutch-book/prototypes/elicitation-run-output.txt`. It ran on 20 conjunction families (10 per domain), 100 statements, each judged in isolation.

| model | method | accuracy | mean L | previsions |
|---|---|---|---|---|
| SmolLM2-135M (base) | logprob | 0.49 | 0.055 | 97% in [0.4, 0.6]: degenerate "≈½ is coherent" |
| OPT-1.3b (base) | logprob | 0.45 | 0.201 | all in [0.5, 0.9]: True bias, so p(A)+p(¬A) > 1 |
| Qwen2.5-1.5B-Instruct | logprob | 0.80 | 0.294 | mostly extreme; informative, and the conjunction is the main failure |
| Qwen2.5-1.5B-Instruct | stated 0–100 | 0.71 | 0.343 | 98% exactly 0 or ~100; negation flips (98 for "has 206 bones", 94 for "doesn't have") |

Mass on the True/False answer tokens was ≥ 0.99 for every model.

Decisions:
1. **Primary method: next-token logprobs**, P(True)/(P(True)+P(False)). **Stated probability** is a secondary arm for instruct models only, reported as near-binary.
2. **Competence bar.** A model enters the elicited arm (and the probe-vs-elicited comparison) only if its elicited accuracy on the training domain is ≥ 0.65. Accuracy is always reported next to L. Models below the bar may still appear in the probe arm.
3. **Prompts.**
   - Instruct models: chat template, one fixed system role, each statement in its own prompt so the model never sees other members of its family.
   - Base models: a 4-shot plain prompt with examples from the atomic datasets.
   - Secondary arm: 3 paraphrased templates. L is computed per template, never on averaged previsions.
4. **Calibration.** Elicited previsions are treated like probe previsions: raw is the headline, and the control applies bias-free temperature scaling to logit(p), fitted on the same training-domain calibration slice.
