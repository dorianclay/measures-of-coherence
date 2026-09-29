# Which models do the experiments run on?

Type: grilling
Status: open
Blocked by: 05

## Question

Fix the model list: open weights only, runnable on the cluster's L40s.

[How do we elicit previsions from model behavior?](05-elicitation-method.md) changed the picture. SmolLM2-135M and OPT-1.3b are at chance when elicited (accuracy 0.49 and 0.45), so they fail the ≥ 0.65 competence bar and can't enter the probe-vs-elicited comparison. Qwen2.5-1.5B-Instruct reached 0.80. Settle:

- Which instruct models are in the comparison.
- Whether to add a size sweep within one family (e.g. Qwen2.5 1.5B / 7B / 14B) to see whether coherence scales with size.
- Whether to pair each instruct model with its base model, to see what instruction tuning does to coherence.
- Whether the two small legacy models stay in the probe-only arm or are dropped.
- The cluster budget: memory at bf16 for the largest model, embedding extraction across the layer sweep, and the elicitation passes (primary + stated + 3 paraphrases).
