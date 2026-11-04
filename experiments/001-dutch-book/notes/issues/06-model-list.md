# Which models do the experiments run on?

Type: grilling
Status: resolved
Blocked by: 05

## Question

Fix the model list: open weights only, runnable on the cluster's L40s.

[How do we elicit previsions from model behavior?](05-elicitation-method.md) changed the picture. SmolLM2-135M and OPT-1.3b are at chance when elicited (accuracy 0.49 and 0.45), so they fail the ≥ 0.65 competence bar and can't enter the probe-vs-elicited comparison. Qwen2.5-1.5B-Instruct reached 0.80. Settle:

- Which instruct models are in the comparison.
- Whether to add a size sweep within one family (e.g. Qwen2.5 1.5B / 7B / 14B) to see whether coherence scales with size.
- Whether to pair each instruct model with its base model, to see what instruction tuning does to coherence.
- Whether the two small legacy models stay in the probe-only arm or are dropped.
- The cluster budget: memory at bf16 for the largest model, embedding extraction across the layer sweep, and the elicitation passes (primary + stated + 3 paraphrases).

## Answer

**Model list:** 8 checkpoints, each a base + instruct pair, each on one L40 in bf16.

| family | sizes | base | instruct |
|---|---|---|---|
| Qwen2.5 (main size sweep) | 1.5B, 7B, 14B | `Qwen/Qwen2.5-{1.5B,7B,14B}` | `Qwen/Qwen2.5-{1.5B,7B,14B}-Instruct` |
| Gemma 4 (second family) | 12B | `google/gemma-4-12B` | `google/gemma-4-12B-it` |

1. **Size sweep.** Qwen2.5 1.5B / 7B / 14B covers a 10× range. Qwen2.5-32B (2 GPUs) is an optional fourth point, added only if the first three show a trend.
2. **Second family.** The author asked for the best current open model. Gemma 4 12B (released July 2026) was chosen from primary sources (the HF model card) rather than aggregator rankings:
   - It is dense (48 layers), Apache 2.0, and **not gated**, so no `HF_TOKEN` is needed.
   - It ships base and instruct checkpoints.
   - At 12B it pairs with Qwen2.5-14B for a cross-family comparison at matched size.
   - Qwen3.8 leads the aggregator lists but ships no base checkpoints, so it can't serve the base-vs-instruct design. Llama and Mistral were not needed once Gemma 4 qualified.
3. **Base vs. instruct.** Every model runs in both forms. Probes run on all of them. Base models enter the elicited arm (4-shot prompt) only if they pass the ≥ 0.65 competence bar; I expect Qwen2.5-1.5B base to fail it.
4. **Legacy models dropped.** SmolLM2-135M and OPT-1.3b are out of this effort.

Implementation notes for the prevision tickets:
- **Gemma 4 is multimodal.** It loads via `AutoModelForMultimodalLM` + `AutoProcessor`, not the `AutoModelForCausalLM` that `Generate_Embeddings.py` uses. Probe activations must come from its text decoder's residual stream.
- **Gemma 4 thinking mode.** Elicitation must set `enable_thinking=False` in `apply_chat_template()` so the first reply token is the answer.
- **Layer sweep.** Qwen2.5 has 28 / 28 / 48 layers (1.5B / 7B / 14B); Gemma 4 12B has 48. The sweep is by relative depth, so every model is sampled at the same fractions.
- **Budget.** ≈3.2k unique family statements to elicit and ≈20k statements to embed per layer. That's a few GPU-hours per model at most, and every model fits one L40 (48 GB) in bf16.
