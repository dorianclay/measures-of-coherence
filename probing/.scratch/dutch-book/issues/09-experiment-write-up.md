# Write up the experiments and every decision behind them

Type: task (HITL)
Status: resolved
Blocked by: 05, 06, 07, 08, 10, 12

## Question

Once the experiments have run, write a thorough account of them that someone could reproduce and defend. It must detail **every decision on this map**: what was decided, the alternatives, and why. That covers:
- the rate-of-loss formulation;
- calibration;
- event-family construction, including drops and label inconsistencies;
- the leakage protocol;
- elicitation;
- models, layers, and seeds;
- aggregation and statistics.

The write-up also covers methods, results (figures and tables for both prevision sources, both family shapes, and all controls), limitations, and the fixes to thesis §4.1.2. Those fixes include the notation corrections found by [Which rate-of-loss formulation(s) do we compute?](01-rate-of-loss-formulation.md) and the worked "Dutch book-ability" example the thesis leaves blank.

Source every decision from the map's resolved tickets and link each one. The write-up restates their content for a reader who never saw the map.

Also blocked by [Implement the aggregation and comparison analysis](12-aggregation-implementation.md), which implements the decisions from [How do per-family rates aggregate and compare across prevision sources?](11-aggregation-and-comparison.md). Worked with the author, who reviews the draft: format and home (repo doc, thesis chapter section, or both) are decided at the start of this ticket.

## Answer

**Kicked off with the author** (grilled to a shared understanding before drafting): two artifacts rather than one, since there's no editable thesis source in this repo (only the compiled PDF in `references/`) —

1. **`Dutch_Book_Write_Up.ipynb`** (repo root, matching the existing `Visualize_Predictions.ipynb` convention) — the full technical report, self-contained, one notebook, 9 sections (background → results at a glance → methods & decisions, synthesized with links back to each ticket rather than restated in full → full results tables/figures → live verification of the conjunction-independence finding and a pedagogical dyadic-bootstrap demo → the `gemma-4-12B-it` split shown in real statements → limitations → thesis fixes → a reproducibility index). Built and **executed end-to-end** (jupyter/nbformat installed mid-ticket) — zero errors across 38 cells, every number cross-checked live against the committed tables rather than hardcoded from memory (e.g. the ceiling/floor and the connected-component sizes are recomputed in the notebook itself, not just quoted).
2. **`.scratch/dutch-book/thesis-4.1.2-patch.md`** — literal paste-ready replacement text for thesis §4.1.2: the KSS/SSK reconciliation paragraph (already drafted on the formulation research branch), three smaller notation fixes, and a new worked example built from the numbers already verified there.

Per the author: use whatever data is needed for a thorough analysis (including the large cluster-local files), prioritizing a strong one-time run on this remote over portability; compress to a small committed artifact where that's easy. Concretely, the notebook's headline tables/figures load only the small already-committed `results/dutch_book/aggregation/` outputs, but its illustrative-examples section reads the large per-model previsions directly (cluster-local only) and the results are saved out to a new small committed file, `results/dutch_book/aggregation/illustrative_examples.csv`.

**What the write-up documents**, sourced and linked from every resolved ticket: the rate-of-loss formulation (01), calibration (02), event families (03), the leakage protocol (04), elicitation (05), models (06), the solver (07), probe previsions (08), elicited previsions (10), and the aggregation/comparison methodology (11) and its implementation (12). Headline results: elicited previsions beat probe previsions in all 32 raw-arm paired-difference cells (none the reverse), and the `gemma-4-12B-it` split (chance-level probe, comfortably-passing elicitation) is shown both statistically and in two real, contrasting statements ("The Earth is flat" vs. "The endocrine system regulates body functions through hormones").

**Ready for review.** The map's only remaining item is the fog note "Merged algebras" (worth a new ticket only if the author wants to pursue it further) — every ticketed question on the map is now resolved.
