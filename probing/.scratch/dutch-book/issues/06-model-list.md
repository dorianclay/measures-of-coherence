# Which models do the experiments run on?

Type: grilling
Status: open
Blocked by: 05

## Question

Fix the model list: open weights only, runnable on the cluster's L40s. The expected starting point is the two already-probed models (SmolLM2-135M, OPT-1.3b) plus one ~7–8B instruct model. Settle whether a size sweep within one family (to see whether coherence scales) is worth adding, given that the elicitation method chosen in 05 may rule out base models for stated probabilities.
