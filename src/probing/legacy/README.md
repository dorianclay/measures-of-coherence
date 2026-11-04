# Probes (legacy)

> **Legacy.** This is upstream's original pipeline from [balevinstein/Probes](https://github.com/balevinstein/Probes) (MIT, see `LICENSE` here), kept for the CCS probe method until it's ported to the activations-based probe interface. The supervised workflow below (`TrainProbes.py`, `Generate_predictions.py`) used TensorFlow and was deleted; it's in git history. Scripts now run as modules from the repo root, e.g. `uv run python -m probing.legacy.train_ccs_probe`; they read their configs from this directory, datasets from `data/`, and write to `$COHERENCE_STORE/legacy/`.



## Workflow for Supervised Probes
For supervised probes, the code is largely based on Azaria and Mitchell's implementation for their paper [`The Internal State of an LLM Knows When It's Lying.'](https://arxiv.org/abs/2304.13734). Datasets used were also theirs, or based on theirs.

1. `GenerateEmbeddings.py` or `LLaMa_generate_embeddings.py` on selected datasets to get the embeddings for the last token at specified layer(s) for a specified model. You can use config.json or commandline arguments. Will save CSV files with the embeddings. Make sure to get embeddings for labeled datasets so the probes can be trained. The latter file implements functionality for LLaMA, but since those models aren't fully publicly available, the implementation won't work generally.
2. `TrainProbes.py` to train the probes on selected datasets that contain embeddings. You can specify lots of parameters in the config file or with commandline flags. The script will test the probes on a different dataset from the training datasets and save the best (by accuracy) probes.
3. If you want to use the trained probes on a new dataset (with embeddings), run `Generate_Embeddings.py` with that dataset selected. Make sure the layer, model, etc. line up with the probes you want to use. It will save the predictions along with the average prediction.

## Workflow for CCS (Unsupervised)
For unsupervised probes, the code is almost entirely based on Burn's et al.'s [implementation](https://github.com/collin-burns/discovering_latent_knowledge)
1. Again, run `GenerateEmbeddings.py` but make sure you use ones with negative and positive paired examples. Right now, those are `neg_facts_true_false.csv` and `neg_companies_true_false.csv`. 
2. Run `Train_CCSProbe.py` with the relevant datasets that include embeddings. These are specified in the companion config file `CCS_config.json`. This will save the CCS probes and also the predictions for the datasets used.
3. If you want to use the trained probes on a new dataset (that includes embeddings), run `Generate_CCS_predictions.py` with that dataset specified in the config file. As always, make sure the model, layer, etc. line up with the probe. 
