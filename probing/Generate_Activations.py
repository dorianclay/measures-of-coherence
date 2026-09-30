"""
Activation Extraction for Dutch Book Probes
Date: 2026-09-29

Extracts residual-stream activations at the last token of each statement (before the final period) for every dataset,
at a sweep of layers chosen by relative depth so that models of different depths are sampled at the same fractions.

Unlike Generate_Embeddings.py, which writes one text CSV per layer, activations are saved as one float16 array per
dataset, activations/<model>/<dataset>.npy with shape (statements, layers, hidden size), in the dataset's row order,
plus activations/<model>/meta.json recording the model, the layers and their depths. This keeps 14B-scale models over a
full layer sweep to a couple of GB.

Loads any causal LM on the Hugging Face hub; multimodal models (e.g. Gemma 4) fall back to AutoModelForMultimodalLM and
are run on text only.

Requirements:
- transformers library
- torch library
- pandas library
- numpy library
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

DATASETS = ["cities", "animals", "elements", "inventions", "capitals", "generated",
            "facts", "companies", "neg_facts", "neg_companies", "conj_neg_facts", "conj_neg_companies"]
DEFAULT_DEPTHS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]


def device_and_dtype():
    if torch.cuda.is_available():
        return "auto", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def init_model(model_name: str):
    device, dtype = device_and_dtype()
    kwargs = {"dtype": dtype, "device_map": device} if device == "auto" else {"dtype": dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (ValueError, KeyError):
        # Multimodal checkpoints (e.g. Gemma 4) have no causal-LM mapping; their text path gives the same residual stream
        model = AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The last real token is found from the attention mask, which assumes padding on the right
    tokenizer.padding_side = "right"
    return model.eval(), tokenizer


def num_layers(model) -> int:
    config = model.config
    config = getattr(config, "text_config", None) or config
    return config.num_hidden_layers


def layers_for_depths(n_layers: int, depths: list) -> list:
    """hidden_states[0] is the embedding output, so layer k's output is hidden_states[k] for k in 1..n_layers."""
    return sorted({max(1, round(d * n_layers)) for d in depths})


@torch.no_grad()
def extract(statements: list, model, tokenizer, layers: list, batch_size: int) -> np.ndarray:
    out = []
    for i in tqdm(range(0, len(statements), batch_size), leave=False):
        batch = [s.rstrip(". ") for s in statements[i:i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        hidden = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states
        last = inputs.attention_mask.sum(dim=1) - 1
        rows = torch.arange(len(batch), device=last.device)
        out.append(torch.stack([hidden[k][rows, last] for k in layers], dim=1).to(torch.float16).cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description="Extract last-token activations for every dataset at a sweep of relative depths.")
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Dataset names without the '_true_false.csv' suffix.")
    parser.add_argument("--depths", nargs="*", type=float, default=DEFAULT_DEPTHS, help="Relative depths in (0, 1].")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset_path", default="datasets")
    parser.add_argument("--output_path", default="activations")
    args = parser.parse_args()

    model, tokenizer = init_model(args.model)
    n_layers = num_layers(model)
    layers = layers_for_depths(n_layers, args.depths)
    out_dir = Path(args.output_path) / args.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"model": args.model, "num_layers": n_layers, "layers": layers,
                   "depths": [k / n_layers for k in layers]}, f, indent=1)

    for name in tqdm(args.datasets, desc="Datasets"):
        statements = pd.read_csv(Path(args.dataset_path) / f"{name}_true_false.csv", encoding="utf-8-sig").statement.tolist()
        activations = extract(statements, model, tokenizer, layers, args.batch_size)
        np.save(out_dir / f"{name}.npy", activations)
        print(f"{name}: {activations.shape} -> {out_dir / f'{name}.npy'}")


if __name__ == "__main__":
    main()
