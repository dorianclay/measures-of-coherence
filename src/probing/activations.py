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

from coherence import runs
from coherence.paths import DATASETS_DIR, STORE

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


def tokenize(batch: list, tokenizer, bos: str):
    """bos="tokenizer" leaves it to the tokenizer, as 001 did: gemma-4-12B adds <bos> but gemma-4-12B-it doesn't.
    "add" and "remove" force it, whatever the tokenizer does."""
    if bos == "tokenizer":
        return tokenizer(batch, return_tensors="pt", padding=True)
    texts = [tokenizer.bos_token + s for s in batch] if bos == "add" else batch
    return tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)


def chat_text(statement: str, tokenizer, presentation: str) -> str:
    """The statement in a user turn, up to where the model's reply would start. The system prompt is the template's
    default, as in production: we add none and remove none. Reasoning is off."""
    if presentation == "judge-turn":
        from coherence.elicitation import INSTRUCT_TEMPLATES
        statement = INSTRUCT_TEMPLATES["t0"].format(s=statement)
    return tokenizer.apply_chat_template([{"role": "user", "content": statement}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def tokenize_chat(batch: list, tokenizer, presentation: str) -> tuple:
    """Tokenized chat prompts (they contain their own special tokens), and the index of the token to read in each:
    the statement's last token for "chat-statement", the last token of the prompt otherwise."""
    texts = [chat_text(s, tokenizer, presentation) for s in batch]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                       return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping")
    if presentation != "chat-statement":
        return inputs, inputs.attention_mask.sum(dim=1) - 1
    positions = []
    for text, statement, row in zip(texts, batch, offsets.tolist()):
        # Some chat templates trim the content, and a few statements end in a non-breaking space
        stripped = statement.strip()
        end = text.rindex(stripped) + len(stripped)
        positions.append(max(i for i, (start, stop) in enumerate(row) if start < end and stop > start))
    return inputs, torch.tensor(positions)


def prompt_record(tokenizer, presentation: str) -> dict:
    """What a model is actually shown and which token is read, since both differ by chat template."""
    if presentation == "raw":
        return {}
    example = "The statement goes here"
    inputs, last = tokenize_chat([example], tokenizer, presentation)
    return {"prompt_example": chat_text(example, tokenizer, presentation),
            "read_token": tokenizer.convert_ids_to_tokens(inputs.input_ids[0, last[0]].item())}


@torch.no_grad()
def extract(statements: list, model, tokenizer, layers: list, batch_size: int, bos: str = "tokenizer",
            presentation: str = "raw") -> np.ndarray:
    out = []
    for i in tqdm(range(0, len(statements), batch_size), leave=False):
        batch = [s.rstrip(". ") for s in statements[i:i + batch_size]]
        if presentation == "raw":
            inputs = tokenize(batch, tokenizer, bos)
            last = inputs.attention_mask.sum(dim=1) - 1
        else:
            inputs, last = tokenize_chat(batch, tokenizer, presentation)
        inputs = inputs.to(model.device)
        hidden = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False).hidden_states
        last = last.to(inputs.attention_mask.device)
        rows = torch.arange(len(batch), device=last.device)
        out.append(torch.stack([hidden[k][rows, last] for k in layers], dim=1).to(torch.float16).cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description="Extract last-token activations for every dataset at a sweep of relative depths.")
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--datasets", nargs="*", default=DATASETS, help="Dataset names without the '_true_false.csv' suffix.")
    parser.add_argument("--depths", nargs="*", type=float, default=DEFAULT_DEPTHS, help="Relative depths in (0, 1].")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--bos", choices=["tokenizer", "add", "remove"], default="tokenizer",
                        help="Whether statements start with the BOS token; see tokenize().")
    parser.add_argument("--presentation", choices=["raw", "chat-statement", "chat-turn", "judge-turn"], default="raw",
                        help="How the model sees a statement and which token is read: raw text at its last token (001's "
                             "way), or in a user turn at its last token, or at the end of the prompt, or at the end of "
                             "a prompt asking whether it's true.")
    parser.add_argument("--dataset_path", default=DATASETS_DIR)
    parser.add_argument("--output_path", default=STORE / "001-dutch-book" / "activations")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    # Only the revision is wanted: repos can hold duplicate weight formats that the cache need not keep
    runs.log(model_ref=f"{args.model}@{Path(snapshot_download(args.model, allow_patterns=['config.json'])).name}")
    model, tokenizer = init_model(args.model)
    files = {name: Path(args.dataset_path) / f"{name}_true_false.csv" for name in args.datasets}
    out_dir = Path(args.output_path) / args.model.split("/")[-1]
    extract_all(model, tokenizer, args.model, files, args.depths, args.batch_size, out_dir, args.bos, args.presentation)
    runs.log(artifacts=[out_dir])


def extract_all(model, tokenizer, model_name: str, files: dict, depths: list, batch_size: int, out_dir: Path,
                bos: str = "tokenizer", presentation: str = "raw"):
    """Activations of every dataset (name -> CSV with a statement column) to out_dir/<name>.npy, plus meta.json; bos
    and presentation as in extract()."""
    n_layers = num_layers(model)
    layers = layers_for_depths(n_layers, depths)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"model": model_name, "num_layers": n_layers, "layers": layers,
                   "depths": [k / n_layers for k in layers], "bos": bos,
                   "presentation": presentation, **prompt_record(tokenizer, presentation)}, f, indent=1)

    for name, file in tqdm(files.items(), desc="Datasets"):
        statements = pd.read_csv(file, encoding="utf-8-sig").statement.tolist()
        activations = extract(statements, model, tokenizer, layers, batch_size, bos, presentation)
        np.save(out_dir / f"{name}.npy", activations)
        print(f"{name}: {activations.shape} -> {out_dir / f'{name}.npy'}")


if __name__ == "__main__":
    main()
