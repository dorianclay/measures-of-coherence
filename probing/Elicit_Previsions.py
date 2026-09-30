"""
Elicited Previsions for Dutch Book Coherence
Date: 2026-09-30

Elicits previsions from a model's own behavior for every statement in every event family, and writes them in the input
format of DutchBook.py. Each statement is asked in its own prompt, so the model never sees other members of its family.

Methods:
- "logprob" (primary): P(True) / (P(True) + P(False)) for the first answer token, summed over the " True"/"True" spellings.
  Instruct models get a chat prompt with one fixed system role (thinking disabled); base models a 4-shot plain prompt
  with examples from the atomic datasets. Three templates: t0 is the primary, t1 and t2 are paraphrases.
- "stated" (instruct models only, template t0): the stated probability 0-100, greedy-decoded and parsed.

Calibration mirrors the probe side: for previsions booked in domain B, a bias-free temperature on logit(prevision) is fitted
by log loss on the calibration slice of the other domain (the same split as Train_Family_Probes.py). The accuracy on that
slice is the competence bar (>= 0.65) for entering the probe-vs-elicited comparison.

Outputs, under --output_path:
- previsions/elicited_<model>.csv: model, method, template, booked_domain, calibrated, family_id, event_index, prevision
- elicitation_metrics/<model>.csv: per method, template and booked domain: temperature, competence accuracy and whether it
  passes the bar, booked accuracy and Brier score, mean probability mass on the answer tokens, stated-parse failures.

Requirements:
- transformers, torch, pandas, numpy, scipy
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from Generate_Activations import init_model
from Train_Family_Probes import DOMAINS, fit_temperature, load_families, normalize, split_training_domain

COMPETENCE_BAR = 0.65
SYSTEM = "You are a careful fact-checker. Judge each statement on its own."

INSTRUCT_TEMPLATES = {
    "t0": "Is the following statement true or false? Answer with one word, True or False.\n\nStatement: {s}",
    "t1": "Statement: {s}\n\nIs this statement correct? Reply with True or False only.",
    "t2": 'Consider the claim "{s}". Answer True if it is accurate and False if it is not. One word only.',
}
STATED_TEMPLATE = ("What is the probability, from 0 to 100, that the following statement is true? "
                   "Reply with a single number only.\n\nStatement: {s}")

# From the atomic datasets, which are never booked
FEWSHOT = [
    ("Thimphu is a name of a city.", "True"),
    ("The eagle has a habitat of urban/wild.", "False"),
    ("Boron is used in the production of glass and ceramics.", "True"),
    ("Praia is a name of a country.", "False"),
]
BASE_TEMPLATES = {
    "t0": "Statement: {s}\nTrue or False? {a}",
    "t1": "Claim: {s}\nIs the claim true? {a}",
    "t2": '"{s}" - True or False? {a}',
}


def looks_instruct(model_name: str) -> bool:
    # Base checkpoints (e.g. Qwen2.5) also ship chat templates, so decide from the name
    return bool(re.search(r"(instruct|-it$|-chat)", model_name.split("/")[-1], re.IGNORECASE))


def chat_prompt(tokenizer, user: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        # Templates without a system role: fold it into the user turn
        messages = [{"role": "user", "content": f"{SYSTEM}\n\n{user}"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def judgment_prompt(tokenizer, instruct: bool, template: str, statement: str) -> str:
    if instruct:
        return chat_prompt(tokenizer, INSTRUCT_TEMPLATES[template].format(s=statement))
    shots = "\n\n".join(BASE_TEMPLATES[template].format(s=s, a=a) for s, a in FEWSHOT)
    return shots + "\n\n" + BASE_TEMPLATES[template].format(s=statement, a="").rstrip()


def answer_token_ids(tokenizer) -> dict:
    ids = {}
    for answer in ["True", "False"]:
        ids[answer] = sorted({tokenizer.encode(v, add_special_tokens=False)[0] for v in (answer, " " + answer)})
    if set(ids["True"]) & set(ids["False"]):
        raise RuntimeError(f"True/False share a first token: {ids}")
    return ids


@torch.no_grad()
def logprob_previsions(prompts, model, tokenizer, instruct: bool, batch_size: int):
    ids = answer_token_ids(tokenizer)
    previsions, masses = [], []
    for i in tqdm(range(0, len(prompts), batch_size), leave=False):
        batch = prompts[i:i + batch_size]
        # Chat prompts already contain their special tokens
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=not instruct).to(model.device)
        logits = model(**inputs).logits
        last = inputs.attention_mask.sum(dim=1) - 1
        probs = torch.softmax(logits[torch.arange(len(batch), device=last.device), last].float(), dim=-1)
        p_true, p_false = probs[:, ids["True"]].sum(1), probs[:, ids["False"]].sum(1)
        previsions += (p_true / (p_true + p_false)).cpu().tolist()
        masses += (p_true + p_false).cpu().tolist()
    return previsions, masses


@torch.no_grad()
def stated_previsions(statements, model, tokenizer):
    previsions = []
    for statement in tqdm(statements, leave=False):
        prompt = chat_prompt(tokenizer, STATED_TEMPLATE.format(s=statement))
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        text = tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        match = re.search(r"\d+(\.\d+)?", text)
        previsions.append(min(float(match.group()), 100) / 100 if match else math.nan)
    return previsions


def logit(p, eps: float = 1e-4):
    p = np.clip(p, eps, 1 - eps)  # stated previsions are often exactly 0 or 1
    return np.log(p) - np.log(1 - p)


def main():
    parser = argparse.ArgumentParser(description="Elicit previsions from model behavior for every event family statement.")
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--instruct", choices=["auto", "yes", "no"], default="auto",
                        help="Chat prompts (instruct) or 4-shot prompts (base); auto decides from the model name.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--families_path", default="event_families")
    parser.add_argument("--output_path", default="results/dutch_book")
    parser.add_argument("--calibration_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=0)
    args = parser.parse_args()

    instruct = looks_instruct(args.model) if args.instruct == "auto" else args.instruct == "yes"
    model_tag = args.model.split("/")[-1]
    model, tokenizer = init_model(args.model)
    print(f"{args.model}: {'instruct (chat prompts)' if instruct else 'base (4-shot prompts)'}; answer ids {answer_token_ids(tokenizer)}")

    families = {d: load_families(Path(args.families_path), d) for d in DOMAINS}
    # Every family event, keyed by normalized statement: each distinct statement is elicited once
    events = {d: [(fam["id"], i, e) for fam in families[d]["negation_pairs"] + families[d]["conjunction_families"]
                  for i, e in enumerate(fam["events"])] for d in DOMAINS}
    statements = {d: sorted({normalize(e["statement"]): e["statement"] for _, _, e in events[d]}.items()) for d in DOMAINS}
    labels = {d: {normalize(e["statement"]): e["label"] for _, _, e in events[d]} for d in DOMAINS}
    calibration_texts = {}
    for d in DOMAINS:
        keys, _ = split_training_domain(families[d], args.calibration_fraction, args.split_seed)
        by_key = {(e["source"].removesuffix("_true_false"), e["row"]): normalize(e["statement"]) for _, _, e in events[d]}
        calibration_texts[d] = sorted({by_key[k] for k in keys})

    runs = [("logprob", t) for t in ["t0", "t1", "t2"]] + ([("stated", "t0")] if instruct else [])
    elicited, masses, failures = {}, {}, {}
    for method, template in runs:
        for d in DOMAINS:
            texts = [s for _, s in statements[d]]
            if method == "logprob":
                prompts = [judgment_prompt(tokenizer, instruct, template, s) for s in texts]
                p, m = logprob_previsions(prompts, model, tokenizer, instruct, args.batch_size)
                masses[method, template, d] = float(np.mean(m))
            else:
                p = stated_previsions(texts, model, tokenizer)
                failures[method, template, d] = int(np.isnan(p).sum())
                p = [0.5 if math.isnan(x) else x for x in p]  # unparseable answers carry no information
            elicited[method, template, d] = dict(zip([k for k, _ in statements[d]], p))

    prevision_rows, metric_rows = [], []
    for method, template in runs:
        for booked_domain in DOMAINS:
            other = next(d for d in DOMAINS if d != booked_domain)
            prev = elicited[method, template, booked_domain]
            cal = np.array([elicited[method, template, other][t] for t in calibration_texts[other]])
            y_cal = np.array([labels[other][t] for t in calibration_texts[other]])
            temperature = fit_temperature(logit(cal), y_cal)
            booked_texts = [k for k, _ in statements[booked_domain]]
            p_raw = np.array([prev[t] for t in booked_texts])
            y = np.array([labels[booked_domain][t] for t in booked_texts])
            p_cal = 1 / (1 + np.exp(-logit(p_raw) / temperature))
            competence = float(np.mean((cal > 0.5) == y_cal))
            metric_rows.append({
                "model": model_tag, "instruct": instruct, "method": method, "template": template,
                "booked_domain": booked_domain, "temperature": temperature,
                "competence_accuracy": competence, "passes_bar": competence >= COMPETENCE_BAR,
                "booked_accuracy": float(np.mean((p_raw > 0.5) == y)),
                "booked_brier_raw": float(np.mean((p_raw - y) ** 2)),
                "booked_brier_calibrated": float(np.mean((p_cal - y) ** 2)),
                "answer_token_mass": masses.get((method, template, booked_domain)),
                "stated_parse_failures": failures.get((method, template, booked_domain)),
            })
            calibrated_of = dict(zip(booked_texts, p_cal))
            for fid, i, e in events[booked_domain]:
                t = normalize(e["statement"])
                for calibrated, p in [(False, prev[t]), (True, calibrated_of[t])]:
                    prevision_rows.append({"model": model_tag, "method": method, "template": template,
                                           "booked_domain": booked_domain, "calibrated": calibrated,
                                           "family_id": fid, "event_index": i, "prevision": float(p)})
            m = metric_rows[-1]
            print(f"{method}/{template} -> {booked_domain}: competence {m['competence_accuracy']:.3f} "
                  f"({'pass' if m['passes_bar'] else 'FAIL'}), booked acc {m['booked_accuracy']:.3f}, T {temperature:.2f}")

    for name, rows, filename in [("previsions", prevision_rows, f"elicited_{model_tag}.csv"),
                                 ("elicitation_metrics", metric_rows, f"{model_tag}.csv")]:
        out = Path(args.output_path) / name / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
