"""
PROTOTYPE — throwaway. Answers: how should elicited previsions be read from a model?

Compares, on a small sample of conjunction families (each contains its two negation pairs):
  - logprob:  P(" True") / (P(" True") + P(" False")) for the next token after a judgment prompt
              (few-shot plain-text prompt for base models; chat template for instruct models)
  - stated:   the instruct model's stated probability 0-100 (greedy decode, parsed), divided by 100
and prints the previsions per family with a naive rate of loss L (Andrews' L1-budget LP).

Run: uv run --no-sync python PROTOTYPE_elicitation.py [--n 10] [--models ...]
"""
import argparse, json, random, re, math
import numpy as np, torch
from scipy.optimize import linprog
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM = "You are a careful fact-checker. Judge each statement on its own."
FEWSHOT = [  # from atomic datasets, never booked
    ("Thimphu is a name of a city.", "True"),
    ("The eagle has a habitat of urban/wild.", "False"),
    ("Boron is used in the production of glass and ceramics.", "True"),
    ("Praia is a name of a country.", "False"),
]


def rate_of_loss(p, atoms):
    """Andrews' L: max_b min_atoms sum b_i (1_Ei - p_i), sum|b| <= 1. Variables [b+ (n), b- (n), t]."""
    A, p = np.array(atoms, float), np.array(p, float)
    n = len(p)
    D = A - p  # atoms x events
    # t - D b+ + D b- <= 0 for every atom
    A_ub = np.vstack([np.hstack([-D, D, np.ones((len(A), 1))]), np.hstack([np.ones(2 * n), [0]])])
    b_ub = np.concatenate([np.zeros(len(A)), [1]])
    res = linprog(np.concatenate([np.zeros(2 * n), [-1]]), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * (2 * n) + [(None, None)])
    return max(0.0, -res.fun)


def load(name):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16 if dev == "mps" else torch.float32).to(dev).eval()
    return tok, model, dev


def is_chat(tok):
    return tok.chat_template is not None


def judgment_prompt(tok, statement):
    if is_chat(tok):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Is the following statement true or false? Answer with one word, True or False.\n\nStatement: {statement}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    shots = "".join(f"Statement: {s}\nTrue or False? {a}\n\n" for s, a in FEWSHOT)
    return shots + f"Statement: {statement}\nTrue or False?"


def answer_token_ids(tok):
    # Chat models emit "True" at the start of the reply; few-shot base prompts continue with " True"
    variants = {"True": ["True", " True"], "False": ["False", " False"]}
    return {k: sorted({tok.encode(v, add_special_tokens=False)[0] for v in vs}) for k, vs in variants.items()}


@torch.no_grad()
def logprob_prevision(tok, model, dev, statement, ids):
    enc = tok(judgment_prompt(tok, statement), return_tensors="pt", add_special_tokens=not is_chat(tok)).to(dev)
    probs = torch.softmax(model(**enc).logits[0, -1].float(), -1)
    pt, pf = probs[ids["True"]].sum().item(), probs[ids["False"]].sum().item()
    return pt / (pt + pf), pt + pf  # prevision, mass on the two answers (sanity)


@torch.no_grad()
def stated_prevision(tok, model, dev, statement):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"What is the probability, from 0 to 100, that the following statement is true? Reply with a single number only.\n\nStatement: {statement}"}]
    enc = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).to(dev)
    out = model.generate(**enc, max_new_tokens=6, do_sample=False)
    text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    m = re.search(r"\d+(\.\d+)?", text)
    return (min(float(m.group()), 100) / 100 if m else math.nan), text.strip()


def sample_families(n):
    rng = random.Random(0)
    fams = []
    for base in ["facts", "companies"]:
        d = json.load(open(f"event_families/{base}_families.json"))
        fams += rng.sample([f for f in d["conjunction_families"] if f["labels_consistent"]], n)
    return fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="families per domain")
    ap.add_argument("--models", nargs="*", default=["HuggingFaceTB/SmolLM2-135M", "facebook/opt-1.3b", "Qwen/Qwen2.5-1.5B-Instruct"])
    args = ap.parse_args()
    fams = sample_families(args.n)
    summary = {}
    for name in args.models:
        tok, model, dev = load(name)
        ids = answer_token_ids(tok)
        methods = ["logprob"] + (["stated"] if is_chat(tok) else [])
        print(f"\n{'=' * 100}\n{name}   methods={methods}   answer token ids={ids}")
        rates = {m: [] for m in methods}; correct = {m: 0 for m in methods}; allp = {m: [] for m in methods}; mass = []
        for f in fams:
            print(f"\n  {f['id']}")
            prev = {m: [] for m in methods}
            for e in f["events"]:
                pl, ms = logprob_prevision(tok, model, dev, e["statement"], ids); mass.append(ms)
                prev["logprob"].append(pl)
                row = f"    [{e['label']}] logprob={pl:.2f}"
                if "stated" in methods:
                    ps, raw = stated_prevision(tok, model, dev, e["statement"])
                    prev["stated"].append(0.5 if math.isnan(ps) else ps)
                    row += f"  stated={ps:.2f} ({raw!r})"
                print(row + f"   {e['statement'][:80]}")
            for m in methods:
                L = rate_of_loss(prev[m], f["atoms"]); rates[m].append(L); allp[m] += prev[m]
                correct[m] += sum((p > 0.5) == bool(e["label"]) for p, e in zip(prev[m], f["events"]))
                print(f"    L[{m}] = {L:.3f}")
        n_ev = 5 * len(fams)
        for m in methods:
            hist = np.histogram(allp[m], bins=10, range=(0, 1))[0].tolist()
            summary[(name, m)] = (np.mean(rates[m]), np.median(rates[m]), correct[m] / n_ev, hist)
        summary[(name, "mass")] = float(np.mean(mass))
        del model; torch.mps.empty_cache() if dev == "mps" else None
    print(f"\n{'=' * 100}\nSUMMARY ({len(fams)} conjunction families, {5 * len(fams)} statements)")
    for k, v in summary.items():
        if k[1] == "mass":
            print(f"  {k[0]:32s} mean P(True)+P(False) on answer tokens = {v:.3f}")
        else:
            print(f"  {k[0]:32s} {k[1]:8s} mean L={v[0]:.3f} median L={v[1]:.3f} acc={v[2]:.2f} prevision histogram(0..1, 10 bins)={v[3]}")


if __name__ == "__main__":
    main()
