"""
Leakage-Free Probe Previsions for Dutch Book Coherence
Date: 2026-09-29

Trains truth probes under the domain-swap protocol and writes their previsions on every event of every event family in the
other domain, in the input format of DutchBook.py.

For each training domain T (facts or companies), the booked domain B is the other one:
- ~20% of T's negation pairs are held out as the calibration slice, together with the conjunction families whose two pairs
  are both held out. Conjunction families with only one held-out pair are used for neither training nor calibration.
- Arm "domain_swap" (primary) trains on the six atomic datasets plus T's base, negation and conjunction datasets, minus
  every statement in the calibration slice. Arm "atomic_only" trains on the six atomic datasets alone.
- Probes: "linear" (standardized logistic regression, the primary probe) for both arms, and "mlp" (the 256-128-64 ReLU
  network of TrainProbes.py, several seeds) for the domain_swap arm.
- A bias-free temperature T is fitted on the calibration slice by log loss; previsions are written raw and calibrated.
No statement of B, and no statement of the calibration slice, is ever trained on; the script checks this and stops if not.

Outputs, under --output_path:
- previsions/<model>.csv: model, arm, probe, seed, layer, depth, booked_domain, calibrated, family_id, event_index, prevision
- probe_metrics/<model>.csv: one row per probe with its temperature, validation accuracy (calibration slice) and
  accuracy and Brier score on the booked domain, raw and calibrated.

Requirements:
- numpy, pandas, scikit-learn, scipy, torch
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ATOMIC_DATASETS = ["cities", "animals", "elements", "inventions", "capitals", "generated"]
DOMAINS = ["facts", "companies"]


def normalize(statement: str) -> str:
    return re.sub(r"\s+", " ", statement.strip().rstrip(".").strip()).lower()


def dataset_of(source: str) -> str:
    return source.removesuffix("_true_false")


class Activations:
    """Activations at every swept layer for every row of every dataset, from Generate_Activations.py."""

    def __init__(self, directory: Path, dataset_path: Path):
        self.meta = json.loads((directory / "meta.json").read_text())
        self.arrays, self.frames = {}, {}
        for name in ATOMIC_DATASETS + [f"{p}{d}" for d in DOMAINS for p in ["", "neg_", "conj_neg_"]]:
            self.arrays[name] = np.load(directory / f"{name}.npy", mmap_mode="r")
            self.frames[name] = pd.read_csv(dataset_path / f"{name}_true_false.csv", encoding="utf-8-sig")

    def rows(self, keys, layer_index: int):
        """keys: iterable of (dataset, row). Returns activations (float32), labels and normalized statements."""
        keys = list(keys)
        X = np.stack([self.arrays[d][r, layer_index] for d, r in keys]).astype(np.float32)
        y = np.array([self.frames[d].label[r] for d, r in keys])
        texts = [normalize(self.frames[d].statement[r]) for d, r in keys]
        return X, y, texts

    def all_rows(self, dataset: str):
        return [(dataset, r) for r in range(len(self.frames[dataset]))]


def load_families(families_path: Path, domain: str) -> dict:
    return json.loads((families_path / f"{domain}_families.json").read_text())


def event_key(event) -> tuple:
    return dataset_of(event["source"]), event["row"]


def split_training_domain(families: dict, fraction: float, seed: int):
    """Returns (calibration keys, statements excluded from training) for the training domain."""
    pairs = families["negation_pairs"]
    rng = np.random.default_rng(seed)
    held = {p["id"] for p in rng.choice(pairs, size=round(fraction * len(pairs)), replace=False)}
    calibration, excluded = set(), set()
    for pair in pairs:
        if pair["id"] in held:
            calibration |= {event_key(e) for e in pair["events"]}
            excluded |= {normalize(e["statement"]) for e in pair["events"]}
    for fam in families["conjunction_families"]:
        n_held = sum(p in held for p in fam["pairs"])
        if n_held:
            excluded.add(normalize(fam["events"][-1]["statement"]))
        if n_held == 2:
            calibration.add(event_key(fam["events"][-1]))
    return sorted(calibration), excluded


def booked_keys(families: dict) -> list:
    return sorted({event_key(e) for fam in families["negation_pairs"] + families["conjunction_families"] for e in fam["events"]})


def training_keys(acts: Activations, arm: str, domain: str, excluded: set) -> list:
    datasets = ATOMIC_DATASETS + ([domain, f"neg_{domain}", f"conj_neg_{domain}"] if arm == "domain_swap" else [])
    keys = [k for d in datasets for k in acts.all_rows(d)]
    return [(d, r) for d, r in keys if normalize(acts.frames[d].statement[r]) not in excluded]


def train_linear(X, y, seed):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=5000).fit(scaler.transform(X), y)
    return lambda Z: clf.decision_function(scaler.transform(Z))


def train_mlp(X, y, seed, epochs: int = 5, batch_size: int = 32):
    """The TrainProbes.py architecture (256-128-64 ReLU, Adam, binary cross-entropy, 5 epochs), in torch."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = StandardScaler().fit(X)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(device)
    optimizer = torch.optim.Adam(net.parameters())
    Xt = torch.tensor(scaler.transform(X), dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xt), generator=generator).split(batch_size):
            optimizer.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[idx]).squeeze(1), yt[idx]).backward()
            optimizer.step()
    net.eval()

    def logits(Z):
        with torch.no_grad():
            return net(torch.tensor(scaler.transform(Z), dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()
    return logits


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_temperature(logits, y) -> float:
    """Bias-free temperature scaling: T minimizing the log loss of sigmoid(logit / T) on the calibration slice."""
    def log_loss(log_t):
        # Computed from logits, not clipped probabilities: saturated probes would otherwise give a flat loss
        z = logits / np.exp(log_t)
        return np.mean(np.logaddexp(0, np.where(y == 1, -z, z)))
    return float(np.exp(minimize_scalar(log_loss, bounds=(-5, 5), method="bounded").x))


def check_no_leakage(train_texts, other_texts, what: str):
    leaked = set(train_texts) & set(other_texts)
    if leaked:
        raise RuntimeError(f"{len(leaked)} {what} statements are in the training set, e.g. {sorted(leaked)[:3]}")


def main():
    parser = argparse.ArgumentParser(description="Train leakage-free probes and write their previsions on event families.")
    parser.add_argument("--model", required=True, help="Model name as used for activations/<model>/ (last part of the HF id).")
    parser.add_argument("--activations_path", default="activations")
    parser.add_argument("--families_path", default="event_families")
    parser.add_argument("--dataset_path", default="datasets")
    parser.add_argument("--output_path", default="results/dutch_book")
    parser.add_argument("--calibration_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--mlp_seeds", type=int, default=5)
    args = parser.parse_args()

    acts = Activations(Path(args.activations_path) / args.model, Path(args.dataset_path))
    layers, depths = acts.meta["layers"], acts.meta["depths"]
    prevision_rows, metric_rows = [], []

    for training_domain in DOMAINS:
        booked_domain = next(d for d in DOMAINS if d != training_domain)
        train_families = load_families(Path(args.families_path), training_domain)
        booked_families = load_families(Path(args.families_path), booked_domain)
        calibration, excluded = split_training_domain(train_families, args.calibration_fraction, args.split_seed)
        booked = booked_keys(booked_families)
        event_lookup = [(fam["id"], i, event_key(e))
                        for fam in booked_families["negation_pairs"] + booked_families["conjunction_families"]
                        for i, e in enumerate(fam["events"])]

        for arm in ["domain_swap", "atomic_only"]:
            train = training_keys(acts, arm, training_domain, excluded)
            probes = [("linear", 0)] + ([("mlp", s) for s in range(args.mlp_seeds)] if arm == "domain_swap" else [])
            for layer_index, (layer, depth) in enumerate(zip(layers, depths)):
                X, y, train_texts = acts.rows(train, layer_index)
                Xc, yc, cal_texts = acts.rows(calibration, layer_index)
                Xb, yb, booked_texts = acts.rows(booked, layer_index)
                check_no_leakage(train_texts, booked_texts, f"booked-domain ({booked_domain})")
                check_no_leakage(train_texts, cal_texts, "calibration-slice")

                for probe, seed in probes:
                    logits_of = (train_linear if probe == "linear" else train_mlp)(X, y, seed)
                    zc, zb = logits_of(Xc), logits_of(Xb)
                    temperature = fit_temperature(zc, yc)
                    by_key = {}
                    for calibrated, p in [(False, sigmoid(zb)), (True, sigmoid(zb / temperature))]:
                        by_key[calibrated] = dict(zip(booked, p))
                    ident = {"model": args.model, "arm": arm, "probe": probe, "seed": seed, "layer": layer,
                             "depth": round(depth, 4), "booked_domain": booked_domain}
                    metric_rows.append({**ident, "training_statements": len(train), "temperature": temperature,
                                        "validation_accuracy": float(np.mean((zc > 0) == yc)),
                                        "booked_accuracy": float(np.mean((zb > 0) == yb)),
                                        "booked_brier_raw": float(np.mean((sigmoid(zb) - yb) ** 2)),
                                        "booked_brier_calibrated": float(np.mean((sigmoid(zb / temperature) - yb) ** 2))})
                    for calibrated in (False, True):
                        prevision_rows += [{**ident, "calibrated": calibrated, "family_id": fid, "event_index": i,
                                            "prevision": by_key[calibrated][key]} for fid, i, key in event_lookup]
                    print(f"{training_domain}->{booked_domain} {arm} {probe}:{seed} layer {layer}: "
                          f"val acc {metric_rows[-1]['validation_accuracy']:.3f}, booked acc {metric_rows[-1]['booked_accuracy']:.3f}, T {temperature:.2f}")

    for name, rows in [("previsions", prevision_rows), ("probe_metrics", metric_rows)]:
        out = Path(args.output_path) / name / f"{args.model}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
