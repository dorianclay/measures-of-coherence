"""
Event Family Construction for Dutch Book Coherence
Date: 2026-09-29

This script builds event families -- sets of statements whose logical relations are known by construction -- from the
negation and conjunction datasets, so that previsions over them can be checked for coherence (rate of loss to a Dutch book).

Two kinds of family are built for each base dataset (e.g. 'facts', 'companies'):
- negation pairs from neg_<base>_true_false.csv, whose rows come in consecutive pairs (S, not-S);
- conjunction families from conj_neg_<base>_true_false.csv, each conjunction "C1, and C2" matched back to the negation
  pairs containing C1 and C2. Events are always ordered [C1, not-C1, C2, not-C2, C1 and C2].

Each family records its events (statement, label, source row) and its atoms: one row per truth-value assignment
consistent with the family's logic, giving which events are true in that atom. The families are saved as JSON,
one file per base dataset, along with a summary of what was dropped and why.

Requirements:
- pandas library
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

NEGATION_PAIR_ATOMS = [[1, 0], [0, 1]]

# Atoms over (C1, C2) in the order TT, TF, FT, FF; events are [C1, not-C1, C2, not-C2, C1 and C2]
CONJUNCTION_ATOMS = [
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0],
]

NEGATION_MARKER = re.compile(r"n't\b|\bnot\b|\bno\b|\bwithout\b|\bnever\b", re.IGNORECASE)


def normalize(statement: str) -> str:
    """Canonical form used to match a conjunct to its source statement: case, whitespace and final period ignored."""
    return re.sub(r"\s+", " ", statement.strip().rstrip(".").strip()).lower()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def event(statement: str, label: int, source: str, row: int) -> dict:
    return {
        "statement": statement,
        "label": int(label),
        "source": source,
        "row": int(row),
        "surface_negated": bool(NEGATION_MARKER.search(statement)),
    }


def labels_consistent(events: list, atoms: list) -> bool:
    """Ground-truth labels are coherent only if they coincide with one of the family's atoms."""
    return [e["label"] for e in events] in atoms


def build_negation_pairs(neg_df: pd.DataFrame, source: str) -> list:
    if len(neg_df) % 2:
        raise ValueError(f"{source} has an odd number of rows; expected consecutive (S, not-S) pairs")
    pairs, seen = [], {}
    for i in range(0, len(neg_df), 2):
        key = frozenset(normalize(neg_df.statement[j]) for j in (i, i + 1))
        if key in seen:
            # A pair repeated in the file is the same family; booking it twice would double its weight
            seen[key]["duplicate_rows"].append(i)
            continue
        events = [event(neg_df.statement[j], neg_df.label[j], source, j) for j in (i, i + 1)]
        seen[key] = {
            "id": f"{source.split('_true_false')[0]}:pair:{i // 2}",
            "kind": "negation_pair",
            "events": events,
            "atoms": NEGATION_PAIR_ATOMS,
            "labels_consistent": labels_consistent(events, NEGATION_PAIR_ATOMS),
            "duplicate_rows": [],
        }
        pairs.append(seen[key])
    return pairs


def index_pairs(pairs: list) -> dict:
    """Maps each normalized statement to the (pair, member) locations it occurs at."""
    index = defaultdict(list)
    for pair_idx, pair in enumerate(pairs):
        for member, e in enumerate(pair["events"]):
            index[normalize(e["statement"])].append((pair_idx, member))
    return index


def split_conjunction(statement: str, index: dict):
    """
    Returns (C1 location, C2 location, reason). Tries every ', and ' boundary, since conjuncts may themselves
    contain lists ("igneous, sedimentary, and metamorphic"); the split must be the unique one where both halves
    are known statements.
    """
    text = normalize(statement)
    parts = text.split(", and ")
    if len(parts) < 2:
        return None, None, "no ', and ' separator"
    matches = []
    for k in range(1, len(parts)):
        left, right = ", and ".join(parts[:k]), ", and ".join(parts[k:])
        if left in index and right in index:
            matches.append((left, right))
    if not matches:
        halves_found = max(
            (", and ".join(parts[:k]) in index) + (", and ".join(parts[k:]) in index) for k in range(1, len(parts))
        )
        return None, None, "one conjunct not found in negation pairs" if halves_found else "neither conjunct found in negation pairs"
    if len(matches) > 1:
        return None, None, "ambiguous split"
    left, right = matches[0]
    if len(index[left]) > 1 or len(index[right]) > 1:
        return None, None, "conjunct occurs in more than one distinct pair"
    return index[left][0], index[right][0], None


def build_conjunction_families(conj_df: pd.DataFrame, source: str, pairs: list) -> tuple:
    index = index_pairs(pairs)
    families, dropped = [], []
    for row, (statement, label) in enumerate(zip(conj_df.statement, conj_df.label)):
        c1, c2, reason = split_conjunction(statement, index)
        if reason is None and c1[0] == c2[0]:
            reason = "both conjuncts from the same pair"
        if reason:
            dropped.append({"source": source, "row": row, "statement": statement, "reason": reason})
            continue
        events = []
        for pair_idx, member in (c1, c2):
            pair_events = pairs[pair_idx]["events"]
            events += [pair_events[member], pair_events[1 - member]]
        events.append(event(statement, label, source, row))
        families.append({
            "id": f"{source.split('_true_false')[0]}:{row}",
            "kind": "conjunction",
            "pairs": [pairs[c1[0]]["id"], pairs[c2[0]]["id"]],
            "events": events,
            "atoms": CONJUNCTION_ATOMS,
            "labels_consistent": labels_consistent(events, CONJUNCTION_ATOMS),
        })
    return families, dropped


def summarize(pairs: list, conjunctions: list, dropped: list, n_conj_rows: int) -> dict:
    statement_counts = Counter(normalize(e["statement"]) for p in pairs for e in p["events"])
    polarity = Counter(
        f"C1 {'neg' if f['events'][0]['surface_negated'] else 'pos'} / C2 {'neg' if f['events'][2]['surface_negated'] else 'pos'}"
        for f in conjunctions
    )
    return {
        "negation_pairs": len(pairs),
        "negation_pair_duplicates_removed": sum(len(p["duplicate_rows"]) for p in pairs),
        "negation_pairs_label_inconsistent": sum(not p["labels_consistent"] for p in pairs),
        "statements_in_more_than_one_pair": sum(c > 1 for c in statement_counts.values()),
        "conjunction_rows": n_conj_rows,
        "conjunction_families": len(conjunctions),
        "conjunction_families_label_inconsistent": sum(not f["labels_consistent"] for f in conjunctions),
        "conjunction_polarity": dict(polarity),
        "conjunction_dropped": dict(Counter(d["reason"] for d in dropped)),
    }


def main():
    parser = argparse.ArgumentParser(description="Build event families from the negation and conjunction datasets.")
    parser.add_argument("--bases", nargs="*", default=["facts", "companies"],
                        help="Base dataset names; uses neg_<base>_true_false.csv and conj_neg_<base>_true_false.csv.")
    parser.add_argument("--dataset_path", default="datasets", help="Directory containing the datasets.")
    parser.add_argument("--output_path", default="event_families", help="Directory to save the event families to.")
    args = parser.parse_args()

    dataset_path, output_path = Path(args.dataset_path), Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for base in args.bases:
        neg_source, conj_source = f"neg_{base}_true_false", f"conj_neg_{base}_true_false"
        pairs = build_negation_pairs(load_csv(dataset_path / f"{neg_source}.csv"), neg_source)
        conj_df = load_csv(dataset_path / f"{conj_source}.csv")
        conjunctions, dropped = build_conjunction_families(conj_df, conj_source, pairs)
        summary = summarize(pairs, conjunctions, dropped, len(conj_df))

        out_file = output_path / f"{base}_families.json"
        with open(out_file, "w") as f:
            json.dump({"base": base, "summary": summary, "negation_pairs": pairs,
                       "conjunction_families": conjunctions, "dropped": dropped}, f, indent=1)
        print(f"{base}: saved to {out_file}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
