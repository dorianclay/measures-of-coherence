"""Train probes on an extraction run's activations, one protocol per run.

    uv run record --experiment 002 --note "seed 0, rl, step 500" -- \
        python -m probing.probe --activations <extraction-run-id> --protocol belief-probe-linear@1

- belief-probe-linear@1, belief-probe-mlp@1: probing.family_probes' domain-swap probes for that probe kind, booked on
  the facts and companies families by coherence.dutch_book. Previsions and rates go to $RUN_STORE; outputs/ gets the
  probe metrics and rates_summary.csv.
- town-belief-probe-linear@1: a logistic-regression probe at every layer, trained on every statement 001's probes train
  on, applied to the umbrella_towns statements. outputs/beliefs.csv has town, rain, layer, depth, prevision and
  whether the layer is the selected one (best accuracy on a held-out share of the training statements).

The run inherits the extraction run's model reference and lists it under inputs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from coherence import datasets, protocols, runs
from coherence.measure_families import FAMILY_DATASETS, book_and_save, output_dirs
from coherence.paths import data_dir, store_root
from probing.family_probes import ATOMIC_DATASETS, DOMAINS, Activations, probe_previsions, sigmoid, train_linear

IMPLEMENTED = ["belief-probe-linear@1", "belief-probe-mlp@1", "town-belief-probe-linear@1"]
EXTRACTION = "activations-last-token@1"


def extraction_dir(run_id: str, protocol) -> tuple[dict, Path]:
    """The extraction run and its activations directory, refusing anything this protocol can't use."""
    source = runs.read(runs.find(run_id))
    if source.get("protocol") != EXTRACTION or source["status"] != "succeeded":
        raise SystemExit(f"Run {run_id} isn't a succeeded {EXTRACTION} run (protocol {source.get('protocol')}, "
                         f"status {source['status']}); probe runs read activations from one.")
    for name in protocol.datasets:
        if name in source["datasets"] and source["datasets"][name] != datasets.version(name):
            raise SystemExit(f"Run {run_id} extracted {name} version {source['datasets'][name]['version']}, but the "
                             f"current version is {datasets.version(name)['version']}; extract again.")
    [artifact] = [a for a in source["artifacts"] if a["path"].endswith("activations")]
    return source, store_root() / artifact["path"]


def town_beliefs(acts: Activations, directory: Path, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [k for d in ATOMIC_DATASETS + [f"{p}{d}" for d in DOMAINS for p in ["", "neg_", "conj_neg_"]]
            for k in acts.all_rows(d)]
    rng = np.random.default_rng(params["split_seed"])
    held = rng.random(len(keys)) < params["holdout_fraction"]
    towns = pd.read_csv(datasets.path("umbrella_towns"))
    town_acts = np.load(directory / "umbrella_towns.npy")
    beliefs, metrics = [], []
    for layer_index, (layer, depth) in enumerate(zip(acts.meta["layers"], acts.meta["depths"])):
        X, y, _ = acts.rows(keys, layer_index)
        logits_of = train_linear(X[~held], y[~held], params["split_seed"])
        accuracy = float(np.mean((logits_of(X[held]) > 0) == y[held]))
        metrics.append({"layer": layer, "depth": round(depth, 4), "holdout_accuracy": accuracy})
        beliefs.append(towns[["town", "rain"]].assign(
            layer=layer, depth=round(depth, 4),
            prevision=sigmoid(logits_of(town_acts[:, layer_index].astype(np.float32)))))
    metrics = pd.DataFrame(metrics)
    best = metrics.loc[metrics["holdout_accuracy"].idxmax(), "layer"]
    beliefs = pd.concat(beliefs, ignore_index=True)
    return beliefs.assign(selected=beliefs["layer"] == best), metrics.assign(selected=metrics["layer"] == best)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", required=True, metavar="RUN_ID", help="The extraction run to read.")
    parser.add_argument("--protocol", required=True, choices=IMPLEMENTED)
    parser.add_argument("--output", help="Where to write outside `record`.")
    args = parser.parse_args(argv)
    protocol = protocols.get(args.protocol)

    source, directory = extraction_dir(args.activations, protocol)
    store, outputs = output_dirs(args.output)
    runs.log(model_ref=source["model_ref"], protocol=args.protocol, datasets=list(protocol.datasets),
             inputs=[args.activations])
    acts = Activations(directory, data_dir() / "datasets")

    if args.protocol == "town-belief-probe-linear@1":
        beliefs, metrics = town_beliefs(acts, directory, protocol.params)
        outputs.mkdir(parents=True, exist_ok=True)
        beliefs.to_csv(outputs / "beliefs.csv", index=False)
        metrics.to_csv(outputs / "metrics.csv", index=False)
        print(beliefs[beliefs["selected"]].to_string(index=False))
        return

    params = protocol.params
    prevision_rows, metric_rows = probe_previsions(
        acts, Path(acts.meta["model"].split("@")[0]).name, datasets.path(FAMILY_DATASETS[0]).parent,
        params["calibration_fraction"], params["split_seed"], params.get("mlp_seeds", 5), probes=(params["probe"],))
    print(book_and_save(prevision_rows, metric_rows, store, outputs).to_string(index=False))


if __name__ == "__main__":
    main()
