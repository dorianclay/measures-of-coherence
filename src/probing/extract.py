"""Extract activations for every probe protocol from one model reference, as its own run.

    uv run record --experiment 002 --note "seed 0, rl, step 500" -- \
        python -m probing.extract --model 20261104-0900-abcd@step500

probing.activations' last-token extraction on every dataset of activations-last-token@1 (001's statement datasets and
umbrella_towns) to $RUN_STORE/activations/. Probe runs (probing.probe) cite this run with --activations, so every
probe method trains on the same activations.
"""

import argparse
import os
from pathlib import Path

from coherence import datasets, models, protocols, runs
from probing.activations import extract_all


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model reference: <hf-id>@<commit>, <path>@local or "
                                                       "<run-id>@step<N>.")
    parser.add_argument("--protocol", default="activations-last-token@1", choices=["activations-last-token@1"])
    parser.add_argument("--output", help="Where to write outside `record`.")
    args = parser.parse_args(argv)
    protocol = protocols.get(args.protocol)

    store = Path(os.environ.get("RUN_STORE") or args.output or parser.error("run under `record`, or give --output"))
    files = {name: datasets.path(name) for name in protocol.datasets}
    runs.log(model_ref=args.model, protocol=args.protocol, datasets=list(protocol.datasets))
    model, tokenizer = models.load(args.model)
    out = store / "activations"
    extract_all(model, tokenizer, args.model, files, protocol.params["depths"], protocol.params["batch_size"], out)
    runs.log(artifacts=[out])


if __name__ == "__main__":
    main()
