"""Book 001's event families against a model's elicited previsions, one protocol per run, on any model reference.

    uv run record --experiment 002 --note "seed 0, rl, step 500" -- \
        python -m coherence.measure_families --model 20261104-0900-abcd@step500 --protocol belief-elicit-logprob@1

Runs coherence.elicitation's method for the protocol (chat prompts for instruct models, decided from the base model's
name as in 001), then coherence.dutch_book on the facts and companies families. Previsions and rates go to
$RUN_STORE; the run directory's outputs/ gets the elicitation metrics and rates_summary.csv, the mean rate of loss per
bookie and normalization.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from coherence import datasets, dutch_book, models, protocols, runs
from coherence.elicitation import elicit, looks_instruct

IMPLEMENTED = ["belief-elicit-logprob@1", "belief-elicit-stated@1"]
FAMILY_DATASETS = ["facts_families", "companies_families"]


def output_dirs(output: str | None) -> tuple[Path, Path]:
    """Where large outputs (the store) and small ones (the run's outputs/) go."""
    if runs.current_run_dir():
        return Path(os.environ["RUN_STORE"]), runs.current_run_dir() / "outputs"
    if not output:
        raise SystemExit("run under `record`, or give --output")
    return Path(output), Path(output)


def book_and_save(prevision_rows: list[dict], metric_rows: list[dict], store: Path, outputs: Path) -> pd.DataFrame:
    """Book the families against the previsions; save previsions and rates to the store, metrics and a summary of the
    mean rate per bookie to outputs. Returns the summary."""
    families = dutch_book.load_families([datasets.path(name) for name in FAMILY_DATASETS])
    previsions = pd.DataFrame(prevision_rows)
    rates = dutch_book.book(families, previsions)
    bookie = [c for c in previsions.columns if c not in dutch_book.PREVISION_KEYS + ["prevision"]]
    summary = (rates.groupby(bookie + ["normalization"], sort=False)["rate"].mean()
               .rename("mean_rate").reset_index())
    store.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    previsions.to_csv(store / "previsions.csv", index=False)
    rates.to_csv(store / "rates.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(outputs / "metrics.csv", index=False)
    summary.to_csv(outputs / "rates_summary.csv", index=False)
    runs.log(artifacts=[store / "previsions.csv", store / "rates.csv"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model reference: <hf-id>@<commit>, <path>@local or "
                                                       "<run-id>@step<N>.")
    parser.add_argument("--protocol", required=True, choices=IMPLEMENTED)
    parser.add_argument("--output", help="Where to write outside `record`.")
    args = parser.parse_args(argv)
    protocol = protocols.get(args.protocol)
    method = protocol.params["method"]

    _, _, base_name = models.resolve(args.model)
    instruct = looks_instruct(base_name)
    if method == "stated" and not instruct:
        raise SystemExit(f"{args.protocol} elicits stated probabilities, which 001 asks only of instruct models; "
                         f"{base_name} isn't one")
    store, outputs = output_dirs(args.output)
    families_path = datasets.path(FAMILY_DATASETS[0]).parent
    runs.log(model_ref=args.model, protocol=args.protocol, datasets=list(protocol.datasets))
    model, tokenizer = models.load(args.model)
    prevision_rows, metric_rows = elicit(model, tokenizer, Path(base_name).name, instruct, families_path,
                                         calibration_fraction=protocol.params["calibration_fraction"],
                                         split_seed=protocol.params["split_seed"], methods=(method,))
    print(book_and_save(prevision_rows, metric_rows, store, outputs).to_string(index=False))


if __name__ == "__main__":
    main()
