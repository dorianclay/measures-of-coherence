"""Measure a model's beliefs about each town's weather (experiment 002's in-task beliefs), one protocol per run.

    uv run record --experiment 002 --note "seed 0, rl, step 250" -- \
        python -m coherence.measure_towns --model 20261104-0900-abcd@step250 --protocol town-belief-elicit-logprob@1

Each statement in the dataset umbrella_towns ("It will rain in <town> tomorrow." and its negation) gets a prevision
by 001's elicitation methods, always with chat prompts, uncalibrated: the towns' rain has no ground truth to calibrate
against, since each seed hides different chances. Writes outputs/beliefs.csv in the run directory (town, rain,
template, prevision; stated answers that don't parse are left empty).
"""

import argparse
from pathlib import Path

import pandas as pd

from coherence import datasets, models, protocols, runs
from coherence.elicitation import judgment_prompt, logprob_previsions, stated_previsions

IMPLEMENTED = ["town-belief-elicit-logprob@1", "town-belief-elicit-stated@1"]


def measure(protocol, model, tokenizer, statements: pd.DataFrame) -> pd.DataFrame:
    method, templates = protocol.params["method"], protocol.params["templates"]
    rows = []
    for template in templates:
        texts = statements["statement"].tolist()
        if method == "logprob":
            prompts = [judgment_prompt(tokenizer, True, template, s) for s in texts]
            previsions, _ = logprob_previsions(prompts, model, tokenizer, True, batch_size=len(prompts))
        else:
            previsions = stated_previsions(texts, model, tokenizer)
        rows.append(statements[["town", "rain"]].assign(template=template, prevision=previsions))
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model reference: <hf-id>@<commit>, <path>@local or "
                                                       "<run-id>@step<N>.")
    parser.add_argument("--protocol", required=True, choices=IMPLEMENTED)
    parser.add_argument("--output", help="Where to write beliefs.csv outside `record`.")
    args = parser.parse_args(argv)
    protocol = protocols.get(args.protocol)

    output = runs.current_run_dir() / "outputs" if runs.current_run_dir() else Path(args.output or ".")
    statements = pd.read_csv(datasets.path("umbrella_towns"))
    runs.log(model_ref=args.model, protocol=args.protocol, datasets=list(protocol.datasets))
    model, tokenizer = models.load(args.model)
    beliefs = measure(protocol, model, tokenizer, statements)
    output.mkdir(parents=True, exist_ok=True)
    beliefs.to_csv(output / "beliefs.csv", index=False)
    print(beliefs.pivot_table(index="town", columns=["template", "rain"], values="prevision").round(3))


if __name__ == "__main__":
    main()
