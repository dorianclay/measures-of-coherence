"""Measure a model's beliefs about each town's weather (experiment 002's in-task beliefs), one protocol per run.

    uv run record --experiment 002 --note "seed 0, rl, step 250" -- \
        python -m coherence.measure_towns --model 20261104-0900-abcd@step250 --protocol town-belief-elicit-logprob@1

Each statement in the dataset umbrella_towns ("It will rain in <town> tomorrow." and its negation) gets a prevision
by 001's elicitation methods, always with chat prompts, uncalibrated: the towns' rain has no ground truth to calibrate
against, since each seed hides different chances. Writes outputs/beliefs.csv in the run directory (town, rain,
template, prevision; stated answers that don't parse are left empty).

--told-chances checks that a protocol can read a belief at all: every town is measured at every given chance of rain,
stated at the top of the prompt, and the output gains a told_chance column. It isn't a measurement of the model's own
beliefs, so these runs are not comparable with runs without it.

The event-belief protocols do the same for the framed task's events (dataset framed_events), with a framing column in
place of the town.
"""

import argparse
from pathlib import Path

import pandas as pd

from coherence import datasets, models, protocols, runs
from coherence.elicitation import INSTRUCT_TEMPLATES, STATED_TEMPLATE, chat_prompt, logprob_previsions, stated_from_prompts

IMPLEMENTED = ["town-belief-elicit-logprob@1", "town-belief-elicit-stated@1",
               "town-forecast-elicit-logprob@1", "town-forecast-elicit-stated@1",
               "event-belief-elicit-logprob@1", "event-belief-elicit-stated@1"]

FORECAST_TEMPLATES = {
    "f0": "Give your best guess about the following forecast. Is it true or false? Answer with one word, True or "
          "False.\n\nForecast: {s}",
    "f1": "Forecast: {s}\n\nDo you expect this to turn out true or false? Reply with True or False only.",
    "f2": 'Consider the forecast "{s}". Answer True if you expect it to come true and False if you do not. One word only.',
    "s0": "What is your best estimate of the probability, from 0 to 100, that the following forecast comes true? "
          "Reply with a single number only.\n\nForecast: {s}",
}


def prompt(protocol, tokenizer, template: str, row) -> str:
    told = ""
    if "told_chance" in row._fields:
        told = (f"The chance of the following is {row.told_chance:.0%}: {row.event}\n\n" if "event" in row._fields
                else f"In {row.town}, it rains on {row.told_chance:.0%} of days.\n\n")
    if protocol.params.get("framing") == "forecast":
        messages = [{"role": "user", "content": told + FORECAST_TEMPLATES[template].format(s=row.statement)}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    question = INSTRUCT_TEMPLATES[template] if protocol.params["method"] == "logprob" else STATED_TEMPLATE
    return chat_prompt(tokenizer, told + question.format(s=row.statement))


def measure(protocol, model, tokenizer, statements: pd.DataFrame) -> pd.DataFrame:
    method, templates = protocol.params["method"], protocol.params["templates"]
    rows = []
    for template in templates:
        prompts = [prompt(protocol, tokenizer, template, row) for row in statements.itertuples()]
        if method == "logprob":
            previsions, _ = logprob_previsions(prompts, model, tokenizer, True, batch_size=len(prompts))
        else:
            previsions = stated_from_prompts(prompts, model, tokenizer)
        rows.append(statements.drop(columns=["statement", "event"], errors="ignore").assign(template=template, prevision=previsions))
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model reference: <hf-id>@<commit>, <path>@local or "
                                                       "<run-id>@step<N>.")
    parser.add_argument("--protocol", required=True, choices=IMPLEMENTED)
    parser.add_argument("--told-chances", type=float, nargs="+", help="State these chances of rain in the prompt.")
    parser.add_argument("--output", help="Where to write beliefs.csv outside `record`.")
    args = parser.parse_args(argv)
    protocol = protocols.get(args.protocol)

    output = runs.current_run_dir() / "outputs" if runs.current_run_dir() else Path(args.output or ".")
    (dataset,) = protocol.datasets
    statements = pd.read_csv(datasets.path(dataset))
    if args.told_chances:
        statements = statements.merge(pd.DataFrame({"told_chance": args.told_chances}), how="cross")
    runs.log(model_ref=args.model, protocol=args.protocol, datasets=list(protocol.datasets))
    model, tokenizer = models.load(args.model)
    beliefs = measure(protocol, model, tokenizer, statements)
    output.mkdir(parents=True, exist_ok=True)
    beliefs.to_csv(output / "beliefs.csv", index=False)
    print(beliefs.pivot_table(index=[c for c in ["town", "framing", "told_chance"] if c in beliefs],
                              columns=["template", "rain"],
                              values="prevision").round(3))


if __name__ == "__main__":
    main()
