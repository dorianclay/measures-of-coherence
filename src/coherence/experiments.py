"""Experiments: one directory per research question, experiments/NNN-<slug>/experiment.org.

The state of an experiment is the TODO keyword of the first top-level heading in its experiment.org, and an
:exploratory: tag on that heading marks it exploratory.

    uv run experiments new <slug> "<Title>"   # the next numbered experiment, as a DRAFT
    uv run experiments index                  # regenerate every run table and experiments/index.org
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from coherence.paths import experiments_dir

STATES = ["DRAFT", "REGISTERED", "RUNNING", "CONCLUDED", "ABANDONED"]
HEADING = re.compile(rf"^\* ({'|'.join(STATES)}) (.*?)\s*(:[\w:]+:)?\s*$", re.MULTILINE)


@dataclass
class Experiment:
    id: str
    path: Path
    state: str
    title: str
    tags: list[str]

    @property
    def name(self) -> str:
        """The title without its leading id."""
        return self.title.removeprefix(self.id).strip()

    @property
    def org(self) -> Path:
        return self.path / "experiment.org"

    @property
    def exploratory(self) -> bool:
        return "exploratory" in self.tags

    @property
    def registered(self) -> bool:
        """Registered or later: the plan is fixed, so runs must come from committed code."""
        return self.state != "DRAFT" and not self.exploratory


def load(path: Path) -> Experiment:
    match = HEADING.search((path / "experiment.org").read_text())
    if not match:
        raise ValueError(f"{path / 'experiment.org'} has no top-level heading with a state ({', '.join(STATES)})")
    state, title, tags = match.groups()
    return Experiment(id=path.name.split("-")[0], path=path, state=state, title=title,
                      tags=[t for t in (tags or "").split(":") if t])


def find(experiment_id: str) -> Experiment:
    matches = sorted(experiments_dir().glob(f"{experiment_id}-*/experiment.org"))
    if len(matches) != 1:
        raise SystemExit(f"Expected one experiment {experiment_id}-*/experiment.org under {experiments_dir()}, "
                         f"found {len(matches)}")
    return load(matches[0].parent)


TEMPLATE = """#+TITLE: {id} {title}
#+TODO: DRAFT REGISTERED RUNNING | CONCLUDED ABANDONED

* DRAFT {id} {title}

** Question

** Hypothesis
What we expect, stated before the first real run. Fixed once the experiment is REGISTERED.

** Analysis plan
The runs (models, protocols, seeds per condition), the comparisons and the figures that will answer the question.
Fixed once the experiment is REGISTERED.

** Tasks

** Deviations
Dated entries for every change to the hypothesis or plan after registration. The plan above is never edited.

** Runs
#+BEGIN: runs
#+END:

** Findings
Written or changed only with the author's sign-off. Cite figures from figures/.
"""


def all_experiments() -> list[Experiment]:
    return [load(p.parent) for p in sorted(experiments_dir().glob("[0-9][0-9][0-9]-*/experiment.org"))]


def new(slug: str, title: str) -> Path:
    ids = [int(e.id) for e in all_experiments()]
    experiment_id = f"{max(ids, default=0) + 1:03d}"
    path = experiments_dir() / f"{experiment_id}-{slug}"
    path.mkdir(parents=True)
    (path / "experiment.org").write_text(TEMPLATE.format(id=experiment_id, title=title))
    write_index()
    return path


def section(org_text: str, heading: str) -> str:
    """The body of a second-level section, up to the next heading."""
    match = re.search(rf"^\*\* {re.escape(heading)}\s*\n(.*?)(?=^\*+ |\Z)", org_text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    return re.split(r"(?<=[.?!])\s", text, maxsplit=1)[0] if text else ""


def summary(experiment: Experiment) -> tuple[str, str]:
    """The experiment's question and its first finding, one sentence each."""
    org = experiment.org.read_text()
    question = section(org, "Question").split("\n\n")[0]
    bullets = [line[2:] for line in section(org, "Findings").splitlines() if line.startswith("- ")]
    return first_sentence(question), first_sentence(bullets[0]) if bullets else ""


def org_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    rule = "|" + "+".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(header), rule, *map(line, rows)])


RUNS_BLOCK = re.compile(r"^#\+BEGIN: runs.*?\n.*?^#\+END:", re.MULTILINE | re.DOTALL)


def write_run_table(experiment: Experiment) -> None:
    """Rewrite the runs dynamic block in experiment.org from the run records."""
    rows = []
    for run_json in sorted(experiment.path.glob("runs/*/run.json")):
        run = json.loads(run_json.read_text())
        rows.append([f"[[file:runs/{run['run_id']}/run.json][{run['run_id']}]]", run["status"],
                     run.get("model_ref", ""), run.get("protocol", ""), run.get("note", "")])
    table = org_table(["run", "status", "model reference", "protocol", "note"], rows)
    org = experiment.org.read_text()
    if not RUNS_BLOCK.search(org):
        raise ValueError(f"{experiment.org} has no '#+BEGIN: runs' ... '#+END:' block for the run table")
    experiment.org.write_text(RUNS_BLOCK.sub(lambda _: f"#+BEGIN: runs\n{table}\n#+END:", org, count=1))


def write_index() -> None:
    rows = []
    for e in all_experiments():
        question, finding = summary(e)
        state = f"{e.state} (exploratory)" if e.exploratory else e.state
        rows.append([e.id, f"[[file:{e.path.name}/experiment.org][{e.name}]]", state, question, finding])
    (experiments_dir() / "index.org").write_text(
        "#+TITLE: Experiments\n\nGenerated by `uv run experiments index`; edit the experiments, not this file.\n\n"
        + org_table(["id", "experiment", "state", "question", "latest finding"], rows) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    new_parser = sub.add_parser("new")
    new_parser.add_argument("slug", help="Short kebab-case name, e.g. umbrella-rl-belief-coherence.")
    new_parser.add_argument("title", help="Human title, e.g. 'Umbrella RL: belief coherence'.")
    sub.add_parser("index")
    args = parser.parse_args(argv)
    if args.action == "new":
        print(new(args.slug, args.title))
    if args.action == "index":
        for e in all_experiments():
            if RUNS_BLOCK.search(e.org.read_text()):
                write_run_table(e)
        write_index()


if __name__ == "__main__":
    main()
