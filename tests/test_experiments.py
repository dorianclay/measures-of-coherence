import pytest

from coherence import experiments


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERENCE_ROOT", str(tmp_path))
    (tmp_path / "experiments").mkdir()
    return tmp_path


def write_experiment(root, dirname, heading, body=""):
    path = root / "experiments" / dirname
    path.mkdir()
    (path / "experiment.org").write_text(f"#+TITLE: {dirname}\n\n* {heading}\n{body}")
    return path


def table_rows(org_text):
    """The data rows of the first Org table: lists of stripped cells, skipping the header and rule."""
    rows = [line for line in org_text.splitlines() if line.startswith("|") and not line.startswith("|-")]
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows[1:]]


def test_new_numbers_the_experiment_after_the_last_one_and_starts_it_as_a_draft(root):
    write_experiment(root, "001-dutch-book", "CONCLUDED 001 Dutch book")

    path = experiments.new("umbrella-rl", "Umbrella RL")

    assert path == root / "experiments" / "002-umbrella-rl"
    experiment = experiments.find("002")
    assert (experiment.state, experiment.title) == ("DRAFT", "002 Umbrella RL")
    org = experiment.org.read_text()
    for section in ["** Question", "** Hypothesis", "** Analysis plan", "** Tasks", "** Deviations", "** Findings"]:
        assert section in org
    assert "#+BEGIN: runs" in org


def test_index_lists_every_experiment_with_its_state_question_and_latest_finding(root):
    write_experiment(root, "001-dutch-book", "CONCLUDED 001 Dutch book",
                     "** Question\nHow incoherent are previsions? And does the source matter?\n\n"
                     "** Findings\n- Elicited beats probe everywhere. Every CI excludes zero.\n- Second finding.\n")
    write_experiment(root, "002-umbrella-rl", "DRAFT 002 Umbrella RL :exploratory:",
                     "** Question\nDoes RL change coherence?\n\n** Findings\n")

    experiments.write_index()

    rows = table_rows((root / "experiments" / "index.org").read_text())
    assert rows == [
        ["001", "[[file:001-dutch-book/experiment.org][Dutch book]]", "CONCLUDED",
         "How incoherent are previsions?", "Elicited beats probe everywhere."],
        ["002", "[[file:002-umbrella-rl/experiment.org][Umbrella RL]]", "DRAFT (exploratory)",
         "Does RL change coherence?", ""],
    ]
