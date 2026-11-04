import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from analysis import runs


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERENCE_ROOT", str(tmp_path))
    (tmp_path / "experiments" / "002-toy").mkdir(parents=True)
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "README").write_text("toy\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def add_run(root, run_id, protocol, datasets=None):
    run_dir = root / "experiments" / "002-toy" / "runs" / run_id
    run_dir.mkdir(parents=True)
    run = {"run_id": run_id, "experiment": "002", "status": "succeeded", "protocol": protocol,
           "datasets": datasets or {}, "segments": []}
    (run_dir / "run.json").write_text(json.dumps(run))


def test_load_finds_runs_by_id_across_experiments(root):
    add_run(root, "20261103-1412-aaaa", "belief-probe-linear@1")
    add_run(root, "20261103-1413-bbbb", "belief-probe-linear@1")

    loaded = runs.load(["20261103-1413-bbbb", "20261103-1412-aaaa"])

    assert [r["run_id"] for r in loaded] == ["20261103-1413-bbbb", "20261103-1412-aaaa"]
    assert loaded[0]["dir"] == root / "experiments" / "002-toy" / "runs" / "20261103-1413-bbbb"


def test_load_refuses_to_mix_protocol_versions_unless_asked(root):
    add_run(root, "20261103-1412-aaaa", "belief-probe-linear@1")
    add_run(root, "20261103-1413-bbbb", "belief-probe-linear@2")

    with pytest.raises(ValueError, match="belief-probe-linear@1.*belief-probe-linear@2"):
        runs.load(["20261103-1412-aaaa", "20261103-1413-bbbb"])
    assert len(runs.load(["20261103-1412-aaaa", "20261103-1413-bbbb"], allow_mixed_protocols=True)) == 2


def test_load_names_a_run_id_that_does_not_exist(root):
    with pytest.raises(KeyError, match="20261103-0000-dead"):
        runs.load(["20261103-0000-dead"])


@pytest.mark.parametrize("make_output, filename", [
    (lambda: plt.subplots()[0], "box.png"),
    (lambda: pd.DataFrame({"model": ["a"], "rate": [0.1]}), "rates.csv"),
])
def test_save_writes_the_output_and_a_sidecar_naming_its_sources(root, make_output, filename):
    add_run(root, "20261103-1412-aaaa", "belief-probe-linear@1", {"toy": {"version": 1, "sha256": "abc"}})
    add_run(root, "20261103-1413-bbbb", "belief-probe-linear@1", {"toy": {"version": 1, "sha256": "abc"}})
    loaded = runs.load(["20261103-1412-aaaa", "20261103-1413-bbbb"])
    out = root / "experiments" / "002-toy" / "figures" / filename

    runs.save(make_output(), out, loaded)

    assert out.exists()
    sidecar = json.loads(out.with_name(filename + ".sources.json").read_text())
    assert sidecar["runs"] == ["20261103-1412-aaaa", "20261103-1413-bbbb"]
    assert sidecar["protocols"] == ["belief-probe-linear@1"]
    assert sidecar["datasets"] == {"toy": {"version": 1, "sha256": "abc"}}
    assert sidecar["analysis_commit"] == {"sha": git(root, "rev-parse", "HEAD"), "dirty": False}
