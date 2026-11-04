import json

import pytest

from coherence import datasets

# sha256 of the version-1 contents, computed independently with `shasum -a 256`
SHA_V1 = "f2be7cba2e3b24386b9aa17c574f8d07c5c097a0414a4307ca79133c8e89d9b2"


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("COHERENCE_ROOT", str(tmp_path))
    (tmp_path / "data" / "datasets").mkdir(parents=True)
    return tmp_path / "data"


def write_dataset(data_root, text):
    path = data_root / "datasets" / "toy_true_false.csv"
    path.write_text(text)
    return path


def test_an_added_dataset_starts_at_version_1_and_resolves_to_its_file(data_root):
    path = write_dataset(data_root, "statement,label\nThe sky is blue.,1\n")

    datasets.add("toy", path)

    assert datasets.path("toy") == path
    assert datasets.version("toy") == {"version": 1, "sha256": SHA_V1}


def test_a_dataset_edited_without_a_version_bump_is_refused(data_root):
    path = write_dataset(data_root, "statement,label\nThe sky is blue.,1\n")
    datasets.add("toy", path)
    path.write_text("statement,label\nThe sky is green.,1\n")

    with pytest.raises(ValueError, match="bump"):
        datasets.path("toy")
    assert datasets.check() == ["toy"]


def test_bumping_records_a_new_version_and_keeps_the_old_one_in_the_history(data_root):
    path = write_dataset(data_root, "statement,label\nThe sky is blue.,1\n")
    datasets.add("toy", path)
    path.write_text("statement,label\nThe sky is green.,0\n")

    datasets.bump("toy")

    assert datasets.version("toy")["version"] == 2
    assert datasets.check() == []
    manifest = json.loads((data_root / "datasets.json").read_text())
    assert [v["version"] for v in manifest["toy"]["history"]] == [1, 2]
    assert manifest["toy"]["history"][0]["sha256"] == SHA_V1
