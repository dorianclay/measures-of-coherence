"""Versioned datasets: data/datasets.json maps each dataset name to its file, version and sha256.

Fixing a dataset makes a new version, never a silent edit: change the file, then `uv run datasets bump <name>`.
Runs record the versions they read, and each version's contents stay recoverable from git history.

    uv run datasets check              # every file still matches its recorded version
    uv run datasets add <name> <path>  # start tracking a file at version 1
    uv run datasets bump <name>        # record the edited file as the next version
"""

import argparse
import json
import sys
from pathlib import Path

from coherence.hashing import sha256
from coherence.paths import data_dir


def manifest_path() -> Path:
    return data_dir() / "datasets.json"


def load_manifest() -> dict:
    return json.loads(manifest_path().read_text()) if manifest_path().exists() else {}


def save_manifest(manifest: dict) -> None:
    manifest_path().write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")


def entry(name: str) -> dict:
    manifest = load_manifest()
    if name not in manifest:
        raise KeyError(f"Dataset {name!r} isn't in {manifest_path()}; add it with `uv run datasets add`")
    return manifest[name]


def version(name: str) -> dict:
    """The current version of a dataset and its recorded sha256."""
    latest = entry(name)["history"][-1]
    return {"version": latest["version"], "sha256": latest["sha256"]}


def path(name: str) -> Path:
    """The dataset's file, refusing it if it no longer matches its recorded version."""
    file = data_dir() / entry(name)["path"]
    if sha256(file) != version(name)["sha256"]:
        raise ValueError(f"Dataset {name!r} ({file}) changed since version {version(name)['version']} was recorded. "
                         f"If the change is intended, record it with `uv run datasets bump {name}`.")
    return file


def check() -> list[str]:
    """Names of datasets whose files no longer match their recorded version."""
    return [name for name, e in load_manifest().items() if sha256(data_dir() / e["path"]) != e["history"][-1]["sha256"]]


def add(name: str, file: Path) -> None:
    manifest = load_manifest()
    if name in manifest:
        raise ValueError(f"Dataset {name!r} is already tracked; use bump to record a new version")
    file = Path(file).resolve()
    manifest[name] = {"path": str(file.relative_to(data_dir().resolve())),
                      "history": [{"version": 1, "sha256": sha256(file)}]}
    save_manifest(manifest)


def bump(name: str) -> int:
    manifest = load_manifest()
    e = manifest[name]
    digest = sha256(data_dir() / e["path"])
    if digest == e["history"][-1]["sha256"]:
        raise ValueError(f"Dataset {name!r} is unchanged since version {e['history'][-1]['version']}")
    new = e["history"][-1]["version"] + 1
    e["history"].append({"version": new, "sha256": digest})
    save_manifest(manifest)
    return new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("check")
    add_parser = sub.add_parser("add")
    add_parser.add_argument("name")
    add_parser.add_argument("path", type=Path)
    bump_parser = sub.add_parser("bump")
    bump_parser.add_argument("name")
    args = parser.parse_args(argv)

    if args.action == "check":
        changed = check()
        for name in changed:
            print(f"{name}: changed since its recorded version; bump it if intended", file=sys.stderr)
        sys.exit(1 if changed else 0)
    if args.action == "add":
        add(args.name, args.path)
    if args.action == "bump":
        print(f"{args.name}: now version {bump(args.name)}")


if __name__ == "__main__":
    main()
