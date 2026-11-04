"""The code version a run or an analysis output came from."""

import subprocess

from coherence.paths import repo_root

# Run records, the tables generated from them and analysis outputs change with every run; they aren't code, so they
# never make the tree count as dirty.
NOT_CODE = [":(exclude)experiments/*/runs/**", ":(exclude)experiments/*/figures/**",
            ":(exclude)experiments/*/experiment.org", ":(exclude)experiments/index.org"]


def git(*args, check=True) -> str:
    return subprocess.run(["git", *args], cwd=repo_root(), check=check, capture_output=True, text=True).stdout


def uncommitted_changes() -> str:
    """A patch of every uncommitted change to code, untracked files included; empty when the tree is clean."""
    patch = git("diff", "HEAD", "--", ".", *NOT_CODE)
    for path in git("ls-files", "--others", "--exclude-standard", "--", ".", *NOT_CODE).splitlines():
        patch += git("diff", "--no-index", "--", "/dev/null", path, check=False)
    return patch


def commit() -> dict:
    """The HEAD commit, and whether code had uncommitted changes on top of it."""
    return {"sha": git("rev-parse", "HEAD").strip(), "dirty": bool(uncommitted_changes())}
