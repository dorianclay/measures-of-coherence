"""Load run records for analysis, and save figures and tables with a sidecar naming what they came from.

    loaded = runs.load(["20261103-1412-a3f9", ...])
    runs.save(fig, experiment_dir / "figures" / "box_rates.png", loaded)

Every figure or table in a write-up comes through `save`, so its sidecar (`<file>.sources.json`) traces it back to
its runs, their protocol and dataset versions, and the analysis code's commit.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from coherence import git
from coherence.paths import experiments_dir


def load(run_ids: list[str], allow_mixed_protocols: bool = False) -> list[dict]:
    """The run records, in the order asked for, each with its directory under "dir".

    Results are comparable only within one protocol version, so runs of different protocols are refused unless
    comparing them is the point.
    """
    loaded = []
    for run_id in run_ids:
        matches = list(experiments_dir().glob(f"*/runs/{run_id}/run.json"))
        if not matches:
            raise KeyError(f"No run {run_id} under {experiments_dir()}")
        loaded.append({**json.loads(matches[0].read_text()), "dir": matches[0].parent})
    found = sorted({str(r.get("protocol")) for r in loaded})
    if len(found) > 1 and not allow_mixed_protocols:
        raise ValueError(f"Runs mix protocols {', '.join(found)}; results are comparable only within one protocol "
                         f"version. Pass allow_mixed_protocols=True if comparing them is the point.")
    return loaded


def save(output, path: Path, loaded: list[dict]) -> None:
    """Write a matplotlib figure or a pandas DataFrame to `path`, with its sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(output, "savefig"):
        output.savefig(path, bbox_inches="tight")
    else:
        output.to_csv(path, index=False)
    datasets = {}
    for run in loaded:
        datasets.update(run.get("datasets", {}))
    sidecar = {"output": path.name, "runs": [r["run_id"] for r in loaded],
               "protocols": sorted({r["protocol"] for r in loaded if r.get("protocol")}),
               "datasets": datasets, "analysis_commit": git.commit(), "script": sys.argv[0],
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.with_name(path.name + ".sources.json").write_text(json.dumps(sidecar, indent=2) + "\n")
