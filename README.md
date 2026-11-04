# measures-of-coherence

Studying how to measure the coherence of agentic systems: whether an LLM's beliefs and values are coherent, measured by probing its activations and eliciting its behavior, and whether that coherence tracks its decisions as it is trained with RL on decision games.

- `ROADMAP.org`: the stages of the work.
- `experiments/`: one directory per experiment, indexed in `experiments/index.org`.
- `docs/design/experiment-recording.org`: how experiments and runs are recorded.
- `CONTEXT.md`: the domain terms.

## Setup

```sh
uv sync
uv run pytest
```

Large outputs go under `$COHERENCE_STORE` (default: the gitignored `store/`). On the cluster, set it to shared storage and submit jobs from the repo root, e.g. `sbatch slurm/probe_previsions.sbatch Qwen/Qwen2.5-7B-Instruct`.

`src/probing/legacy/` and `data/datasets/` come from [balevinstein/Probes](https://github.com/balevinstein/Probes) (MIT; see `src/probing/legacy/LICENSE`).
