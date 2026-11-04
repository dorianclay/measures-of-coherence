# measures-of-coherence

Thesis research: does an LLM's coherence track its decisions? We train models with RL on decision games and measure the coherence of their beliefs and values (probing and elicitation) at checkpoints along the way. `ROADMAP.org` has the stages and what's current. Domain terms are in `CONTEXT.md`; use them.

## Experiments and runs

Every result belongs to an experiment and traces back to its runs. How experiments, runs, protocols, artifacts and segments are recorded is in `docs/design/experiment-recording.org`. Read it before creating an experiment, launching a run, or writing analysis. Its build order says which parts of the recorder exist yet.

The rules that hold everywhere:

- A registered experiment's hypothesis and analysis plan stay as committed. Any change is a dated deviation entry.
- Every run keeps its record, failed or not. Analysis reports every seed that ran.
- Figures and tables come from analysis scripts that take run ids; findings cite those figures.

## What needs the author's sign-off

Claude may freely: create DRAFT and exploratory experiments; write code and tests; launch runs estimated at up to 48 GPU-hours in total, and re-run failed segments of approved runs; write run notes; regenerate run tables, figures and `experiments/index.org`; draft analysis; push commits that contain only run records.

Ask the author first to: move an experiment to REGISTERED; add a deviation; write or change an experiment's **Findings**; bump a protocol version; delete any artifact or run record; launch a run estimated above 48 GPU-hours in total (state the estimate); push anything else.

## Code

- One uv project; packages live in `src/` (`coherence`, `probing`, `analysis`). Run code as modules, e.g. `uv run python -m coherence.dutch_book`. Paths resolve from the repo root through `src/coherence/paths.py`, so the working directory doesn't matter.
- Large outputs go under `$COHERENCE_STORE` (local default: the gitignored `store/`), with only small outputs in git.
- Cluster jobs are sbatch files in `slurm/`, submitted from the repo root. The cluster caps each job at 48 GPU-hours, so long training resumes across segments.
- `src/probing/legacy/` is upstream's unported CCS pipeline; build new probe methods against the activations in `src/probing/`.

## Commits

One commit per finished unit of work. Experiment code goes on `exp/NNN-<slug>` and merges to `main` when stable; run-record commits go on the branch the code ran from.
