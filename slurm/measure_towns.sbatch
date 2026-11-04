#!/bin/bash
#SBATCH --job-name=measure-towns
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Measure beliefs about each town on every eval checkpoint of a training run, one recorded run per checkpoint and
# protocol (experiments 002 and 003). The measurement runs belong to the training run's experiment.
#
# Usage (from the repo root; `mkdir -p logs` once):
#   sbatch slurm/measure_towns.sbatch <training-run-id>
#   PROTOCOLS="town-belief-elicit-logprob@1" sbatch slurm/measure_towns.sbatch <training-run-id>
#
# Submit it after the training chain, e.g. with --dependency=afterok:<last job of the chain>.

set -euo pipefail

RUN="${1:?usage: sbatch measure_towns.sbatch <training-run-id>}"
PROTOCOLS="${PROTOCOLS:-town-belief-elicit-logprob@1 town-belief-elicit-stated@1}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
uv sync --frozen

read -r EXPERIMENT STEPS < <(uv run python -c "
from coherence import runs
run = runs.read(runs.find('$RUN'))
steps = sorted(a['step'] for a in run.get('artifacts', []) if a.get('checkpoint') == 'eval')
print(run['experiment'], ' '.join(map(str, steps)))")
echo "Job ${SLURM_JOB_ID:-?}: run $RUN of experiment $EXPERIMENT, eval checkpoints at steps $STEPS"

# A failed measurement is recorded as a failed run; carry on with the rest and fail the job at the end
FAILED=0
for STEP in $STEPS; do
    for PROTOCOL in $PROTOCOLS; do
        uv run record --experiment "$EXPERIMENT" --note "$PROTOCOL on $RUN@step$STEP" -- \
            python -m coherence.measure_towns --model "$RUN@step$STEP" --protocol "$PROTOCOL" || FAILED=1
    done
done
exit $FAILED
