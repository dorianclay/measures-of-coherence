#!/bin/bash
#SBATCH --job-name=measure-checkpoints
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Experiment 002's measurement schedule on a training run's eval checkpoints, one recorded run per checkpoint and
# protocol, in the training run's experiment:
#   - every eval checkpoint: the town-belief protocols (two elicitation, one probe);
#   - eval checkpoints at 0, 50% and 100% of the planned steps: 001's four general-coherence protocols.
# Probe protocols read one activation extraction run per checkpoint.
#
# Usage (from the repo root; `mkdir -p logs` once), after the training chain:
#   sbatch --dependency=afterok:<last job of the chain> slurm/measure_checkpoints.sbatch <training-run-id>
#   STEPS="0 500" sbatch slurm/measure_checkpoints.sbatch <training-run-id>   # only these checkpoints

set -euo pipefail

RUN="${1:?usage: sbatch measure_checkpoints.sbatch <training-run-id>}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
uv sync --frozen

# Prints: experiment, then every eval step, then the general-coherence steps (those at 0, 50% and 100% of the plan)
mapfile -t PLAN < <(uv run python -c "
from coherence import runs
run = runs.read(runs.find('$RUN'))
steps = sorted(a['step'] for a in run.get('artifacts', []) if a.get('checkpoint') == 'eval')
planned = run['planned_steps']
print(run['experiment']); print(*steps); print(*[s for s in steps if s in {0, round(planned / 2), planned}])")
EXPERIMENT="${PLAN[0]}"
STEPS="${STEPS:-${PLAN[1]}}"
GENERAL_STEPS=" ${PLAN[2]} "
echo "Job ${SLURM_JOB_ID:-?}: run $RUN of experiment $EXPERIMENT; checkpoints $STEPS; general coherence at$GENERAL_STEPS"

# A failed measurement is recorded as a failed run; carry on with the rest and fail the job at the end
FAILED=0
measure() {  # note, then the command
    local note="$1"; shift
    uv run record --experiment "$EXPERIMENT" --note "$note" -- "$@" || FAILED=1
}
for STEP in $STEPS; do
    REF="$RUN@step$STEP"
    for PROTOCOL in town-belief-elicit-logprob@1 town-belief-elicit-stated@1; do
        measure "$PROTOCOL on $REF" python -m coherence.measure_towns --model "$REF" --protocol "$PROTOCOL"
    done
    EXTRACTION=$(uv run python -c "from coherence.record import new_run_id; print(new_run_id())")
    uv run record --experiment "$EXPERIMENT" --run-id "$EXTRACTION" --note "activations of $REF" -- \
        python -m probing.extract --model "$REF" || { FAILED=1; continue; }
    PROBES="town-belief-probe-linear@1"
    if [[ "$GENERAL_STEPS" == *" $STEP "* ]]; then
        PROBES="$PROBES belief-probe-linear@1 belief-probe-mlp@1"
        for PROTOCOL in belief-elicit-logprob@1 belief-elicit-stated@1; do
            measure "$PROTOCOL on $REF" python -m coherence.measure_families --model "$REF" --protocol "$PROTOCOL"
        done
    fi
    for PROTOCOL in $PROBES; do
        measure "$PROTOCOL on $REF" python -m probing.probe --activations "$EXTRACTION" --protocol "$PROTOCOL"
    done
done
exit $FAILED
