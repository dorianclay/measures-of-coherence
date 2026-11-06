#!/bin/bash
# Experiment 004's wider test for one model: for each recipe's presentation, an activations job on <gpus> GPUs with
# <hours> to run, then a probes job on one GPU. --nice leaves room for other people's jobs on the shared node.
#
# Usage (from the repo root):
#   slurm/wider_models.sh 2 8 Qwen/Qwen3.8-27B [batch-size]

set -euo pipefail

GPUS="${1:?usage: wider_models.sh <gpus> <hours> <hf-model-id>}"
HOURS="${2:?usage: wider_models.sh <gpus> <hours> <hf-model-id>}"
MODEL="${3:?usage: wider_models.sh <gpus> <hours> <hf-model-id>}"
BATCH_SIZE="${4:-32}"

for presentation in judge-turn chat-turn; do
    run="$(uv run python -c 'from coherence.record import new_run_id; print(new_run_id())')"
    job="$(sbatch --parsable --nice=100 --gpus="$GPUS" --mem="$((96 + 48 * (GPUS - 1)))G" --time="$HOURS:00:00" \
        slurm/activations_presentation.sbatch "$MODEL" "$presentation" "$run" "$BATCH_SIZE")"
    sbatch --nice=100 --dependency="afterok:$job" --kill-on-invalid-dep=yes \
        slurm/probes_presentation.sbatch "$MODEL" "$presentation" "$run"
done
