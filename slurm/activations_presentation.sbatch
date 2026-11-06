#!/bin/bash
#SBATCH --job-name=activations-presentation
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00

# Activations on every dataset with --presentation <presentation>, as a recorded run with a given id, so that the
# probes job can be submitted before this one starts. slurm/wider_models.sh submits the pair.
#
# Usage (from the repo root; `mkdir -p logs` once):
#   sbatch --gpus=2 slurm/activations_presentation.sbatch Qwen/Qwen3.8-27B judge-turn <run-id>

set -euo pipefail

export MODEL="${1:?usage: sbatch activations_presentation.sbatch <hf-model-id> <presentation> <run-id>}"
export PRESENTATION="${2:?usage: sbatch activations_presentation.sbatch <hf-model-id> <presentation> <run-id>}"
RUN="${3:?usage: sbatch activations_presentation.sbatch <hf-model-id> <presentation> <run-id>}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
uv sync --frozen

uv run record --experiment 004 --run-id "$RUN" --note "activations, ${MODEL##*/}, presentation=$PRESENTATION" -- bash -c \
    'python -m probing.activations --model "$MODEL" --presentation "$PRESENTATION" --output_path "$RUN_STORE"'
