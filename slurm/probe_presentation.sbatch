#!/bin/bash
#SBATCH --job-name=probe-presentation
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00

# One arm of experiment 004's second question: 001's probes on statements presented differently, as two recorded runs:
#   1. activations on every dataset with --presentation <presentation>;
#   2. probes on those activations, and their rates of loss.
#
# Usage (from the repo root; `mkdir -p logs` once):
#   sbatch slurm/probe_presentation.sbatch google/gemma-4-12B-it chat-statement

set -euo pipefail

MODEL="${1:?usage: sbatch probe_presentation.sbatch <hf-model-id> <presentation>}"
PRESENTATION="${2:?usage: sbatch probe_presentation.sbatch <hf-model-id> <presentation>}"
export TAG="${MODEL##*/}" MODEL PRESENTATION

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
uv sync --frozen

export ACTIVATIONS_RUN="$(uv run python -c 'from coherence.record import new_run_id; print(new_run_id())')"

uv run record --experiment 004 --run-id "$ACTIVATIONS_RUN" --note "activations, $TAG, presentation=$PRESENTATION" -- bash -c \
    'python -m probing.activations --model "$MODEL" --presentation "$PRESENTATION" --output_path "$RUN_STORE"'

uv run record --experiment 004 --note "probes, $TAG, presentation=$PRESENTATION, on activations $ACTIVATIONS_RUN" -- bash -c '
    set -euo pipefail
    python -m probing.family_probes --model "$TAG" --activations_path "$(dirname "$RUN_STORE")/$ACTIVATIONS_RUN" \
        --output_path "$RUN_STORE"
    python -m coherence.dutch_book \
        --families data/event_families/facts_families.json data/event_families/companies_families.json \
        --previsions "$RUN_STORE/previsions/$TAG.csv" --output "$RUN_STORE/probe_rates/$TAG.csv"
    mkdir -p "$RUN_DIR/outputs"
    cp "$RUN_STORE/probe_metrics/$TAG.csv" "$RUN_DIR/outputs/probe_metrics.csv"
    python -c "
import os
from coherence import runs
runs.log(artifacts=[f\"{os.environ[\"RUN_STORE\"]}/previsions\", f\"{os.environ[\"RUN_STORE\"]}/probe_rates\"])"'
