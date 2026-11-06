#!/bin/bash
#SBATCH --job-name=probes-presentation
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00

# Probes and their rates of loss on the activations of a given run, as a recorded run. slurm/wider_models.sh submits it
# to follow slurm/activations_presentation.sbatch.
#
# Usage (from the repo root; `mkdir -p logs` once):
#   sbatch slurm/probes_presentation.sbatch Qwen/Qwen3.8-27B judge-turn <activations-run-id>

set -euo pipefail

MODEL="${1:?usage: sbatch probes_presentation.sbatch <hf-model-id> <presentation> <activations-run-id>}"
PRESENTATION="${2:?usage: sbatch probes_presentation.sbatch <hf-model-id> <presentation> <activations-run-id>}"
export ACTIVATIONS_RUN="${3:?usage: sbatch probes_presentation.sbatch <hf-model-id> <presentation> <activations-run-id>}"
export TAG="${MODEL##*/}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
uv sync --frozen

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
