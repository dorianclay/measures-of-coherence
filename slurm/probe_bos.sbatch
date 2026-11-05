#!/bin/bash
#SBATCH --job-name=probe-bos
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00

# One arm of experiment 004: 001's probes with the BOS token forced on or off, as three recorded runs:
#   1. activations on facts with the tokenizer's defaults, to compare with 001's saved activations;
#   2. activations on every dataset with --bos <add|remove>;
#   3. probes on those activations, and their rates of loss.
#
# Usage (from the repo root; `mkdir -p logs` once):
#   sbatch slurm/probe_bos.sbatch google/gemma-4-12B-it add

set -euo pipefail

MODEL="${1:?usage: sbatch probe_bos.sbatch <hf-model-id> <add|remove>}"
BOS="${2:?usage: sbatch probe_bos.sbatch <hf-model-id> <add|remove>}"
export TAG="${MODEL##*/}" MODEL BOS

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
uv sync --frozen

export ACTIVATIONS_RUN="$(uv run python -c 'from coherence.record import new_run_id; print(new_run_id())')"

uv run record --experiment 004 --note "facts activations, $TAG, bos=tokenizer, to compare with 001's" -- bash -c \
    'python -m probing.activations --model "$MODEL" --bos tokenizer --datasets facts --output_path "$RUN_STORE"'

uv run record --experiment 004 --run-id "$ACTIVATIONS_RUN" --note "activations, $TAG, bos=$BOS" -- bash -c \
    'python -m probing.activations --model "$MODEL" --bos "$BOS" --output_path "$RUN_STORE"'

uv run record --experiment 004 --note "probes, $TAG, bos=$BOS, on activations $ACTIVATIONS_RUN" -- bash -c '
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
