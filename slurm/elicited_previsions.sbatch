#!/bin/bash
#SBATCH --job-name=elicited-previsions
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Elicited previsions and their rates of loss for the Dutch book experiments, one model per job:
#   1. coherence.elicitation asks the model about every event family statement, each in its own prompt
#      (True/False logprobs over three templates; stated probabilities too for instruct models)
#   2. coherence.dutch_book books every family against every elicitation method and template
#
# Usage (from the repo root; create the log dir once with `mkdir -p logs`):
#   sbatch slurm/elicited_previsions.sbatch Qwen/Qwen2.5-7B-Instruct
#
# All eight models of the experiments:
#   for m in Qwen/Qwen2.5-{1.5B,7B,14B}{,-Instruct} google/gemma-4-12B{,-it}; do
#       sbatch slurm/elicited_previsions.sbatch "$m"; done
#
# Outputs, all under $COHERENCE_STORE/001-dutch-book/: previsions/elicited_<model>.csv and elicited_rates/<model>.csv (large;
# stay on the cluster), plus the small elicitation_metrics/<model>.csv to copy into experiments/001-dutch-book/results/.
#
# Cluster-specific settings (partition, account, qos) can be passed at submit time, or exported as SBATCH_PARTITION
# etc. If compute nodes have no internet access, download the models on the login node first and submit with
# HF_HUB_OFFLINE=1; see slurm/download_models.sh.

set -euo pipefail

MODEL="${1:?usage: sbatch elicited_previsions.sbatch <hf-model-id>}"
MODEL_TAG="${MODEL##*/}"
BATCH_SIZE="${BATCH_SIZE:-16}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
OUT="${COHERENCE_STORE:-store}/001-dutch-book"  # same default as coherence.paths.STORE
FAMILIES="data/event_families/facts_families.json data/event_families/companies_families.json"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Job ${SLURM_JOB_ID:-?} on $(hostname): model=$MODEL batch_size=$BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this job'"

uv run python -m coherence.elicitation --model "$MODEL" --batch_size "$BATCH_SIZE"
# $FAMILIES is intentionally unquoted so it splits into one argument per file
uv run python -m coherence.dutch_book --families $FAMILIES \
    --previsions "$OUT/previsions/elicited_$MODEL_TAG.csv" \
    --output "$OUT/elicited_rates/$MODEL_TAG.csv"
