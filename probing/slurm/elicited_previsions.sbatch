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
#   1. Elicit_Previsions.py asks the model about every event family statement, each in its own prompt
#      (True/False logprobs over three templates; stated probabilities too for instruct models)
#   2. DutchBook.py books every family against every elicitation method and template
#
# Usage (from the repo root; create the log dir once with `mkdir -p logs`):
#   sbatch slurm/elicited_previsions.sbatch Qwen/Qwen2.5-7B-Instruct
#
# All eight models of the experiments:
#   for m in Qwen/Qwen2.5-{1.5B,7B,14B}{,-Instruct} google/gemma-4-12B{,-it}; do
#       sbatch slurm/elicited_previsions.sbatch "$m"; done
#
# Outputs: results/dutch_book/previsions/elicited_<model>.csv and results/dutch_book/elicited_rates/<model>.csv (large;
# gitignored, stay on the cluster), plus the small results/dutch_book/elicitation_metrics/<model>.csv to commit.
#
# Cluster-specific settings (partition, account, qos) can be passed at submit time, or exported as SBATCH_PARTITION
# etc. If compute nodes have no internet access, download the models on the login node first and submit with
# HF_HUB_OFFLINE=1; see slurm/download_models.sh.

set -euo pipefail

MODEL="${1:?usage: sbatch elicited_previsions.sbatch <hf-model-id>}"
MODEL_TAG="${MODEL##*/}"
BATCH_SIZE="${BATCH_SIZE:-16}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Job ${SLURM_JOB_ID:-?} on $(hostname): model=$MODEL batch_size=$BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this job'"

uv run Elicit_Previsions.py --model "$MODEL" --batch_size "$BATCH_SIZE"
uv run DutchBook.py --families event_families/facts_families.json event_families/companies_families.json \
    --previsions "results/dutch_book/previsions/elicited_$MODEL_TAG.csv" \
    --output "results/dutch_book/elicited_rates/$MODEL_TAG.csv"
