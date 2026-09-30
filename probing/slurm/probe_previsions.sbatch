#!/bin/bash
#SBATCH --job-name=probe-previsions
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

# Probe previsions and their rates of loss for the Dutch book experiments, one model per job:
#   1. Generate_Activations.py extracts last-token activations for every dataset at 8 relative depths
#   2. Train_Family_Probes.py trains leakage-free probes (domain swap) and writes previsions on the event families
#   3. DutchBook.py books every family against every probe
#
# Usage (from the repo root; create the log dir once with `mkdir -p logs`):
#   sbatch slurm/probe_previsions.sbatch Qwen/Qwen2.5-7B-Instruct
#
# All eight models of the experiments, each fits one L40 (48GB) in bf16:
#   for m in Qwen/Qwen2.5-{1.5B,7B,14B}{,-Instruct} google/gemma-4-12B{,-it}; do
#       sbatch slurm/probe_previsions.sbatch "$m"; done
#
# Outputs: activations/<model>/ and results/dutch_book/previsions/<model>.csv (large; not committed), plus the small
# results/dutch_book/probe_metrics/<model>.csv and results/dutch_book/probe_rates/<model>.csv to copy back and commit.
#
# Cluster-specific settings (partition, account, qos) can be passed at submit time. If compute nodes have no
# internet access, warm the caches from the login node first:
#   uv sync --frozen
#   uv run hf download <model-id>

set -euo pipefail

MODEL="${1:?usage: sbatch probe_previsions.sbatch <hf-model-id>}"
MODEL_TAG="${MODEL##*/}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MLP_SEEDS="${MLP_SEEDS:-5}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Job ${SLURM_JOB_ID:-?} on $(hostname): model=$MODEL batch_size=$BATCH_SIZE mlp_seeds=$MLP_SEEDS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this job'"

if [ ! -f "activations/$MODEL_TAG/conj_neg_companies.npy" ]; then
    uv run Generate_Activations.py --model "$MODEL" --batch_size "$BATCH_SIZE"
fi
uv run Train_Family_Probes.py --model "$MODEL_TAG" --mlp_seeds "$MLP_SEEDS"
uv run DutchBook.py --families event_families/facts_families.json event_families/companies_families.json \
    --previsions "results/dutch_book/previsions/$MODEL_TAG.csv" --output "results/dutch_book/probe_rates/$MODEL_TAG.csv"
