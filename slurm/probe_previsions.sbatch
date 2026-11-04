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
#   1. probing.activations extracts last-token activations for every dataset at 8 relative depths
#   2. probing.family_probes trains leakage-free probes (domain swap) and writes previsions on the event families
#   3. coherence.dutch_book books every family against every probe
#
# Usage (from the repo root; create the log dir once with `mkdir -p logs`):
#   sbatch slurm/probe_previsions.sbatch Qwen/Qwen2.5-7B-Instruct
#
# All eight models of the experiments, each fits one L40 (48GB) in bf16:
#   for m in Qwen/Qwen2.5-{1.5B,7B,14B}{,-Instruct} google/gemma-4-12B{,-it}; do
#       sbatch slurm/probe_previsions.sbatch "$m"; done
#
# Outputs, all under $COHERENCE_STORE/001-dutch-book/: activations/<model>/, previsions/<model>.csv and probe_rates/<model>.csv
# (large; stay on the cluster), plus the small probe_metrics/<model>.csv to copy into experiments/001-dutch-book/results/.
#
# Cluster-specific settings (partition, account, qos) can be passed at submit time. If compute nodes have no
# internet access, download the models on the login node first and submit with HF_HUB_OFFLINE=1; see
# slurm/download_models.sh.

set -euo pipefail

MODEL="${1:?usage: sbatch probe_previsions.sbatch <hf-model-id>}"
MODEL_TAG="${MODEL##*/}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MLP_SEEDS="${MLP_SEEDS:-5}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
OUT="${COHERENCE_STORE:-store}/001-dutch-book"  # same default as coherence.paths.STORE
FAMILIES="data/event_families/facts_families.json data/event_families/companies_families.json"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Job ${SLURM_JOB_ID:-?} on $(hostname): model=$MODEL batch_size=$BATCH_SIZE mlp_seeds=$MLP_SEEDS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this job'"

if [ ! -f "$OUT/activations/$MODEL_TAG/conj_neg_companies.npy" ]; then
    uv run python -m probing.activations --model "$MODEL" --batch_size "$BATCH_SIZE"
fi
uv run python -m probing.family_probes --model "$MODEL_TAG" --mlp_seeds "$MLP_SEEDS"
# $FAMILIES is intentionally unquoted so it splits into one argument per file
uv run python -m coherence.dutch_book --families $FAMILIES \
    --previsions "$OUT/previsions/$MODEL_TAG.csv" --output "$OUT/probe_rates/$MODEL_TAG.csv"
