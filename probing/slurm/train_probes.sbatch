#!/bin/bash
#SBATCH --job-name=probes
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Train probes for the truth-probe pipeline.
#
# Usage (from the repo root; create the log dir once with `mkdir -p logs`)
#   sbatch slurm/train_probes.sbatch meta-llama/Llama-3.1-8B
#   sbatch slurm/train_probes.sbatch Qwen/Qwen3-14B facts companies
#   LAYERS="-1 -8 -16" BATCH_SIZE=64 sbatch slurm/train_probes.sbatch <model>
#
# One L40 (48GB) fits models up to ~20B in bf16. For bigger models request
# more GPUs and memory; the script shards automatically via device_map:
#   sbatch --gpus=4 --mem=256G --cpus-per-task=16 slurm/train_probes.sbatch meta-llama/Llama-3.3-70B
#
# Cluster-specific settings (partition, account, qos) can be passed at submit
# time, e.g. `sbatch --partition=gpu --account=<acct> ...`, or added as
# #SBATCH lines above.
#
# Gated models (e.g. Llama): export HF_TOKEN before submitting; sbatch
# forwards the environment to the job.
#
# If compute nodes have no internet access, warm the caches from the login
# node first:
#   uv sync --frozen
#   uv run hf download <model-id>

set -euo pipefail

MODEL="${1:?usage: sbatch train_probes.sbatch <hf-model-id> [dataset ...]}"
shift
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    # The six datasets from Levinstein & Herrmann
    DATASETS=(cities companies animals elements inventions facts)
fi
# Negative indices from the end; -1..-28 covers the full depth of a 32-layer 8B model
LAYERS=${LAYERS:-"-1 -4 -8 -12 -16 -20 -24 -28"}
BATCH_SIZE="${BATCH_SIZE:-32}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Model cache; point at scratch/project storage if home quota is tight
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

command -v uv >/dev/null || {
    echo "uv not found; install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}

echo "Job ${SLURM_JOB_ID:-?} on $(hostname): model=$MODEL datasets=${DATASETS[*]} layers=$LAYERS batch_size=$BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in this job'; print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count())"

# $LAYERS is intentionally unquoted so it splits into one argument per layer
uv run TrainProbes.py \
    --model "$MODEL" \
    --dataset_names "${DATASETS[@]}" \
    --layers $LAYERS
