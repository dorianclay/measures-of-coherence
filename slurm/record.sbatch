#!/bin/bash
#SBATCH --job-name=record
#SBATCH --output=logs/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00

# One segment of a recorded run: `uv run record` with this job's arguments. submit-chain submits it, e.g.
#   uv run submit-chain --segments 4 --sbatch-args "--gpus=2 --time=24:00:00" --experiment 002 -- python -m ...
# The cluster caps a job at 48 GPU-hours, so --gpus times --time must stay within 48 h.
#
# Submit from the repo root (create the log dir once with `mkdir -p logs`). Cluster-specific settings (partition,
# account, qos) go in --sbatch-args. If compute nodes have no internet access, download models on the login node
# first and submit with HF_HUB_OFFLINE=1; see slurm/download_models.sh.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "Job ${SLURM_JOB_ID:-?} on $(hostname): record $*"
uv sync --frozen
# exec, so slurm's SIGTERM at the time limit reaches record, which passes it to the command
exec uv run record "$@"
