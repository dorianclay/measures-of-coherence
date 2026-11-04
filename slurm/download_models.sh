#!/bin/bash
# Download the Dutch book experiments' models into the Hugging Face cache. Run on the LOGIN node (which has internet),
# from the repo root, before submitting jobs whose compute nodes are offline:
#
#   export HF_HOME=/path/to/scratch/huggingface   # optional; ~145GB total, default is ~/.cache/huggingface
#   bash slurm/download_models.sh                 # all eight models
#   bash slurm/download_models.sh Qwen/Qwen2.5-7B # or just some
#
# Then submit with the same HF_HOME exported plus HF_HUB_OFFLINE=1, so jobs read the cache and never touch the network
# (sbatch forwards the submitting shell's environment to the job):
#
#   export HF_HUB_OFFLINE=1
#   sbatch slurm/probe_previsions.sbatch google/gemma-4-12B-it
#
# Re-running is safe: files already in the cache are skipped, and interrupted downloads resume.

set -euo pipefail

MODELS=("$@")
if [ "${#MODELS[@]}" -eq 0 ]; then
    MODELS=(
        google/gemma-4-12B-it google/gemma-4-12B
        Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct
        Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct
        Qwen/Qwen2.5-14B Qwen/Qwen2.5-14B-Instruct
    )
fi

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "Cache: $HF_HOME"

# Also installs the project's Python packages, so `uv sync --frozen` in the jobs needs no internet either
uv sync --frozen

for model in "${MODELS[@]}"; do
    echo "== $model"
    uv run hf download "$model"
done

echo "Done. Cached models:"
du -sh "$HF_HOME"/hub/models--* 2>/dev/null || true
