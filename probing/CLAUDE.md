# Probing

This directory is a self-contained subproject: probes trained on LLM activations, plus the Dutch book experiments built on top of them. It came from the fork of `balevinstein/Probes` (MIT, see `LICENSE`) with its history intact.

Domain language is in `CONTEXT.md`; use its terms.

## Running anything

Run every command from this directory, not from the parent repo's root. Scripts, configs, tests and slurm jobs all resolve paths like `datasets/`, `event_families/` and `results/` relative to the working directory. This project has its own `pyproject.toml`, `uv.lock` and `.venv`: `uv sync`, then `uv run pytest`.

Submit slurm jobs from here too (`sbatch slurm/<job>.sbatch …`), since `#SBATCH --output=logs/…` resolves against the submit directory.

## Dutch book experiments

All 12 tickets are resolved; the write-up is `Dutch_Book_Write_Up.ipynb`. The record of every decision is a wayfinder map at `.scratch/dutch-book/map.md` with tickets in `.scratch/dutch-book/issues/`. Read the map before touching that work; its **Resuming** note says what lives where, including the large outputs that are not in git.

Keep result files larger than a few MB out of git: add them to `.gitignore` and record their location on the ticket instead.
