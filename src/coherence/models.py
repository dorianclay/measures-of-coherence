"""Load the exact weights a model reference names.

A model reference is `<hf-id>@<commit>`, `<local-path>@local`, or `<run-id>@step<N>`: an eval checkpoint of a
training run, a LoRA adapter on that run's own base model, which is loaded merged.
"""

from pathlib import Path

from coherence import runs


def resolve(model_ref: str) -> tuple[str, Path | None, str]:
    """The base model's local path, the adapter to merge into it (if any), and the base's Hugging Face id or path."""
    if runs.CHECKPOINT_REF.match(model_ref):
        adapter = runs.model_path(model_ref)
        run_id = model_ref.split("@")[0]
        base_path, _, base_name = resolve(runs.read(runs.find(run_id))["model_ref"])
        return base_path, adapter, base_name
    name, _, revision = model_ref.rpartition("@")
    if not name:
        raise ValueError(f"Model reference {model_ref!r} has no @revision; write <hf-id>@<commit> or <path>@local")
    if revision == "local":
        return name, None, name
    from huggingface_hub import snapshot_download
    return snapshot_download(name, revision=revision), None, name


def load(model_ref: str):
    """The model (in eval mode, adapter merged) and its tokenizer, set up as probing.activations.init_model does."""
    from probing.activations import init_model
    base_path, adapter, _ = resolve(model_ref)
    model, tokenizer = init_model(base_path)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    return model, tokenizer
