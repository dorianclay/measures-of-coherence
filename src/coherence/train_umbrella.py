"""Train a model with GRPO on the umbrella/rain task (experiment 002), as a training run that resumes across jobs.

    uv run submit-chain --segments 2 --experiment 003 --note "pilot, 1.5B, rl, seed 0" -- \
        python -m coherence.train_umbrella --model Qwen/Qwen2.5-1.5B-Instruct --seed 0 --arm rl --steps 500

Before training, the seed's whole dataset is generated to $RUN_STORE/dataset.jsonl and trained on in order. Every
completion's answer and reward goes to $RUN_STORE/rewards.jsonl. LoRA adapters are saved as eval checkpoints at step 0
and at each --eval-fractions of the steps ($RUN_STORE/eval/step<N>, model reference <run-id>@step<N>). Resume state
(adapter, optimizer, scheduler, RNG) is saved every --resume-every steps and when the job is about to run out of time,
after which the command exits 0 and the next segment continues from it.

The rl arm is rewarded with the utility of its answer in the drawn weather; the control arm with the utility of a
random action in the same weather, whatever it answered.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer

from coherence import runs, umbrella


def resolve_model(name: str, revision: str | None) -> tuple[str, str]:
    """Where to load the weights from, and the model reference naming them exactly (HF id@commit)."""
    if Path(name).exists():
        return name, f"{Path(name).resolve()}@local"
    from huggingface_hub import snapshot_download
    path = snapshot_download(name, revision=revision)
    return path, f"{name}@{Path(path).name}"


def replace_dir(source: Path, target: Path) -> None:
    """Move `source` to `target`, replacing it with one rename so a crash never leaves neither."""
    old = target.with_name(target.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    source.rename(target)
    shutil.rmtree(old, ignore_errors=True)


class Segment(TrainerCallback):
    """Ties the trainer to the run record: progress, eval checkpoints, resume state and the end of the job."""

    def __init__(self, args, store: Path, start_step: int):
        self.args, self.store, self.start_step = args, store, start_step
        self.eval_steps = {0} | {round(f * args.steps) for f in args.eval_fractions}

    def save_eval(self, model, step: int):
        target = self.store / "eval" / f"step{step}"
        tmp = target.with_name(target.name + ".tmp")
        model.save_pretrained(tmp)
        # PEFT writes target_modules from a set, in an order that changes between processes; sort it so identical
        # training gives identical checkpoint hashes
        config_path = tmp / "adapter_config.json"
        config = json.loads(config_path.read_text())
        if isinstance(config.get("target_modules"), list):
            config["target_modules"] = sorted(config["target_modules"])
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        replace_dir(tmp, target)
        runs.checkpoint(target, step=step)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.start_step == 0:
            self.save_eval(model, 0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        runs.progress(step)
        if step in self.eval_steps:
            self.save_eval(model, step)
        if step >= self.args.steps:
            return
        stop = ((self.args.stop_after and step - self.start_step >= self.args.stop_after)
                or runs.out_of_time(self.args.margin_minutes * 60))
        if stop or step % self.args.resume_every == 0:
            control.should_save = True
        if stop:
            control.should_training_stop = True

    def on_save(self, args, state, control, **kwargs):
        target = self.store / "resume"
        replace_dir(Path(args.output_dir) / f"checkpoint-{state.global_step}", target)
        runs.checkpoint(target, step=state.global_step, kind="resume")


def reward_function(samples: list[dict], arm: str, seed: int, log_path: Path):
    def reward(prompts, completions, index, trainer_state, **kwargs):
        rewards, seen = [], {}
        with open(log_path, "a") as log:
            for completion, i in zip(completions, index):
                text = completion[0]["content"] if isinstance(completion, list) else completion
                answer = umbrella.parse_answer(text)
                k = seen[i] = seen.get(i, -1) + 1
                if arm == "rl":
                    r = umbrella.reward(samples[i], answer)
                else:
                    r = umbrella.control_reward(samples[i], random.Random(f"control-{seed}-{i}-{k}"))
                rewards.append(float(r))
                log.write(json.dumps({"step": trainer_state.global_step + 1, "index": i, "completion": text,
                                      "answer": answer, "reward": r}) + "\n")
        return rewards
    reward.__name__ = f"umbrella_{arm}"
    return reward


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Hugging Face id or local path of the base model.")
    parser.add_argument("--revision", help="Hugging Face revision; default: the latest.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=["rl", "control"], required=True)
    parser.add_argument("--steps", type=int, required=True, help="Planned optimizer steps for the whole run.")
    parser.add_argument("--prompts-per-step", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=8, help="Completions per prompt (GRPO group).")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl", type=float, default=0.04, help="Weight of the KL penalty to the base model (beta).")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-completion-length", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-fractions", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--resume-every", type=int, default=50, help="Save resume state every this many steps.")
    parser.add_argument("--margin-minutes", type=float, default=30,
                        help="Save and stop when the job has less than this left.")
    parser.add_argument("--stop-after", type=int, help="Stop each segment after this many steps (for testing).")
    parser.add_argument("--output", help="Where to write outside `record`; under it, $RUN_STORE.")
    args = parser.parse_args(argv)

    store = Path(os.environ.get("RUN_STORE") or args.output or parser.error("run under `record`, or give --output"))
    store.mkdir(parents=True, exist_ok=True)
    model_path, model_ref = resolve_model(args.model, args.revision)
    dataset_path, log_path = store / "dataset.jsonl", store / "rewards.jsonl"
    resume = runs.resume_state()
    start_step = resume[1] if resume else 0

    if resume is None:
        umbrella.write_dataset(umbrella.generate(args.seed, args.steps * args.prompts_per_step), dataset_path)
        log_path.write_text("")
        runs.log(model_ref=model_ref, seed=args.seed, planned_steps=args.steps, artifacts=[dataset_path])
        if run_dir := runs.current_run_dir():
            (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    else:
        # Rewards logged after the resume state was saved belong to steps that will be trained again
        kept = [line for line in log_path.read_text().splitlines() if json.loads(line)["step"] <= start_step]
        log_path.write_text("".join(line + "\n" for line in kept))
    samples = umbrella.read_dataset(dataset_path)

    set_seed(args.seed)  # before the LoRA adapter's random initialization
    cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16 if cuda else torch.float32)
    config = GRPOConfig(
        output_dir=str(store / "trainer"), max_steps=args.steps, seed=args.seed,
        per_device_train_batch_size=args.prompts_per_step * args.group_size, num_generations=args.group_size,
        gradient_accumulation_steps=1, learning_rate=args.learning_rate, beta=args.kl,
        max_completion_length=args.max_completion_length, temperature=args.temperature, shuffle_dataset=False,
        save_strategy="no", logging_steps=1, report_to="none", bf16=cuda, use_cpu=not cuda)
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, args=config,
        reward_funcs=reward_function(samples, args.arm, args.seed, log_path),
        train_dataset=Dataset.from_list([{k: s[k] for k in ["index", "prompt"]} for s in samples]),
        peft_config=LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, target_modules="all-linear",
                               task_type="CAUSAL_LM"),
        callbacks=[Segment(args, store, start_step)])
    trainer.train(resume_from_checkpoint=str(resume[0]) if resume else None)
    runs.log(artifacts=[log_path])


if __name__ == "__main__":
    main()
