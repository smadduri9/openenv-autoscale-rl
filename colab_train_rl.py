from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Notebook-friendly HF TRL GRPO training entrypoint.")
    p.add_argument("--rollout-log", type=Path, default=Path("rl_rollouts.jsonl"))
    p.add_argument("--init-model", type=str, default="sft_model")
    p.add_argument("--output-dir", type=Path, default=Path("rl_model"))
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-completion-length", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def load_rollout_steps(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Rollout file not found: {path}")
    out: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for step in row.get("steps", []):
                if isinstance(step, dict):
                    out.append(step)
    return out


def build_reward_lookup(steps: List[Dict[str, object]]) -> Dict[Tuple[str, str], float]:
    rewards: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for s in steps:
        obs = s.get("observation")
        if not isinstance(obs, dict):
            continue
        prompt = (
            f"timestep={obs.get('timestep')} incoming_rps={obs.get('incoming_rps')} "
            f"ready_pods={obs.get('ready_pods')} queue_depth={obs.get('queue_depth')} "
            f"p95_latency_ms={obs.get('p95_latency_ms')} error_rate={obs.get('error_rate')}"
        )
        action = str(s.get("chosen_action", "hold"))
        reward = float(s.get("reward", 0.0))
        rewards[(prompt, action)].append(reward)
    return {k: sum(v) / len(v) for k, v in rewards.items()}


def completion_to_text(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(str(v) for v in x)
    return str(x)


def make_reward_func(lookup: Dict[Tuple[str, str], float]):
    def reward_func(prompts, completions, **kwargs):  # noqa: ANN001
        out = []
        for p, c in zip(prompts, completions):
            p_text = completion_to_text(p)
            c_text = completion_to_text(c).strip().splitlines()[0].strip()
            out.append(float(lookup.get((p_text, c_text), 0.0)))
        return out

    return reward_func


def main() -> None:
    args = parse_args()
    steps = load_rollout_steps(args.rollout_log)
    if not steps:
        raise RuntimeError("No rollout steps found for GRPO training.")

    reward_lookup = build_reward_lookup(steps)
    dataset = Dataset.from_dict({"prompt": [k[0] for k in reward_lookup.keys()]})
    tokenizer = AutoTokenizer.from_pretrained(args.init_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        seed=args.seed,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.init_model,
        reward_funcs=make_reward_func(reward_lookup),
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Saved RL model to {args.output_dir}")


if __name__ == "__main__":
    main()
