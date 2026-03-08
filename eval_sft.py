from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from prompts import ACTIONS

LEGAL_ACTIONS = set(ACTIONS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SFT action model.")
    p.add_argument("--dataset-path", type=Path, default=Path("sft_data/val.jsonl"))
    p.add_argument("--model-path", type=str, default="sft_model")
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--print-examples", type=int, default=5)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_reference_action(row: Dict[str, object]) -> str:
    completion = row.get("completion")
    if not isinstance(completion, str):
        raise ValueError("Row missing string completion field.")
    return completion.strip()


def extract_prompt_text(row: Dict[str, object]) -> str:
    if isinstance(row.get("prompt"), str):
        return str(row["prompt"])
    if isinstance(row.get("text"), str):
        text = str(row["text"])
        return text.rsplit("\n", 1)[0]
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
    raise ValueError("Could not extract prompt text from row.")


def normalize_generated_output(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line.strip()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.dataset_path)[: args.max_examples]
    if not rows:
        raise RuntimeError("No evaluation rows loaded.")

    tokenizer = None
    model = None
    if not args.mock:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
        model.eval()

    valid_count = 0
    exact_count = 0
    predicted_actions: Counter[str] = Counter()
    for idx, row in enumerate(rows):
        prompt = extract_prompt_text(row)
        ref = extract_reference_action(row)

        if args.mock:
            raw_generated = ref
        else:
            assert tokenizer is not None and model is not None
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
            raw_generated = tokenizer.decode(gen_ids, skip_special_tokens=True)

        normalized_action = normalize_generated_output(raw_generated)
        is_valid = normalized_action in LEGAL_ACTIONS
        if is_valid:
            valid_count += 1
        if normalized_action == ref:
            exact_count += 1
        predicted_actions[normalized_action] += 1

        if idx < args.print_examples:
            print(f"\nExample {idx}")
            print(f"reference:  {ref}")
            print(f"raw output: {raw_generated!r}")
            print(f"normalized: {normalized_action!r}")

    total = len(rows)
    print("\nEvaluation summary")
    print(f"- total: {total}")
    print(f"- valid_action_rate: {valid_count / total:.4f}")
    print(f"- exact_match_accuracy: {exact_count / total:.4f}")
    print("- action_distribution:")
    for action, count in predicted_actions.most_common():
        print(f"  - {action}: {count}")


if __name__ == "__main__":
    main()
