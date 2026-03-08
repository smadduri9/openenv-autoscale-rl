from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from random import Random
from typing import Dict, List, Sequence

from prompts import build_chat_messages_example, build_plain_text_example


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SFT dataset from expert trajectories.")
    p.add_argument("--input", type=Path, default=Path("expert_trajectories.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("sft_data"))
    p.add_argument("--format", choices=["plain_text", "chat_messages"], default="plain_text")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rebalance", action="store_true")
    p.add_argument("--hold-cap", type=int, default=500)
    return p.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Input trajectory file not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("No rows loaded from trajectories file.")
    return rows


def build_examples(rows: Sequence[Dict[str, object]], fmt: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in rows:
        observation = row["observation"]
        action = str(row["chosen_action"])
        if fmt == "plain_text":
            ex = build_plain_text_example(observation, action)
            out.append({"prompt": ex["text"], "completion": action, "chosen_action": action})
        else:
            ex = build_chat_messages_example(observation, action)
            out.append({"messages": ex["messages"], "completion": action, "chosen_action": action})
    return out


def class_counts(rows: Sequence[Dict[str, object]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for row in rows:
        c[str(row.get("chosen_action", ""))] += 1
    return c


def print_class_counts(title: str, rows: Sequence[Dict[str, object]]) -> None:
    counts = class_counts(rows)
    print(f"\n{title}")
    for k in sorted(counts):
        print(f"- {k}: {counts[k]}")


def rebalance_rows(rows: Sequence[Dict[str, object]], hold_cap: int, rng: Random) -> List[Dict[str, object]]:
    if hold_cap < 0:
        raise ValueError("--hold-cap must be >= 0")
    hold_rows = [r for r in rows if str(r.get("chosen_action")) == "hold"]
    keep_rows = [r for r in rows if str(r.get("chosen_action")) != "hold"]
    rng.shuffle(hold_rows)
    return keep_rows + hold_rows[:hold_cap]


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1).")

    raw = read_jsonl(args.input)
    rows = build_examples(raw, args.format)
    rng = Random(args.seed)
    rng.shuffle(rows)
    print_class_counts("Class counts before rebalancing", rows)

    if args.rebalance:
        rows = rebalance_rows(rows, hold_cap=args.hold_cap, rng=rng)
        rng.shuffle(rows)
        print_class_counts("Class counts after rebalancing", rows)

    n_val = int(len(rows) * args.val_ratio)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    val_path = args.output_dir / "val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    print(f"Wrote train={len(train_rows)} rows to {train_path}")
    print(f"Wrote val={len(val_rows)} rows to {val_path}")


if __name__ == "__main__":
    main()
