from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from client import OpenEnvClient
from rollout import EpisodeRollout, HeuristicPolicyAdapter, RandomPolicyAdapter, run_episode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal RL rollout + local debug trainer scaffold.")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--init-model", type=str, default="sft_model")
    p.add_argument("--output-dir", type=Path, default=Path("rl_model"))
    p.add_argument("--rollout-log", type=Path, default=Path("rl_rollouts.jsonl"))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--trace-index", type=int, default=None)
    p.add_argument("--backend", choices=["local_debug", "colab_grpo"], default="local_debug")
    p.add_argument("--policy", choices=["heuristic", "random"], default="heuristic")
    p.add_argument("--do-local-update", action="store_true")
    p.add_argument("--max-update-steps", type=int, default=20)
    p.add_argument("--reward-quantile", type=float, default=0.6)
    return p.parse_args()


def serialize_rollout(rollout: EpisodeRollout) -> Dict[str, object]:
    return {
        "episode_id": rollout.episode_id,
        "trace_id": rollout.trace_id,
        "family": rollout.family,
        "cumulative_reward": rollout.cumulative_reward,
        "invalid_output_count": rollout.invalid_output_count,
        "action_counts": rollout.action_counts,
        "steps": [
            {
                "timestep": s.timestep,
                "observation": s.observation,
                "chosen_action": s.chosen_action,
                "raw_action_text": s.raw_action_text,
                "normalized_action": s.normalized_action,
                "reward": s.reward,
                "next_observation": s.next_observation,
                "done": s.done,
            }
            for s in rollout.steps
        ],
    }


def write_rollouts(path: Path, rollouts: List[EpisodeRollout]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rollouts:
            f.write(json.dumps(serialize_rollout(r)) + "\n")


def maybe_local_reward_weighted_update(args: argparse.Namespace, rollouts: List[EpisodeRollout]) -> None:
    if not args.do_local_update:
        return
    rewards = [r.cumulative_reward for r in rollouts]
    threshold = sorted(rewards)[int(max(0, min(len(rewards) - 1, len(rewards) * args.reward_quantile)))]
    kept = [r for r in rollouts if r.cumulative_reward >= threshold]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "local_debug_update.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "note": "Placeholder local reward-weighted update artifact.",
                "init_model": args.init_model,
                "episodes_total": len(rollouts),
                "episodes_kept": len(kept),
                "threshold": threshold,
                "max_update_steps": args.max_update_steps,
            },
            f,
            indent=2,
        )
    print(f"Wrote local debug update artifact to {out}")


def main() -> None:
    args = parse_args()
    if args.backend == "colab_grpo":
        print("Use colab_train_rl.py in GPU Colab for strict HF TRL GRPO.")
        print("This local script remains a rollout/debug scaffold only.")
        return

    client = OpenEnvClient(args.base_url)
    policy = HeuristicPolicyAdapter() if args.policy == "heuristic" else RandomPolicyAdapter(seed=args.seed)

    rollouts: List[EpisodeRollout] = []
    for ep in range(args.episodes):
        rollout = run_episode(client, policy, seed=args.seed + ep, trace_index=args.trace_index)
        rollouts.append(rollout)
        print(
            f"episode={ep} trace={rollout.trace_id} reward={rollout.cumulative_reward:.4f} "
            f"invalid={rollout.invalid_output_count}"
        )
    write_rollouts(args.rollout_log, rollouts)

    print("\nAggregate rollout stats")
    print(f"- episodes: {len(rollouts)}")
    print(f"- avg_cumulative_reward: {mean([r.cumulative_reward for r in rollouts]):.6f}")
    print(f"- avg_invalid_outputs: {mean([r.invalid_output_count for r in rollouts]):.6f}")

    maybe_local_reward_weighted_update(args, rollouts)


if __name__ == "__main__":
    main()
