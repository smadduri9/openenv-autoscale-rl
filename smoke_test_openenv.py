from __future__ import annotations

import argparse
from random import Random

from client import OpenEnvClient

VALID_ACTIONS = ("scale_down_2", "scale_down_1", "hold", "scale_up_1", "scale_up_2", "scale_up_4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test OpenEnv HTTP wrapper.")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--policy", choices=["random", "hold"], default="random")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--trace-index", type=int, default=None)
    return p.parse_args()


def pick_action(policy: str, rng: Random) -> str:
    return "hold" if policy == "hold" else rng.choice(VALID_ACTIONS)


def main() -> None:
    args = parse_args()
    rng = Random(args.seed)
    client = OpenEnvClient(args.base_url)

    print("health:", client.health().model_dump())
    reset = client.reset(seed=args.seed, trace_index=args.trace_index)
    print("reset:", reset.model_dump())

    done = reset.observation.done
    total_reward = 0.0
    steps = 0
    while not done:
        action = pick_action(args.policy, rng)
        resp = client.step(action)
        total_reward += resp.reward
        done = resp.done
        steps += 1
        print(
            f"step={resp.observation.timestep:>3} action={action:<12} "
            f"reward={resp.reward:>8.4f} queue={resp.observation.queue_depth:>8.2f} "
            f"lat={resp.observation.p95_latency_ms:>7.2f} done={resp.done}"
        )

    st = client.state()
    print(f"finished episode_id={st.episode_id} steps={steps} total_reward={total_reward:.4f}")
    print("state.metrics:", st.state.metrics)


if __name__ == "__main__":
    main()
