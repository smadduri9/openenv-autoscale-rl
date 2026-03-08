from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Dict, List

from simulator import AutoscaleSimConfig, AutoscaleSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one autoscaling simulator episode.")
    parser.add_argument("--trace-path", type=Path, default=Path("traces.jsonl"))
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--policy", choices=["hold", "random"], default="hold")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def load_trace(path: Path, index: int) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if index < 0 or index >= len(records):
        raise IndexError(f"--trace-index out of range: {index}")
    return records[index]


def choose_action(policy: str, rng: Random) -> str:
    return "hold" if policy == "hold" else rng.choice(list(AutoscaleSimulator.ACTION_TO_DELTA.keys()))


def maybe_plot(series: Dict[str, List[float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plot skipped: matplotlib unavailable ({exc})")
        return

    t = list(range(len(series["incoming_rps"])))
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(t, series["incoming_rps"], label="incoming_rps")
    axes[1].plot(t, series["ready_pods"], label="ready_pods")
    axes[2].plot(t, series["queue_depth"], label="queue_depth")
    axes[3].plot(t, series["p95_latency_ms"], label="p95_latency_ms")
    for ax in axes:
        ax.legend(loc="upper right")
    axes[3].set_xlabel("timestep")
    fig.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    row = load_trace(args.trace_path, args.trace_index)
    trace = list(row["rps"])
    sim = AutoscaleSimulator(AutoscaleSimConfig(episode_length=len(trace)), trace=trace, seed=args.seed)
    rng = Random(args.seed)
    obs = sim.reset()

    series: Dict[str, List[float]] = {
        "incoming_rps": [],
        "ready_pods": [],
        "queue_depth": [],
        "p95_latency_ms": [],
    }
    done = False
    print(f"Running trace_id={row['trace_id']} family={row['family']} policy={args.policy}")
    while not done:
        action = choose_action(args.policy, rng)
        obs, reward, done, _ = sim.step(action)
        print(
            f"t={obs['timestep']:>3} action={action:<12} "
            f"in={obs['incoming_rps']:>7.2f} ready={obs['ready_pods']:>2} "
            f"queue={obs['queue_depth']:>8.2f} lat={obs['p95_latency_ms']:>7.2f} "
            f"err={obs['error_rate']:>5.3f} reward={reward:>8.4f}"
        )
        series["incoming_rps"].append(float(obs["incoming_rps"]))
        series["ready_pods"].append(float(obs["ready_pods"]))
        series["queue_depth"].append(float(obs["queue_depth"]))
        series["p95_latency_ms"].append(float(obs["p95_latency_ms"]))

    print("\nFinal summary metrics:")
    for k, v in sorted(sim.get_metrics().items()):
        print(f"- {k}: {v}")
    if args.plot:
        maybe_plot(series)


if __name__ == "__main__":
    main()
