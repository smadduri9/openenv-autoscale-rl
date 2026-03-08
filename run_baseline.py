from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, TypedDict

from hpa_policy import HeuristicHPAPolicy, HeuristicHPAPolicyConfig
from simulator import AutoscaleSimConfig, AutoscaleSimulator


class TraceRecord(TypedDict):
    trace_id: str
    family: str
    rps: List[float]


AGG_KEYS: Sequence[str] = (
    "avg_reward",
    "avg_ready_pods",
    "total_scale_actions",
    "mean_latency_ms",
    "drop_fraction",
    "final_queue_depth",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline autoscaling policy over traces.")
    p.add_argument("--trace-path", type=Path, default=Path("traces.jsonl"))
    p.add_argument("--max-traces", type=int, default=0)
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--compare-random", action="store_true")
    p.add_argument("--export-trajectories", action="store_true")
    p.add_argument("--trajectory-output", type=Path, default=Path("expert_trajectories.jsonl"))
    return p.parse_args()


def load_traces(path: Path) -> List[TraceRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")
    out: List[TraceRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("trace_id", "family", "rps"):
                if key not in row:
                    raise ValueError(f"Missing key {key!r} at line {ln}")
            rps = [max(0.0, float(v)) for v in row["rps"]]
            if not rps:
                raise ValueError(f"Empty rps at line {ln}")
            out.append({"trace_id": str(row["trace_id"]), "family": str(row["family"]), "rps": rps})
    if not out:
        raise ValueError(f"No valid traces found in {path}")
    return out


def maybe_filter_traces(traces: Sequence[TraceRecord], families: Sequence[str] | None) -> List[TraceRecord]:
    if not families:
        return list(traces)
    keep = set(families)
    out = [t for t in traces if t["family"] in keep]
    if not out:
        raise ValueError(f"No traces matched families: {sorted(keep)}")
    return out


def evaluate_trace(
    trace: TraceRecord,
    policy_name: str,
    policy: HeuristicHPAPolicy | None,
    random_gen: Random | None,
) -> tuple[Dict[str, float | int], List[Dict[str, object]]]:
    sim = AutoscaleSimulator(
        AutoscaleSimConfig(episode_length=len(trace["rps"]), history_length=len(trace["rps"])),
        trace["rps"],
    )
    obs = sim.reset()
    if policy is not None:
        policy.reset()

    traj: List[Dict[str, object]] = []
    done = False
    while not done:
        if policy_name == "baseline":
            assert policy is not None
            action = policy.choose_action(obs)
        elif policy_name == "random":
            assert random_gen is not None
            action = random_gen.choice(list(AutoscaleSimulator.ACTION_TO_DELTA.keys()))
        else:
            raise ValueError(f"Unsupported policy_name: {policy_name}")
        before = copy.deepcopy(obs)
        next_obs, reward, done, _ = sim.step(action)
        traj.append(
            {
                "trace_id": trace["trace_id"],
                "family": trace["family"],
                "timestep": before["timestep"],
                "observation": before,
                "chosen_action": action,
                "reward": float(reward),
                "next_observation": copy.deepcopy(next_obs),
                "done": bool(done),
            }
        )
        obs = next_obs
    return sim.get_metrics(), traj


def aggregate_metrics(per_trace_metrics: Sequence[Mapping[str, float | int]]) -> Dict[str, float]:
    return {k: mean([float(m[k]) for m in per_trace_metrics]) if per_trace_metrics else 0.0 for k in AGG_KEYS}


def print_trace_metrics(name: str, trace: TraceRecord, metrics: Mapping[str, float | int]) -> None:
    print(
        f"[{name}] trace_id={trace['trace_id']} family={trace['family']} "
        f"avg_reward={float(metrics['avg_reward']):.4f} avg_ready_pods={float(metrics['avg_ready_pods']):.2f} "
        f"scales={int(metrics['total_scale_actions'])} mean_latency_ms={float(metrics['mean_latency_ms']):.2f} "
        f"drop_fraction={float(metrics['drop_fraction']):.4f} final_queue={float(metrics['final_queue_depth']):.2f}"
    )


def print_aggregate(title: str, agg: Mapping[str, float]) -> None:
    print(f"\n{title}")
    for key in AGG_KEYS:
        print(f"- {key}: {agg[key]:.6f}")


def export_trajectories(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            c += 1
    return c


def main() -> None:
    args = parse_args()
    traces = maybe_filter_traces(load_traces(args.trace_path), args.families)
    if args.max_traces > 0:
        traces = traces[: args.max_traces]

    policy = HeuristicHPAPolicy(HeuristicHPAPolicyConfig())
    baseline_metrics: List[Dict[str, float | int]] = []
    all_steps: List[Dict[str, object]] = []
    for trace in traces:
        metrics, traj = evaluate_trace(trace, "baseline", policy, None)
        baseline_metrics.append(metrics)
        all_steps.extend(traj)
        print_trace_metrics("baseline", trace, metrics)
    baseline_agg = aggregate_metrics(baseline_metrics)
    print_aggregate("Baseline aggregate metrics", baseline_agg)

    if args.compare_random:
        rng = Random(args.seed)
        random_metrics: List[Dict[str, float | int]] = []
        for trace in traces:
            metrics, _ = evaluate_trace(trace, "random", None, rng)
            random_metrics.append(metrics)
            print_trace_metrics("random", trace, metrics)
        random_agg = aggregate_metrics(random_metrics)
        print_aggregate("Random aggregate metrics", random_agg)
        print("\nBaseline vs Random (aggregate deltas, baseline-random)")
        for key in AGG_KEYS:
            print(f"- {key}: {baseline_agg[key] - random_agg[key]:.6f}")

    if args.export_trajectories:
        total = export_trajectories(args.trajectory_output, all_steps)
        print(
            f"\nExported {total} expert transitions to {args.trajectory_output} "
            f"with policy_config={json.dumps(asdict(policy.config))}"
        )


if __name__ == "__main__":
    main()
