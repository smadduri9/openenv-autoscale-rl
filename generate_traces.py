from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from random import Random
from typing import Callable, Dict, List


TraceGenerator = Callable[[int, Random], List[float]]


def _clamp_non_negative(values: List[float]) -> List[float]:
    return [max(0.0, round(v, 3)) for v in values]


def gen_steady(length: int, rng: Random) -> List[float]:
    baseline = rng.uniform(40.0, 160.0)
    noise_span = baseline * rng.uniform(0.03, 0.10)
    return _clamp_non_negative([baseline + rng.uniform(-noise_span, noise_span) for _ in range(length)])


def gen_sustained_spike(length: int, rng: Random) -> List[float]:
    baseline = rng.uniform(25.0, 80.0)
    spike_mult = rng.uniform(2.0, 4.0)
    start = rng.randint(max(1, length // 6), max(1, length // 3))
    duration = rng.randint(max(4, length // 6), max(6, length // 2))
    end = min(length, start + duration)
    out: List[float] = []
    for t in range(length):
        level = baseline * spike_mult if start <= t < end else baseline
        out.append(level + rng.uniform(-0.06 * baseline, 0.06 * baseline))
    return _clamp_non_negative(out)


def gen_transient_spike(length: int, rng: Random) -> List[float]:
    baseline = rng.uniform(30.0, 100.0)
    spike_height = baseline * rng.uniform(2.5, 5.0)
    peak_t = rng.randint(max(1, length // 5), max(1, (4 * length) // 5))
    spike_width = rng.randint(2, max(3, length // 12))
    out: List[float] = []
    for t in range(length):
        distance = abs(t - peak_t)
        spike_component = max(0.0, 1.0 - (distance / max(spike_width, 1)))
        level = baseline + spike_height * spike_component
        out.append(level + rng.uniform(-0.05 * baseline, 0.05 * baseline))
    return _clamp_non_negative(out)


def gen_cyclical(length: int, rng: Random) -> List[float]:
    baseline = rng.uniform(50.0, 140.0)
    amplitude = baseline * rng.uniform(0.2, 0.8)
    period = rng.uniform(max(8.0, length / 6.0), max(10.0, length / 2.0))
    phase = rng.uniform(0.0, 2.0 * math.pi)
    out: List[float] = []
    for t in range(length):
        sine = math.sin((2.0 * math.pi * t / period) + phase)
        level = baseline + amplitude * sine
        out.append(level + rng.uniform(-0.08 * baseline, 0.08 * baseline))
    return _clamp_non_negative(out)


def gen_drifting_noisy(length: int, rng: Random) -> List[float]:
    start = rng.uniform(30.0, 120.0)
    drift_per_step = rng.uniform(-0.5, 1.2)
    noise_scale = max(2.0, 0.08 * start)
    out: List[float] = []
    value = start
    for _ in range(length):
        value += drift_per_step + rng.uniform(-noise_scale, noise_scale)
        out.append(value)
    return _clamp_non_negative(out)


def gen_traffic_spike(length: int, rng: Random) -> List[float]:
    baseline = rng.uniform(35.0, 90.0)
    out: List[float] = []
    spike_starts = sorted({rng.randint(5, max(6, length - 10)) for _ in range(2)})
    spike_len = max(4, length // 10)
    for t in range(length):
        level = baseline
        for start in spike_starts:
            if start <= t < min(length, start + spike_len):
                level *= rng.uniform(2.5, 4.5)
        out.append(level + rng.uniform(-0.08 * baseline, 0.08 * baseline))
    return _clamp_non_negative(out)


def gen_bad_deploy(length: int, rng: Random) -> List[float]:
    baseline = rng.uniform(60.0, 140.0)
    drift = rng.uniform(-0.1, 0.4)
    out: List[float] = []
    value = baseline
    for _ in range(length):
        value += drift + rng.uniform(-0.04 * baseline, 0.04 * baseline)
        out.append(value)
    return _clamp_non_negative(out)


def gen_dependency_slowdown(length: int, rng: Random) -> List[float]:
    baseline = rng.uniform(45.0, 120.0)
    out: List[float] = []
    for t in range(length):
        pulse = 1.0 + 0.3 * math.sin(2.0 * math.pi * t / max(12.0, length / 4.0))
        level = baseline * pulse
        out.append(level + rng.uniform(-0.05 * baseline, 0.05 * baseline))
    return _clamp_non_negative(out)


FAMILIES: Dict[str, TraceGenerator] = {
    "steady": gen_steady,
    "sustained_spike": gen_sustained_spike,
    "transient_spike": gen_transient_spike,
    "cyclical": gen_cyclical,
    "drifting_noisy": gen_drifting_noisy,
    "traffic_spike": gen_traffic_spike,
    "bad_deploy": gen_bad_deploy,
    "dependency_slowdown": gen_dependency_slowdown,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic autoscaling workload traces.")
    parser.add_argument("--num-traces", type=int, default=20, help="Number of traces to generate.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--episode-length", type=int, default=120, help="Length of each trace.")
    parser.add_argument("--seed", type=int, default=7, help="Global RNG seed.")
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(FAMILIES.keys()),
        choices=list(FAMILIES.keys()),
        help="Trace families to sample from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_traces <= 0:
        raise ValueError("--num-traces must be > 0")
    if args.episode_length <= 0:
        raise ValueError("--episode-length must be > 0")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = Random(args.seed)
    families = list(args.families)

    with args.output.open("w", encoding="utf-8") as f:
        for idx in range(args.num_traces):
            family = families[idx % len(families)]
            generator = FAMILIES[family]
            trace_seed = rng.randint(0, 2**31 - 1)
            trace_rng = Random(trace_seed)
            rps = generator(args.episode_length, trace_rng)
            record = {"trace_id": f"{family}_{idx:04d}", "family": family, "rps": rps}
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {args.num_traces} traces to {args.output}")


if __name__ == "__main__":
    main()
