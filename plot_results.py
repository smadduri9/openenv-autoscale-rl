from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping

import matplotlib.pyplot as plt

from eval_policy import (
    aggregate_metrics,
    evaluate_heuristic,
    evaluate_model,
    load_traces,
    try_load_model,
)
from hpa_policy import HeuristicHPAPolicy
from prompts import ACTIONS
from rollout import normalize_action_output
from simulator import AutoscaleSimConfig, AutoscaleSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plotting utilities for autoscaling project.")
    parser.add_argument("--trace-path", type=Path, default=Path("traces.jsonl"))
    parser.add_argument("--sft-model-path", type=str, default="sft_model")
    parser.add_argument("--rl-model-path", type=str, default="rl_model")
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--max-traces", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    return parser.parse_args()


def run_single_trace_heuristic(trace: Mapping[str, object]) -> Dict[str, List[float]]:
    policy = HeuristicHPAPolicy()
    sim = AutoscaleSimulator(
        config=AutoscaleSimConfig(episode_length=len(trace["rps"]), history_length=len(trace["rps"])),
        trace=trace["rps"],
        trace_family=str(trace["family"]),
        seed=None,
    )
    obs = sim.reset()
    policy.reset()
    out = {"ready_pods": [], "queue_depth": [], "p95_latency_ms": []}
    done = False
    while not done:
        action = policy.choose_action(obs)
        obs, _, done, _ = sim.step(action)
        out["ready_pods"].append(float(obs["ready_pods"]))
        out["queue_depth"].append(float(obs["queue_depth"]))
        out["p95_latency_ms"].append(float(obs["p95_latency_ms"]))
    return out


def run_single_trace_model(trace: Mapping[str, object], text_model) -> Dict[str, List[float]]:  # noqa: ANN001
    sim = AutoscaleSimulator(
        config=AutoscaleSimConfig(episode_length=len(trace["rps"]), history_length=len(trace["rps"])),
        trace=trace["rps"],
        trace_family=str(trace["family"]),
        seed=None,
    )
    obs = sim.reset()
    out = {"ready_pods": [], "queue_depth": [], "p95_latency_ms": []}
    done = False
    while not done:
        raw = text_model.predict_raw(obs)
        action = normalize_action_output(raw)
        if action not in ACTIONS:
            action = "hold"
        obs, _, done, _ = sim.step(action)
        out["ready_pods"].append(float(obs["ready_pods"]))
        out["queue_depth"].append(float(obs["queue_depth"]))
        out["p95_latency_ms"].append(float(obs["p95_latency_ms"]))
    return out


def make_single_trace_plot(
    output_path: Path,
    trace_label: str,
    heuristic_series: Mapping[str, List[float]],
    sft_series: Mapping[str, List[float]],
    rl_series: Mapping[str, List[float]],
) -> None:
    t = list(range(1, len(heuristic_series["ready_pods"]) + 1))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, heuristic_series["ready_pods"], label="Heuristic", linewidth=2)
    axes[0].plot(t, sft_series["ready_pods"], label="SFT", linewidth=2, linestyle="--")
    axes[0].plot(t, rl_series["ready_pods"], label="RL", linewidth=2, linestyle=":")
    axes[0].set_ylabel("Ready Pods")
    axes[0].set_title(f"Single Trace Comparison ({trace_label})")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, heuristic_series["queue_depth"], label="Heuristic", linewidth=2)
    axes[1].plot(t, sft_series["queue_depth"], label="SFT", linewidth=2, linestyle="--")
    axes[1].plot(t, rl_series["queue_depth"], label="RL", linewidth=2, linestyle=":")
    axes[1].set_ylabel("Queue Depth")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, heuristic_series["p95_latency_ms"], label="Heuristic", linewidth=2)
    axes[2].plot(t, sft_series["p95_latency_ms"], label="SFT", linewidth=2, linestyle="--")
    axes[2].plot(t, rl_series["p95_latency_ms"], label="RL", linewidth=2, linestyle=":")
    axes[2].set_ylabel("P95 Latency (ms)")
    axes[2].set_xlabel("Timestep")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def make_aggregate_bar_chart(
    output_path: Path,
    heuristic_agg: Mapping[str, float],
    sft_agg: Mapping[str, float],
    rl_agg: Mapping[str, float],
) -> None:
    metrics = ["avg_reward", "mean_latency_ms", "drop_fraction", "total_scale_actions"]
    x = list(range(len(metrics)))
    width = 0.25
    heuristic_vals = [float(heuristic_agg[m]) for m in metrics]
    sft_vals = [float(sft_agg[m]) for m in metrics]
    rl_vals = [float(rl_agg[m]) for m in metrics]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_h = ax.bar([i - width for i in x], heuristic_vals, width=width, label="Heuristic")
    bars_s = ax.bar(x, sft_vals, width=width, label="SFT")
    bars_r = ax.bar([i + width for i in x], rl_vals, width=width, label="RL")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Aggregate Policy Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    add_bar_value_labels(ax, bars_h, metrics)
    add_bar_value_labels(ax, bars_s, metrics)
    add_bar_value_labels(ax, bars_r, metrics)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def add_bar_value_labels(ax, bars, metrics: List[str]) -> None:  # noqa: ANN001
    for idx, bar in enumerate(bars):
        h = bar.get_height()
        metric = metrics[idx] if idx < len(metrics) else ""
        if metric == "drop_fraction":
            label = f"{h:.2%}"
        else:
            label = f"{h:.4f}"
        y_offset = max(abs(h) * 0.02, 0.01)
        y = h + y_offset if h >= 0 else h - y_offset
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va=va, fontsize=9)


def make_aggregate_metric_subplots(
    output_path: Path,
    heuristic_agg: Mapping[str, float],
    sft_agg: Mapping[str, float],
    rl_agg: Mapping[str, float],
) -> None:
    metrics = ["avg_reward", "mean_latency_ms", "drop_fraction", "total_scale_actions"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    flat_axes = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = flat_axes[i]
        values = [float(heuristic_agg[metric]), float(sft_agg[metric]), float(rl_agg[metric])]
        bars = ax.bar(["Heuristic", "SFT", "RL"], values, width=0.55)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        add_bar_value_labels(ax, bars, [metric, metric])
        if metric == "drop_fraction":
            ax.set_ylabel("Fraction (and % labels)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_aggregate_summary_json(
    output_path: Path,
    heuristic_agg: Mapping[str, float],
    sft_agg: Mapping[str, float],
    rl_agg: Mapping[str, float],
) -> None:
    metrics = ["avg_reward", "mean_latency_ms", "drop_fraction", "total_scale_actions"]
    payload = {
        "heuristic": {m: float(heuristic_agg[m]) for m in metrics},
        "sft": {m: float(sft_agg[m]) for m in metrics},
        "rl": {m: float(rl_agg[m]) for m in metrics},
        "formatted": {
            "heuristic": {"drop_fraction_pct": f"{float(heuristic_agg['drop_fraction']):.2%}"},
            "sft": {"drop_fraction_pct": f"{float(sft_agg['drop_fraction']):.2%}"},
            "rl": {"drop_fraction_pct": f"{float(rl_agg['drop_fraction']):.2%}"},
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    traces = load_traces(args.trace_path)
    if not traces:
        raise RuntimeError("No traces available.")
    if args.trace_index < 0 or args.trace_index >= len(traces):
        raise IndexError(f"trace-index out of range: {args.trace_index}")

    sft_model = try_load_model(args.sft_model_path, "SFT")
    rl_model = try_load_model(args.rl_model_path, "RL")
    if sft_model is None or rl_model is None:
        raise RuntimeError("Both SFT and RL models must be loadable for triplet comparison plots.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    single_trace = traces[args.trace_index]
    heuristic_series = run_single_trace_heuristic(single_trace)
    sft_series = run_single_trace_model(single_trace, sft_model)
    rl_series = run_single_trace_model(single_trace, rl_model)
    single_path = args.output_dir / "single_trace_heuristic_vs_sft_vs_rl.png"
    make_single_trace_plot(
        output_path=single_path,
        trace_label=f"{single_trace['trace_id']} / {single_trace['family']}",
        heuristic_series=heuristic_series,
        sft_series=sft_series,
        rl_series=rl_series,
    )

    limited_traces = traces[: args.max_traces] if args.max_traces > 0 else traces
    heuristic_metrics = [evaluate_heuristic(t) for t in limited_traces]
    sft_metrics = [evaluate_model(t, sft_model) for t in limited_traces]
    rl_metrics = [evaluate_model(t, rl_model) for t in limited_traces]
    heuristic_agg = aggregate_metrics(heuristic_metrics)
    sft_agg = aggregate_metrics(sft_metrics)
    rl_agg = aggregate_metrics(rl_metrics)

    agg_path = args.output_dir / "aggregate_heuristic_vs_sft_vs_rl.png"
    make_aggregate_bar_chart(
        output_path=agg_path,
        heuristic_agg=heuristic_agg,
        sft_agg=sft_agg,
        rl_agg=rl_agg,
    )
    agg_subplots_path = args.output_dir / "aggregate_heuristic_vs_sft_vs_rl_subplots.png"
    make_aggregate_metric_subplots(
        output_path=agg_subplots_path,
        heuristic_agg=heuristic_agg,
        sft_agg=sft_agg,
        rl_agg=rl_agg,
    )
    summary_path = args.output_dir / "aggregate_heuristic_vs_sft_vs_rl_summary.json"
    save_aggregate_summary_json(
        output_path=summary_path,
        heuristic_agg=heuristic_agg,
        sft_agg=sft_agg,
        rl_agg=rl_agg,
    )

    print(f"Saved single-trace plot: {single_path}")
    print(f"Saved aggregate plot: {agg_path}")
    print(f"Saved aggregate subplot figure: {agg_subplots_path}")
    print(f"Saved aggregate summary file: {summary_path}")


if __name__ == "__main__":
    main()
