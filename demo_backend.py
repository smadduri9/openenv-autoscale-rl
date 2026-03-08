from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from eval_policy import TextActionModel, load_traces
from hpa_policy import HeuristicHPAPolicy
from prompts import ACTIONS
from rollout import normalize_action_output
from simulator import AutoscaleSimConfig, AutoscaleSimulator


class TraceRecord(TypedDict):
    trace_id: str
    family: str
    rps: List[float]


@dataclass
class ReplayPolicyResult:
    policy_name: str
    total_reward: float
    frames: List[Dict[str, Any]]
    metrics: Dict[str, float | int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve side-by-side policy replay data for demo UI.")
    p.add_argument("--trace-path", type=Path, default=Path("traces.jsonl"))
    p.add_argument("--rl-model-path", type=str, default="rl_model_unsloth_grpo")
    p.add_argument("--output-json", type=Path, default=Path("demo/replays.json"))
    p.add_argument("--max-traces", type=int, default=8)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def _frame_from_obs(
    obs: Mapping[str, object],
    *,
    action: str,
    step_reward: float,
    cumulative_reward: float,
    info: Mapping[str, object],
) -> Dict[str, Any]:
    rate_limited = float(info.get("rate_limited", 0.0))
    incoming_rps = float(obs.get("incoming_rps", 0.0))
    error_rate = float(obs.get("error_rate", 0.0))
    return {
        "timestep": int(obs.get("timestep", 0)),
        "action": action,
        "step_reward": float(step_reward),
        "cumulative_reward": float(cumulative_reward),
        "incoming_rps": incoming_rps,
        "ready_pods": int(obs.get("ready_pods", 0)),
        "pending_pods": int(obs.get("pending_pods", 0)),
        "queue_depth": float(obs.get("queue_depth", 0.0)),
        "p95_latency_ms": float(obs.get("p95_latency_ms", 0.0)),
        "error_rate": error_rate,
        "drop_estimate": incoming_rps * error_rate + rate_limited,
        "rate_limit_enabled": bool(obs.get("rate_limit_enabled", False)),
        "bad_deploy_active": bool(obs.get("bad_deploy_active", False)),
        "dependency_slowdown_active": bool(obs.get("dependency_slowdown_active", False)),
        "rollback_pending_steps": int(obs.get("rollback_pending_steps", 0)),
    }


def _run_policy(trace: TraceRecord, pick_action: Callable[[Mapping[str, object]], str]) -> ReplayPolicyResult:
    sim = AutoscaleSimulator(
        config=AutoscaleSimConfig(episode_length=len(trace["rps"]), history_length=len(trace["rps"])),
        trace=trace["rps"],
        trace_family=trace["family"],
        seed=None,
    )
    obs = sim.reset()
    done = False
    cumulative_reward = 0.0
    frames: List[Dict[str, Any]] = []
    while not done:
        action = pick_action(obs)
        next_obs, reward, done, info = sim.step(action)
        cumulative_reward += float(reward)
        frames.append(
            _frame_from_obs(
                next_obs,
                action=action,
                step_reward=float(reward),
                cumulative_reward=cumulative_reward,
                info=info,
            )
        )
        obs = next_obs
    return ReplayPolicyResult(
        policy_name="unknown",
        total_reward=cumulative_reward,
        frames=frames,
        metrics=sim.get_metrics(),
    )


def _build_replays(
    traces: List[TraceRecord],
    *,
    rl_model: TextActionModel,
) -> Dict[str, Any]:
    heuristic = HeuristicHPAPolicy()
    out_traces: List[Dict[str, Any]] = []
    for trace in traces:
        heuristic.reset()

        def heuristic_pick(obs: Mapping[str, object]) -> str:
            return heuristic.choose_action(obs)

        def rl_pick(obs: Mapping[str, object]) -> str:
            raw = rl_model.predict_raw(obs)
            action = normalize_action_output(raw)
            return action if action in ACTIONS else "hold"

        h = _run_policy(trace, heuristic_pick)
        h.policy_name = "heuristic"
        r = _run_policy(trace, rl_pick)
        r.policy_name = "rl"
        out_traces.append(
            {
                "trace_id": trace["trace_id"],
                "family": trace["family"],
                "steps": min(len(h.frames), len(r.frames)),
                "heuristic": {
                    "total_reward": h.total_reward,
                    "metrics": h.metrics,
                    "frames": h.frames,
                },
                "rl": {
                    "total_reward": r.total_reward,
                    "metrics": r.metrics,
                    "frames": r.frames,
                },
            }
        )
    return {
        "version": 2,
        "description": "Side-by-side heuristic vs RL replays on identical traces",
        "actions": list(ACTIONS),
        "traces": out_traces,
    }


def build_replay_json(
    trace_path: Path,
    rl_model_path: str,
    output_json: Path,
    max_traces: int,
) -> Dict[str, Any]:
    traces = load_traces(trace_path)[: max(1, max_traces)]
    if not traces:
        raise RuntimeError("No traces available to build demo replays.")
    rl_model = TextActionModel(rl_model_path)
    payload = _build_replays(traces, rl_model=rl_model)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


def create_app(replay_json_path: Path) -> FastAPI:
    app = FastAPI(title="Autoscale Demo Backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/replays")
    def replays() -> Dict[str, Any]:
        with replay_json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    return app


def main() -> None:
    args = parse_args()
    if args.rebuild or not args.output_json.exists():
        print(f"[demo-backend] Building replay JSON at {args.output_json} ...")
        build_replay_json(
            trace_path=args.trace_path,
            rl_model_path=args.rl_model_path,
            output_json=args.output_json,
            max_traces=args.max_traces,
        )
        print("[demo-backend] Replay build complete.")
    app = create_app(args.output_json)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
