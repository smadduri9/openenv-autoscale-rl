from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from hpa_policy import HeuristicHPAPolicy
from prompts import ACTIONS
from rollout import normalize_action_output
from run_baseline import load_traces
from simulator import AutoscaleSimConfig, AutoscaleSimulator


class TraceRecord(TypedDict):
    trace_id: str
    family: str
    rps: List[float]


ReplayMode = Literal["heuristic", "rl", "compare"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge demo backend for autoscaling replay UI.")
    p.add_argument("--trace-path", type=Path, default=Path("traces.jsonl"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--rl-model-path", default="rl_model")
    p.add_argument("--replay-json", type=Path, default=Path("demo/replays.json"))
    p.add_argument("--max-traces", type=int, default=10)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--replay-only", action="store_true")
    return p.parse_args()


class RLPolicy:
    def __init__(self, model_path: str) -> None:
        from eval_policy import TextActionModel

        self.model = TextActionModel(model_path)

    def choose(self, observation: Mapping[str, object]) -> str:
        raw = self.model.predict_raw(observation)
        action = normalize_action_output(raw)
        return action if action in ACTIONS else "hold"


def run_policy_episode(trace: TraceRecord, chooser) -> Dict[str, Any]:  # noqa: ANN001
    sim = AutoscaleSimulator(
        AutoscaleSimConfig(episode_length=len(trace["rps"]), history_length=len(trace["rps"])),
        trace=trace["rps"],
        seed=7,
    )
    obs = sim.reset()
    done = False
    cumulative = 0.0
    frames: List[Dict[str, Any]] = []
    while not done:
        action = chooser(obs)
        next_obs, reward, done, _ = sim.step(action)
        cumulative += float(reward)
        frames.append(
            {
                "timestep": int(next_obs["timestep"]),
                "action": action,
                "step_reward": float(reward),
                "cumulative_reward": cumulative,
                "incoming_rps": float(next_obs["incoming_rps"]),
                "ready_pods": int(next_obs["ready_pods"]),
                "queue_depth": float(next_obs["queue_depth"]),
                "p95_latency_ms": float(next_obs["p95_latency_ms"]),
                "previous_action": str(next_obs["previous_action"]),
                "error_rate": float(next_obs["error_rate"]),
            }
        )
        obs = next_obs
    return {"frames": frames, "metrics": sim.get_metrics(), "total_reward": cumulative}


def build_replay_payload(trace_path: Path, rl_model_path: str, max_traces: int) -> Dict[str, Any]:
    traces = load_traces(trace_path)[: max(1, max_traces)]
    heuristic = HeuristicHPAPolicy()
    rl_policy = None
    rl_error: str | None = None
    try:
        rl_policy = RLPolicy(rl_model_path)
    except Exception as exc:  # pragma: no cover - optional model dependency
        rl_error = str(exc)

    replay_traces: List[Dict[str, Any]] = []
    for trace in traces:
        heuristic.reset()
        h = run_policy_episode(trace, heuristic.choose_action)
        record: Dict[str, Any] = {
            "trace_id": trace["trace_id"],
            "family": trace["family"],
            "length": len(trace["rps"]),
            "heuristic": h,
        }
        if rl_policy is not None:
            record["rl"] = run_policy_episode(trace, rl_policy.choose)
        replay_traces.append(record)

    return {
        "version": 1,
        "source": "autoscale-judge-demo",
        "rl_available": rl_policy is not None,
        "rl_error": rl_error,
        "traces": replay_traces,
    }


def load_replay_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Replay JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "traces" not in payload or not isinstance(payload["traces"], list):
        raise ValueError("Replay JSON must contain a top-level 'traces' list.")
    return payload


def save_replay_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def create_app(replay_payload: Mapping[str, Any]) -> FastAPI:
    traces = list(replay_payload.get("traces", []))
    global_rl_available = bool(replay_payload.get("rl_available", False))
    global_rl_error = replay_payload.get("rl_error")

    app = FastAPI(title="Autoscale Judge Demo Backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {"ok": True, "rl_available": global_rl_available}

    @app.get("/api/traces")
    def list_traces() -> Dict[str, object]:
        return {
            "traces": [
                {
                    "index": i,
                    "trace_id": t["trace_id"],
                    "family": t["family"],
                    "length": int(t.get("length", len(t.get("heuristic", {}).get("frames", [])))),
                    "rl_available": "rl" in t,
                }
                for i, t in enumerate(traces)
            ],
            "rl_available": global_rl_available,
            "rl_error": global_rl_error,
        }

    @app.get("/api/replay")
    def replay(trace_index: int = 0, mode: ReplayMode = "heuristic") -> Dict[str, Any]:
        try:
            if trace_index < 0 or trace_index >= len(traces):
                raise IndexError(f"trace_index out of range: {trace_index}")
            trace = traces[trace_index]
            payload: Dict[str, Any] = {
                "trace_id": trace["trace_id"],
                "family": trace["family"],
                "mode": mode,
                "heuristic": trace["heuristic"],
                "rl_available": "rl" in trace,
            }
            if mode in {"rl", "compare"}:
                if "rl" not in trace:
                    raise RuntimeError("RL replay missing for selected trace. Use replay JSON with RL data.")
                payload["rl"] = trace["rl"]
            return payload
        except (IndexError, RuntimeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> None:
    args = parse_args()
    if args.replay_only:
        replay_payload = load_replay_payload(args.replay_json)
    elif args.rebuild or not args.replay_json.exists():
        replay_payload = build_replay_payload(args.trace_path, args.rl_model_path, args.max_traces)
        save_replay_payload(args.replay_json, replay_payload)
    else:
        replay_payload = load_replay_payload(args.replay_json)
    app = create_app(replay_payload)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
