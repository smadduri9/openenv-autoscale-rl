from __future__ import annotations

import json
import sys
from pathlib import Path
from random import Random
from typing import List, Mapping, TypedDict
from uuid import uuid4

from .models import AutoscaleObservation, AutoscaleState, ObservationHistory, ResetResponse, StepResponse


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()
from simulator import AutoscaleSimConfig, AutoscaleSimulator  # noqa: E402


class TraceRecord(TypedDict):
    trace_id: str
    family: str
    rps: List[float]


class AutoscaleOpenEnv:
    def __init__(self, trace_path: Path, config: AutoscaleSimConfig, seed: int = 7) -> None:
        self.trace_path = trace_path
        self.config = config
        self.seed = seed
        self._rng = Random(seed)
        self._traces = self._load_traces(trace_path)
        self._sim: AutoscaleSimulator | None = None
        self._episode_id = ""
        self._trace: TraceRecord | None = None
        self._episode_seed = seed

    def _load_traces(self, path: Path) -> List[TraceRecord]:
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")
        rows: List[TraceRecord] = []
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                for key in ("trace_id", "family", "rps"):
                    if key not in raw:
                        raise ValueError(f"Missing key {key!r} at line {ln}")
                rows.append(
                    {
                        "trace_id": str(raw["trace_id"]),
                        "family": str(raw["family"]),
                        "rps": [float(v) for v in raw["rps"]],
                    }
                )
        if not rows:
            raise ValueError(f"No traces in {path}")
        return rows

    def _pick_trace(self, trace_id: str | None = None, trace_index: int | None = None) -> TraceRecord:
        if trace_id is not None:
            for trace in self._traces:
                if trace["trace_id"] == trace_id:
                    return trace
            raise ValueError(f"Unknown trace_id: {trace_id}")
        if trace_index is not None:
            if trace_index < 0 or trace_index >= len(self._traces):
                raise IndexError(f"trace_index out of range: {trace_index}")
            return self._traces[trace_index]
        return self._rng.choice(self._traces)

    def _to_observation(
        self,
        obs: Mapping[str, object],
        reward: float = 0.0,
        done: bool = False,
    ) -> AutoscaleObservation:
        history_raw = obs.get("history")
        history = None
        if isinstance(history_raw, Mapping):
            history = ObservationHistory(
                incoming_rps=[float(v) for v in history_raw.get("incoming_rps", [])],
                cpu_utilization=[float(v) for v in history_raw.get("cpu_utilization", [])],
                queue_depth=[float(v) for v in history_raw.get("queue_depth", [])],
                p95_latency_ms=[float(v) for v in history_raw.get("p95_latency_ms", [])],
                ready_pods=[float(v) for v in history_raw.get("ready_pods", [])],
            )
        return AutoscaleObservation(
            timestep=int(obs["timestep"]),
            incoming_rps=float(obs["incoming_rps"]),
            ready_pods=int(obs["ready_pods"]),
            pending_pods=int(obs["pending_pods"]),
            cpu_utilization=float(obs["cpu_utilization"]),
            queue_depth=float(obs["queue_depth"]),
            p95_latency_ms=float(obs["p95_latency_ms"]),
            error_rate=float(obs["error_rate"]),
            previous_action=str(obs["previous_action"]),
            reward=float(reward),
            done=bool(done),
            history=history,
        )

    def reset(self, seed: int | None = None, trace_id: str | None = None, trace_index: int | None = None) -> ResetResponse:
        self._trace = self._pick_trace(trace_id=trace_id, trace_index=trace_index)
        self._episode_seed = int(seed if seed is not None else self._rng.randint(0, 2**31 - 1))
        cfg = AutoscaleSimConfig(**{**self.config.__dict__, "episode_length": len(self._trace["rps"])})
        self._sim = AutoscaleSimulator(cfg, trace=self._trace["rps"], seed=self._episode_seed)
        obs = self._sim.reset()
        self._episode_id = str(uuid4())
        return ResetResponse(
            episode_id=self._episode_id,
            trace_id=self._trace["trace_id"],
            family=self._trace["family"],
            seed=self._episode_seed,
            observation=self._to_observation(obs),
        )

    def step(self, action: str) -> StepResponse:
        if self._sim is None or self._trace is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        obs, reward, done, info = self._sim.step(action)
        return StepResponse(
            episode_id=self._episode_id,
            trace_id=self._trace["trace_id"],
            family=self._trace["family"],
            reward=float(reward),
            done=bool(done),
            observation=self._to_observation(obs, reward=reward, done=done),
            info=info,
        )

    def state(self) -> AutoscaleState:
        if self._sim is None or self._trace is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        return AutoscaleState(
            episode_id=self._episode_id,
            trace_id=self._trace["trace_id"],
            family=self._trace["family"],
            seed=self._episode_seed,
            step_count=self._sim.timestep,
            done=self._sim.done,
            observation=self._to_observation(self._sim.get_observation(), done=self._sim.done),
            metrics=self._sim.get_metrics(),
            debug={"trace_len": len(self._trace["rps"])},
        )
