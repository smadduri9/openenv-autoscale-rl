from __future__ import annotations

import json
from typing import Any, Mapping, Type, TypeVar
from urllib import request

from pydantic import BaseModel

from .models import (
    AutoscaleAction,
    AutoscaleObservation,
    AutoscaleState,
    HealthResponse,
    ResetResponse,
    StateResponse,
    StepResponse,
)

T = TypeVar("T", bound=BaseModel)


class OpenEnvClient:
    """Legacy-compatible HTTP client used by existing scripts."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._episode_id = ""
        self._trace_id = "unknown_trace"
        self._family = "unknown_family"
        self._seed = 0

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _parse_model(self, model_cls: Type[T], raw: dict[str, Any]) -> T:
        return model_cls.model_validate(raw)

    def _normalize_reset_payload(self, raw: dict[str, Any], requested_seed: int | None) -> dict[str, Any]:
        if {"episode_id", "trace_id", "family", "seed", "observation"}.issubset(raw.keys()):
            return raw
        return {
            "episode_id": str(raw.get("episode_id", "legacy-episode")),
            "trace_id": str(raw.get("trace_id", "unknown_trace")),
            "family": str(raw.get("family", "unknown_family")),
            "seed": int(raw.get("seed", requested_seed if requested_seed is not None else 0)),
            "observation": raw.get("observation", {}),
        }

    def _normalize_step_payload(self, raw: dict[str, Any]) -> dict[str, Any]:
        if {"episode_id", "trace_id", "family", "reward", "done", "observation", "info"}.issubset(raw.keys()):
            return raw
        observation = raw.get("observation", {})
        return {
            "episode_id": str(raw.get("episode_id", self._episode_id or "legacy-episode")),
            "trace_id": str(raw.get("trace_id", self._trace_id)),
            "family": str(raw.get("family", self._family)),
            "reward": float(raw.get("reward", 0.0)),
            "done": bool(raw.get("done", False)),
            "observation": observation,
            "info": raw.get("info", {}),
        }

    def _normalize_health_payload(self, raw: dict[str, Any]) -> dict[str, Any]:
        if {"ok", "message"}.issubset(raw.keys()):
            return raw
        status = str(raw.get("status", "")).strip().lower()
        ok = bool(raw.get("ok", status in {"ok", "healthy", "ready"}))
        message = str(raw.get("message", raw.get("status", "health-check")))
        return {"ok": ok, "message": message}

    def _normalize_state_payload(self, raw: dict[str, Any]) -> dict[str, Any]:
        if {"episode_id", "state"}.issubset(raw.keys()):
            state_obj = raw.get("state", {})
            if isinstance(state_obj, Mapping) and {
                "trace_id",
                "family",
                "seed",
                "step_count",
                "done",
                "observation",
                "metrics",
            }.issubset(state_obj.keys()):
                return raw

        raw_state_obj = raw.get("state", raw)
        state_obj = dict(raw_state_obj) if isinstance(raw_state_obj, Mapping) else {}
        base_observation: dict[str, Any] = {
            "timestep": int(state_obj.get("timestep", 0)),
            "incoming_rps": float(state_obj.get("incoming_rps", 0.0)),
            "ready_pods": int(state_obj.get("ready_pods", 0)),
            "pending_pods": int(state_obj.get("pending_pods", 0)),
            "cpu_utilization": float(state_obj.get("cpu_utilization", 0.0)),
            "queue_depth": float(state_obj.get("queue_depth", 0.0)),
            "p95_latency_ms": float(state_obj.get("p95_latency_ms", 0.0)),
            "error_rate": float(state_obj.get("error_rate", 0.0)),
            "previous_action": str(state_obj.get("previous_action", "hold")),
            "reward": float(state_obj.get("reward", 0.0)),
            "done": bool(state_obj.get("done", False)),
        }
        raw_obs = state_obj.get("observation")
        if isinstance(raw_obs, Mapping):
            merged_obs = dict(base_observation)
            merged_obs.update(dict(raw_obs))
            observation = merged_obs
        else:
            observation = base_observation

        raw_metrics = state_obj.get("metrics")
        if isinstance(raw_metrics, Mapping):
            metrics = dict(raw_metrics)
        else:
            metrics = {}
            for key, value in state_obj.items():
                if isinstance(value, (int, float)) and key not in {"seed", "step_count", "done"}:
                    metrics[key] = value

        done = bool(state_obj.get("done", observation.get("done", False)))
        return {
            "episode_id": str(raw.get("episode_id", self._episode_id or "legacy-episode")),
            "state": {
                "episode_id": str(raw.get("episode_id", self._episode_id or "legacy-episode")),
                "trace_id": str(state_obj.get("trace_id", self._trace_id)),
                "family": str(state_obj.get("family", self._family)),
                "seed": int(state_obj.get("seed", self._seed)),
                "step_count": int(state_obj.get("step_count", observation.get("timestep", 0))),
                "done": done,
                "observation": observation,
                "metrics": metrics,
                "debug": dict(state_obj.get("debug", {}))
                if isinstance(state_obj.get("debug", {}), Mapping)
                else {},
            },
        }

    def health(self) -> HealthResponse:
        raw = self._request("GET", "/health")
        normalized = self._normalize_health_payload(raw)
        return self._parse_model(HealthResponse, normalized)

    def reset(self, seed: int | None = None, trace_id: str | None = None, trace_index: int | None = None) -> ResetResponse:
        payload: dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        if trace_id is not None:
            payload["trace_id"] = trace_id
        if trace_index is not None:
            payload["trace_index"] = trace_index
        raw = self._request("POST", "/reset", payload)
        normalized = self._normalize_reset_payload(raw, requested_seed=seed)
        parsed = self._parse_model(ResetResponse, normalized)
        self._episode_id = parsed.episode_id
        self._trace_id = parsed.trace_id
        self._family = parsed.family
        self._seed = parsed.seed
        return parsed

    def step(self, action: str) -> StepResponse:
        payload = {"action": {"action": action}}
        raw = self._request("POST", "/step", payload)
        normalized = self._normalize_step_payload(raw)
        parsed = self._parse_model(StepResponse, normalized)
        self._episode_id = parsed.episode_id
        self._trace_id = parsed.trace_id
        self._family = parsed.family
        return parsed

    def state(self) -> StateResponse:
        raw = self._request("GET", "/state")
        normalized = self._normalize_state_payload(raw)
        return self._parse_model(StateResponse, normalized)


try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except Exception:  # pragma: no cover - only available when openenv-core is installed
    StepResult = None
    EnvClient = None


if EnvClient is not None and StepResult is not None:

    class AutoscaleEnv(EnvClient[AutoscaleAction, AutoscaleObservation, AutoscaleState]):
        """Official OpenEnv client class for packaged environment usage."""

        def _step_payload(self, action: AutoscaleAction) -> dict[str, Any]:
            return {"action": action.action}

        def _parse_result(self, payload: dict[str, Any]) -> StepResult[AutoscaleObservation]:
            obs = AutoscaleObservation.model_validate(payload.get("observation", {}))
            return StepResult(
                observation=obs,
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: dict[str, Any]) -> AutoscaleState:
            return AutoscaleState.model_validate(payload)
