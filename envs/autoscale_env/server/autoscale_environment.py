from __future__ import annotations

from pathlib import Path
from typing import Any

from ..environment import AutoscaleOpenEnv
from ..models import AutoscaleAction, AutoscaleObservation, AutoscaleState

try:
    from openenv.core.env_server.interfaces import Environment
except Exception:  # pragma: no cover - fallback for local usage without openenv-core
    class Environment:  # type: ignore[no-redef]
        pass


class AutoscaleEnvironment(Environment):
    """OpenEnv-compatible environment adapter over AutoscaleOpenEnv."""

    def __init__(self, trace_path: str = "traces.jsonl", seed: int = 7) -> None:
        from simulator import AutoscaleSimConfig

        self._env = AutoscaleOpenEnv(
            trace_path=Path(trace_path),
            config=AutoscaleSimConfig(),
            seed=seed,
        )
        self._last_state: AutoscaleState | None = None

    def reset(self, seed: int | None = None, **kwargs: Any) -> AutoscaleObservation:
        trace_id = kwargs.get("trace_id")
        trace_index = kwargs.get("trace_index")
        reset_resp = self._env.reset(seed=seed, trace_id=trace_id, trace_index=trace_index)
        self._last_state = self._env.state()
        return reset_resp.observation

    def step(self, action: AutoscaleAction) -> AutoscaleObservation:
        step_resp = self._env.step(action.action)
        self._last_state = self._env.state()
        return step_resp.observation

    @property
    def state(self) -> AutoscaleState:
        if self._last_state is not None:
            return self._last_state
        self._last_state = self._env.state()
        return self._last_state
