from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.types import Action as OpenEnvAction
    from openenv.core.env_server.types import Observation as OpenEnvObservation
    from openenv.core.env_server.types import State as OpenEnvState
except Exception:  # pragma: no cover - allows local fallback without openenv-core installed
    class OpenEnvAction(BaseModel):
        metadata: Dict[str, object] = Field(default_factory=dict)

    class OpenEnvObservation(BaseModel):
        done: bool = False
        reward: float | int | bool | None = None
        metadata: Dict[str, object] = Field(default_factory=dict)

    class OpenEnvState(BaseModel):
        episode_id: str | None = None
        step_count: int = 0


AllowedAction = Literal[
    "scale_down_2",
    "scale_down_1",
    "hold",
    "scale_up_1",
    "scale_up_2",
    "scale_up_4",
    "enable_rate_limit",
    "disable_rate_limit",
    "rollback_release",
]


class AutoscaleAction(OpenEnvAction):
    action: AllowedAction


class ObservationHistory(BaseModel):
    incoming_rps: List[float] = Field(default_factory=list)
    cpu_utilization: List[float] = Field(default_factory=list)
    queue_depth: List[float] = Field(default_factory=list)
    p95_latency_ms: List[float] = Field(default_factory=list)
    ready_pods: List[float] = Field(default_factory=list)


class AutoscaleObservation(OpenEnvObservation):
    timestep: int
    incoming_rps: float
    ready_pods: int
    pending_pods: int
    cpu_utilization: float
    queue_depth: float
    p95_latency_ms: float
    error_rate: float
    previous_action: str
    rate_limit_enabled: bool = False
    bad_deploy_active: bool = False
    dependency_slowdown_active: bool = False
    rollback_pending_steps: int = 0
    reward: float = 0.0
    done: bool = False
    history: Optional[ObservationHistory] = None


class AutoscaleState(OpenEnvState):
    trace_id: str
    family: str
    seed: int
    step_count: int
    done: bool
    observation: AutoscaleObservation
    metrics: Dict[str, float | int]
    debug: Dict[str, object] = Field(default_factory=dict)


# Legacy HTTP compatibility request/response models.
class ResetRequest(BaseModel):
    seed: Optional[int] = None
    trace_id: Optional[str] = None
    trace_index: Optional[int] = None


class ResetResponse(BaseModel):
    episode_id: str
    trace_id: str
    family: str
    seed: int
    observation: AutoscaleObservation


class StepRequest(BaseModel):
    action: AutoscaleAction


class StepResponse(BaseModel):
    episode_id: str
    trace_id: str
    family: str
    reward: float
    done: bool
    observation: AutoscaleObservation
    info: Dict[str, float | int]


class StateResponse(BaseModel):
    episode_id: str
    state: AutoscaleState


class HealthResponse(BaseModel):
    ok: bool
    message: str
