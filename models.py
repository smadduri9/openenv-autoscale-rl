from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


AllowedAction = Literal["scale_down_2", "scale_down_1", "hold", "scale_up_1", "scale_up_2", "scale_up_4"]


class AutoscaleAction(BaseModel):
    action: AllowedAction


class ObservationHistory(BaseModel):
    incoming_rps: List[float] = Field(default_factory=list)
    cpu_utilization: List[float] = Field(default_factory=list)
    queue_depth: List[float] = Field(default_factory=list)
    p95_latency_ms: List[float] = Field(default_factory=list)
    ready_pods: List[float] = Field(default_factory=list)


class AutoscaleObservation(BaseModel):
    timestep: int
    incoming_rps: float
    ready_pods: int
    pending_pods: int
    cpu_utilization: float
    queue_depth: float
    p95_latency_ms: float
    error_rate: float
    previous_action: str
    reward: float = 0.0
    done: bool = False
    history: Optional[ObservationHistory] = None


class AutoscaleState(BaseModel):
    trace_id: str
    family: str
    seed: int
    step_count: int
    done: bool
    observation: AutoscaleObservation
    metrics: Dict[str, float | int]
    debug: Dict[str, object] = Field(default_factory=dict)


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
