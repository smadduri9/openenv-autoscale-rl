from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Deque, Dict, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class AutoscaleSimConfig:
    episode_length: int = 120
    min_pods: int = 1
    max_pods: int = 20
    initial_pods: int = 2
    pod_capacity_rps: float = 40.0
    startup_delay_steps: int = 3
    max_queue_depth: float = 10000.0
    latency_slo_ms: float = 200.0
    base_latency_ms: float = 50.0
    latency_per_queue_ratio_ms: float = 250.0
    queue_error_threshold: float = 250.0
    queue_error_saturation: float = 2000.0
    max_error_rate: float = 1.0
    history_length: int = 5
    w_slo: float = 2.0
    w_error: float = 4.0
    w_cost: float = 0.05
    w_flap: float = 0.10
    w_queue: float = 0.001


class AutoscaleSimulator:
    ACTION_TO_DELTA: Mapping[str, int] = {
        "scale_down_2": -2,
        "scale_down_1": -1,
        "hold": 0,
        "scale_up_1": 1,
        "scale_up_2": 2,
        "scale_up_4": 4,
    }

    def __init__(
        self,
        config: AutoscaleSimConfig,
        trace: Sequence[float],
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.trace = [max(0.0, float(v)) for v in trace]
        self.seed = seed
        self._rng = Random(seed)
        self._validate_inputs()
        self.max_steps = min(self.config.episode_length, len(self.trace))

        self.timestep = 0
        self.incoming_rps = self.trace[0] if self.trace else 0.0
        self.ready_pods = self.config.initial_pods
        self._pending_timers: List[int] = []
        self.cpu_utilization = 0.0
        self.queue_depth = 0.0
        self.p95_latency_ms = self.config.base_latency_ms
        self.error_rate = 0.0
        self.previous_action = "hold"
        self.total_cost_pod_steps = 0.0
        self.total_scale_actions = 0
        self.requests_served = 0.0
        self.requests_dropped = 0.0
        self.done = False
        self._step_rewards: List[float] = []
        self._history: Dict[str, Deque[float]] = {}
        self._init_history()

    def _validate_inputs(self) -> None:
        c = self.config
        if c.episode_length <= 0:
            raise ValueError("episode_length must be > 0")
        if c.min_pods < 0:
            raise ValueError("min_pods must be >= 0")
        if c.max_pods < c.min_pods:
            raise ValueError("max_pods must be >= min_pods")
        if not (c.min_pods <= c.initial_pods <= c.max_pods):
            raise ValueError("initial_pods must satisfy min_pods <= initial_pods <= max_pods")
        if c.pod_capacity_rps <= 0:
            raise ValueError("pod_capacity_rps must be > 0")
        if c.startup_delay_steps < 0:
            raise ValueError("startup_delay_steps must be >= 0")
        if c.latency_slo_ms <= 0:
            raise ValueError("latency_slo_ms must be > 0")
        if c.history_length < 0:
            raise ValueError("history_length must be >= 0")
        if not self.trace:
            raise ValueError("trace must contain at least one timestep")

    def _init_history(self) -> None:
        keys = (
            "incoming_rps",
            "cpu_utilization",
            "queue_depth",
            "p95_latency_ms",
            "ready_pods",
        )
        self._history = {
            k: deque(maxlen=self.config.history_length) for k in keys
        }
        self._record_history()

    def reset(self) -> Dict[str, float | int | str | Dict[str, List[float]]]:
        self._rng = Random(self.seed)
        self.timestep = 0
        self.incoming_rps = self.trace[0]
        self.ready_pods = self.config.initial_pods
        self._pending_timers = []
        self.cpu_utilization = 0.0
        self.queue_depth = 0.0
        self.p95_latency_ms = self.config.base_latency_ms
        self.error_rate = 0.0
        self.previous_action = "hold"
        self.total_cost_pod_steps = 0.0
        self.total_scale_actions = 0
        self.requests_served = 0.0
        self.requests_dropped = 0.0
        self.done = False
        self._step_rewards = []
        self._init_history()
        return self.get_observation()

    def step(
        self, action: str
    ) -> Tuple[Dict[str, float | int | str | Dict[str, List[float]]], float, bool, Dict[str, float | int]]:
        if self.done:
            raise RuntimeError("Cannot call step() when episode is done. Call reset() first.")
        if action not in self.ACTION_TO_DELTA:
            allowed = ", ".join(self.ACTION_TO_DELTA.keys())
            raise ValueError(f"Invalid action: {action!r}. Allowed actions: {allowed}")

        requested_delta = self.ACTION_TO_DELTA[action]
        applied_delta = self._apply_scaling(requested_delta)
        self._advance_pending()

        self.incoming_rps = self.trace[self.timestep]
        capacity = self.ready_pods * self.config.pod_capacity_rps
        queue_prev = self.queue_depth
        total_demand = self.incoming_rps + queue_prev
        served = min(total_demand, capacity)
        self.queue_depth = max(0.0, total_demand - served)
        if self.config.max_queue_depth > 0:
            self.queue_depth = min(self.queue_depth, self.config.max_queue_depth)

        if capacity <= 0:
            self.cpu_utilization = 0.0
        else:
            effective_load = min(total_demand, capacity)
            self.cpu_utilization = min(1.0, effective_load / capacity)

        queue_ratio_den = capacity if capacity > 0 else self.config.pod_capacity_rps
        queue_ratio = self.queue_depth / max(queue_ratio_den, 1e-6)
        self.p95_latency_ms = self.config.base_latency_ms + (
            self.config.latency_per_queue_ratio_ms * queue_ratio
        )

        over_threshold = max(0.0, self.queue_depth - self.config.queue_error_threshold)
        self.error_rate = min(
            self.config.max_error_rate,
            over_threshold / max(self.config.queue_error_saturation, 1e-6),
        )

        self.requests_served += served * (1.0 - self.error_rate)
        self.requests_dropped += self.incoming_rps * self.error_rate
        self.total_cost_pod_steps += float(self.ready_pods)
        if applied_delta != 0:
            self.total_scale_actions += 1

        slo_violation = max(0.0, self.p95_latency_ms - self.config.latency_slo_ms) / self.config.latency_slo_ms
        reward = (
            -self.config.w_slo * slo_violation
            -self.config.w_error * self.error_rate
            -self.config.w_cost * float(self.ready_pods)
            -self.config.w_flap * abs(float(applied_delta))
            -self.config.w_queue * self.queue_depth
        )
        if not isfinite(reward):
            raise RuntimeError("Reward became non-finite; check simulator configuration.")
        self._step_rewards.append(reward)
        self.previous_action = action
        self._record_history()

        self.timestep += 1
        self.done = self.timestep >= self.max_steps
        obs = self.get_observation()
        info: Dict[str, float | int] = {
            "capacity_rps": capacity,
            "served": served,
            "slo_violation": slo_violation,
            "action_delta_applied": applied_delta,
            "action_delta_requested": requested_delta,
            "queue_depth_prev": queue_prev,
        }
        return obs, reward, self.done, info

    def _apply_scaling(self, requested_delta: int) -> int:
        if requested_delta < 0:
            scale_down = min(-requested_delta, self.ready_pods - self.config.min_pods)
            self.ready_pods -= scale_down
            return -scale_down

        if requested_delta > 0:
            current_total = self.ready_pods + len(self._pending_timers)
            available = self.config.max_pods - current_total
            scale_up = min(requested_delta, max(0, available))
            self._pending_timers.extend([self.config.startup_delay_steps] * scale_up)
            return scale_up

        return 0

    def _advance_pending(self) -> None:
        if not self._pending_timers:
            return
        next_timers: List[int] = []
        promoted = 0
        for timer in self._pending_timers:
            timer -= 1
            if timer <= 0:
                promoted += 1
            else:
                next_timers.append(timer)
        self.ready_pods += promoted
        self._pending_timers = next_timers

    def _record_history(self) -> None:
        if self.config.history_length == 0:
            return
        self._history["incoming_rps"].append(float(self.incoming_rps))
        self._history["cpu_utilization"].append(float(self.cpu_utilization))
        self._history["queue_depth"].append(float(self.queue_depth))
        self._history["p95_latency_ms"].append(float(self.p95_latency_ms))
        self._history["ready_pods"].append(float(self.ready_pods))

    def get_observation(self) -> Dict[str, float | int | str | Dict[str, List[float]]]:
        obs: Dict[str, float | int | str | Dict[str, List[float]]] = {
            "timestep": self.timestep,
            "incoming_rps": float(self.incoming_rps),
            "ready_pods": self.ready_pods,
            "pending_pods": len(self._pending_timers),
            "cpu_utilization": float(self.cpu_utilization),
            "queue_depth": float(self.queue_depth),
            "p95_latency_ms": float(self.p95_latency_ms),
            "error_rate": float(self.error_rate),
            "previous_action": self.previous_action,
        }
        if self.config.history_length > 0:
            obs["history"] = {k: list(v) for k, v in self._history.items()}
        return obs

    def get_metrics(self) -> Dict[str, float | int]:
        steps = max(1, self.timestep)
        avg_reward = sum(self._step_rewards) / max(len(self._step_rewards), 1)
        return {
            "steps": self.timestep,
            "done": int(self.done),
            "final_ready_pods": self.ready_pods,
            "final_pending_pods": len(self._pending_timers),
            "final_queue_depth": self.queue_depth,
            "avg_reward": avg_reward,
            "mean_cpu_utilization": self._safe_mean(self._history.get("cpu_utilization")),
            "mean_latency_ms": self._safe_mean(self._history.get("p95_latency_ms")),
            "total_cost_pod_steps": self.total_cost_pod_steps,
            "total_scale_actions": self.total_scale_actions,
            "requests_served": self.requests_served,
            "requests_dropped": self.requests_dropped,
            "drop_fraction": 0.0
            if (self.requests_served + self.requests_dropped) <= 0
            else self.requests_dropped / (self.requests_served + self.requests_dropped),
            "avg_ready_pods": self.total_cost_pod_steps / steps,
        }

    @staticmethod
    def _safe_mean(values: Deque[float] | None) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)
