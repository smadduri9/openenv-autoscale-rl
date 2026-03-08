from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class HeuristicHPAPolicyConfig:
    cpu_scale_up_threshold: float = 0.78
    cpu_scale_down_threshold: float = 0.30
    queue_scale_up_threshold: float = 80.0
    queue_aggressive_scale_up_threshold: float = 220.0
    latency_scale_up_threshold: float = 220.0
    latency_aggressive_scale_up_threshold: float = 320.0
    error_aggressive_scale_up_threshold: float = 0.10
    empty_queue_threshold: float = 5.0
    empty_queue_patience: int = 3
    deep_idle_patience: int = 6
    cooldown_steps: int = 2
    allow_break_cooldown_on_distress: bool = True
    rollback_error_threshold: float = 0.08
    rollback_latency_threshold: float = 320.0
    enable_rate_limit_queue_threshold: float = 160.0
    disable_rate_limit_queue_threshold: float = 40.0


class HeuristicHPAPolicy:
    VALID_ACTIONS = {
        "scale_down_2",
        "scale_down_1",
        "hold",
        "scale_up_1",
        "scale_up_2",
        "scale_up_4",
        "enable_rate_limit",
        "disable_rate_limit",
        "rollback_release",
    }

    def __init__(self, config: HeuristicHPAPolicyConfig | None = None) -> None:
        self.config = config or HeuristicHPAPolicyConfig()
        self._cooldown_remaining = 0
        self._idle_streak = 0

    def reset(self) -> None:
        self._cooldown_remaining = 0
        self._idle_streak = 0

    def choose_action(self, observation: Mapping[str, object]) -> str:
        cpu = float(observation.get("cpu_utilization", 0.0))
        queue = float(observation.get("queue_depth", 0.0))
        latency = float(observation.get("p95_latency_ms", 0.0))
        error_rate = float(observation.get("error_rate", 0.0))
        rate_limit_enabled = bool(observation.get("rate_limit_enabled", False))
        bad_deploy_active = bool(observation.get("bad_deploy_active", False))
        rollback_pending_steps = int(observation.get("rollback_pending_steps", 0))
        dependency_slowdown_active = bool(observation.get("dependency_slowdown_active", False))

        severe = (
            queue >= self.config.queue_aggressive_scale_up_threshold
            or latency >= self.config.latency_aggressive_scale_up_threshold
            or error_rate >= self.config.error_aggressive_scale_up_threshold
        )
        moderate = (
            queue >= self.config.queue_scale_up_threshold
            or latency >= self.config.latency_scale_up_threshold
            or cpu >= self.config.cpu_scale_up_threshold
        )

        idle_like = (
            queue <= self.config.empty_queue_threshold
            and cpu <= self.config.cpu_scale_down_threshold
            and error_rate <= 0.001
        )
        self._idle_streak = self._idle_streak + 1 if idle_like else 0

        if (
            bad_deploy_active
            and rollback_pending_steps <= 0
            and (
                error_rate >= self.config.rollback_error_threshold
                or latency >= self.config.rollback_latency_threshold
            )
        ):
            return self._finalize_action("rollback_release")

        if (
            dependency_slowdown_active
            and not rate_limit_enabled
            and (queue >= self.config.enable_rate_limit_queue_threshold or latency >= self.config.latency_scale_up_threshold)
        ):
            return self._finalize_action("enable_rate_limit")
        if (
            rate_limit_enabled
            and queue <= self.config.disable_rate_limit_queue_threshold
            and error_rate <= 0.01
        ):
            return self._finalize_action("disable_rate_limit")

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self.config.allow_break_cooldown_on_distress and severe:
                return self._finalize_action("scale_up_4")
            return self._finalize_action("hold")

        if severe:
            return self._finalize_action("scale_up_4")
        if queue >= self.config.queue_scale_up_threshold or latency >= self.config.latency_scale_up_threshold:
            return self._finalize_action("scale_up_2")
        if moderate:
            return self._finalize_action("scale_up_1")
        if self._idle_streak >= self.config.deep_idle_patience:
            return self._finalize_action("scale_down_2")
        if self._idle_streak >= self.config.empty_queue_patience:
            return self._finalize_action("scale_down_1")
        return self._finalize_action("hold")

    def _finalize_action(self, action: str) -> str:
        if action not in self.VALID_ACTIONS:
            raise ValueError(f"Policy produced invalid action: {action}")
        if action != "hold":
            self._cooldown_remaining = self.config.cooldown_steps
        return action
