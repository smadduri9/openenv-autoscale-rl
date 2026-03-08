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


class HeuristicHPAPolicy:
    VALID_ACTIONS = {
        "scale_down_2",
        "scale_down_1",
        "hold",
        "scale_up_1",
        "scale_up_2",
        "scale_up_4",
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
