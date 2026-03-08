from __future__ import annotations

from hpa_policy import HeuristicHPAPolicy, HeuristicHPAPolicyConfig


def test_aggressive_scale_up_on_distress() -> None:
    p = HeuristicHPAPolicy(HeuristicHPAPolicyConfig())
    obs = {
        "cpu_utilization": 0.5,
        "queue_depth": 300.0,
        "p95_latency_ms": 400.0,
        "error_rate": 0.2,
    }
    assert p.choose_action(obs) == "scale_up_4"


def test_scale_down_after_idle_patience() -> None:
    p = HeuristicHPAPolicy(HeuristicHPAPolicyConfig(empty_queue_patience=2, deep_idle_patience=4, cooldown_steps=0))
    obs = {
        "cpu_utilization": 0.1,
        "queue_depth": 0.0,
        "p95_latency_ms": 80.0,
        "error_rate": 0.0,
    }
    assert p.choose_action(obs) == "hold"
    assert p.choose_action(obs) == "scale_down_1"
    assert p.choose_action(obs) == "scale_down_1"
