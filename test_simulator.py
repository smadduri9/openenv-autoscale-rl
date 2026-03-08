from __future__ import annotations

from math import isfinite

from simulator import AutoscaleSimConfig, AutoscaleSimulator


def make_config(**overrides: object) -> AutoscaleSimConfig:
    base = AutoscaleSimConfig(
        episode_length=8,
        min_pods=1,
        max_pods=8,
        initial_pods=2,
        pod_capacity_rps=50.0,
        startup_delay_steps=2,
        history_length=5,
    )
    params = {**base.__dict__, **overrides}
    return AutoscaleSimConfig(**params)


def test_reset_initializes_state() -> None:
    sim = AutoscaleSimulator(make_config(), trace=[20.0, 25.0, 30.0], seed=123)
    obs = sim.reset()
    assert obs["timestep"] == 0
    assert obs["incoming_rps"] == 20.0
    assert obs["ready_pods"] == 2
    assert obs["pending_pods"] == 0
    assert obs["queue_depth"] == 0.0
    assert obs["previous_action"] == "hold"
    assert sim.done is False


def test_step_hold_advances_timestep() -> None:
    sim = AutoscaleSimulator(make_config(), trace=[10.0, 10.0, 10.0], seed=1)
    sim.reset()
    obs, reward, done, _ = sim.step("hold")
    assert obs["timestep"] == 1
    assert isfinite(reward)
    assert done is False


def test_scaling_up_adds_pending_pods() -> None:
    sim = AutoscaleSimulator(make_config(startup_delay_steps=3), trace=[20.0, 20.0, 20.0], seed=1)
    sim.reset()
    obs, _, _, info = sim.step("scale_up_2")
    assert obs["pending_pods"] == 2
    assert obs["ready_pods"] == 2
    assert info["action_delta_applied"] == 2


def test_pending_pods_become_ready_after_delay() -> None:
    sim = AutoscaleSimulator(make_config(startup_delay_steps=2), trace=[40.0, 40.0, 40.0], seed=1)
    sim.reset()
    sim.step("scale_up_1")
    obs, _, _, _ = sim.step("hold")
    assert obs["pending_pods"] == 0
    assert obs["ready_pods"] == 3


def test_scaling_down_respects_min_pods() -> None:
    sim = AutoscaleSimulator(make_config(initial_pods=1, min_pods=1), trace=[15.0, 15.0], seed=1)
    sim.reset()
    obs, _, _, info = sim.step("scale_down_2")
    assert obs["ready_pods"] == 1
    assert info["action_delta_applied"] == 0


def test_queue_grows_when_over_capacity() -> None:
    sim = AutoscaleSimulator(
        make_config(initial_pods=1, pod_capacity_rps=30.0),
        trace=[120.0, 120.0],
        seed=1,
    )
    sim.reset()
    obs, _, _, _ = sim.step("hold")
    assert obs["queue_depth"] > 0.0


def test_queue_drains_when_capacity_exceeds_demand() -> None:
    sim = AutoscaleSimulator(
        make_config(initial_pods=1, pod_capacity_rps=50.0, startup_delay_steps=0),
        trace=[200.0, 10.0, 10.0],
        seed=1,
    )
    sim.reset()
    obs1, _, _, _ = sim.step("hold")
    assert obs1["queue_depth"] > 0.0
    obs2, _, _, _ = sim.step("scale_up_2")
    assert obs2["queue_depth"] < obs1["queue_depth"]


def test_reward_is_numeric_and_finite() -> None:
    sim = AutoscaleSimulator(make_config(), trace=[80.0, 90.0, 100.0], seed=1)
    sim.reset()
    _, reward, _, _ = sim.step("hold")
    assert isinstance(reward, float)
    assert isfinite(reward)


def test_episode_ends_at_expected_length() -> None:
    sim = AutoscaleSimulator(make_config(episode_length=3), trace=[10, 10, 10, 10, 10], seed=1)
    sim.reset()
    done = False
    for _ in range(3):
        _, _, done, _ = sim.step("hold")
    assert done is True


def test_seed_reproducibility_for_same_actions() -> None:
    actions = ["hold", "scale_up_1", "hold", "scale_down_1", "hold"]
    cfg = make_config(episode_length=5)
    trace = [80.0, 120.0, 20.0, 35.0, 55.0]
    sim_a = AutoscaleSimulator(cfg, trace=trace, seed=42)
    sim_b = AutoscaleSimulator(cfg, trace=trace, seed=42)
    sim_a.reset()
    sim_b.reset()
    rewards_a = []
    rewards_b = []
    for action in actions:
        obs_a, rew_a, done_a, _ = sim_a.step(action)
        obs_b, rew_b, done_b, _ = sim_b.step(action)
        rewards_a.append(rew_a)
        rewards_b.append(rew_b)
        assert obs_a == obs_b
        assert done_a == done_b
    assert rewards_a == rewards_b
