from __future__ import annotations

from typing import Mapping, Sequence

ACTIONS: Sequence[str] = (
    "scale_down_2",
    "scale_down_1",
    "hold",
    "scale_up_1",
    "scale_up_2",
    "scale_up_4",
    "enable_rate_limit",
    "disable_rate_limit",
    "rollback_release",
)

SYSTEM_LINE = (
    "You are an autoscaling controller. Pick exactly one valid action for the current state. "
    "Return only the action."
)


def format_observation_prompt(observation: Mapping[str, object]) -> str:
    lines = [
        "Role: Autoscaling controller",
        "Task: choose exactly one valid action",
        "",
        "Current state:",
        f"- timestep: {int(observation.get('timestep', 0))}",
        f"- incoming_rps: {float(observation.get('incoming_rps', 0.0)):.3f}",
        f"- ready_pods: {int(observation.get('ready_pods', 0))}",
        f"- pending_pods: {int(observation.get('pending_pods', 0))}",
        f"- cpu_utilization: {float(observation.get('cpu_utilization', 0.0)):.6f}",
        f"- queue_depth: {float(observation.get('queue_depth', 0.0)):.3f}",
        f"- p95_latency_ms: {float(observation.get('p95_latency_ms', 0.0)):.3f}",
        f"- error_rate: {float(observation.get('error_rate', 0.0)):.6f}",
        f"- previous_action: {str(observation.get('previous_action', 'hold'))}",
        f"- rate_limit_enabled: {bool(observation.get('rate_limit_enabled', False))}",
        f"- bad_deploy_active: {bool(observation.get('bad_deploy_active', False))}",
        f"- dependency_slowdown_active: {bool(observation.get('dependency_slowdown_active', False))}",
        f"- rollback_pending_steps: {int(observation.get('rollback_pending_steps', 0))}",
        "",
        "Valid actions:",
        "- scale_down_2",
        "- scale_down_1",
        "- hold",
        "- scale_up_1",
        "- scale_up_2",
        "- scale_up_4",
        "- enable_rate_limit",
        "- disable_rate_limit",
        "- rollback_release",
        "",
        "Return only the action.",
    ]
    return "\n".join(lines)


def build_plain_text_example(observation: Mapping[str, object], action: str) -> dict[str, str]:
    return {"text": f"{format_observation_prompt(observation)}\n{action}"}


def build_chat_messages_example(observation: Mapping[str, object], action: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_LINE},
            {"role": "user", "content": format_observation_prompt(observation)},
            {"role": "assistant", "content": action},
        ]
    }
