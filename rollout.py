from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Mapping, Sequence

from client import OpenEnvClient
from hpa_policy import HeuristicHPAPolicy
from prompts import ACTIONS, format_observation_prompt

LEGAL_ACTIONS = set(ACTIONS)


def normalize_action_output(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


@dataclass
class StepTransition:
    timestep: int
    observation: Dict[str, object]
    chosen_action: str
    raw_action_text: str
    normalized_action: str
    reward: float
    next_observation: Dict[str, object]
    done: bool


@dataclass
class EpisodeRollout:
    episode_id: str
    trace_id: str
    family: str
    steps: List[StepTransition] = field(default_factory=list)
    cumulative_reward: float = 0.0
    invalid_output_count: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)


class PolicyAdapter(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def choose_action(self, observation: Mapping[str, object]) -> str:
        pass

    def choose_raw(self, observation: Mapping[str, object]) -> str:
        return self.choose_action(observation)


class HeuristicPolicyAdapter(PolicyAdapter):
    def __init__(self) -> None:
        self.policy = HeuristicHPAPolicy()

    def reset(self) -> None:
        self.policy.reset()

    def choose_action(self, observation: Mapping[str, object]) -> str:
        return self.policy.choose_action(observation)


class RandomPolicyAdapter(PolicyAdapter):
    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.rng = Random(seed)

    def reset(self) -> None:
        self.rng = Random(self.seed)

    def choose_action(self, observation: Mapping[str, object]) -> str:
        _ = observation
        return self.rng.choice(tuple(LEGAL_ACTIONS))


class TextModelPolicyAdapter(PolicyAdapter):
    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def reset(self) -> None:
        return

    def choose_raw(self, observation: Mapping[str, object]) -> str:
        prompt = format_observation_prompt(observation)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def choose_action(self, observation: Mapping[str, object]) -> str:
        return normalize_action_output(self.choose_raw(observation))


def run_episode(
    client: OpenEnvClient,
    policy: PolicyAdapter,
    *,
    seed: int | None = None,
    trace_index: int | None = None,
) -> EpisodeRollout:
    reset = client.reset(seed=seed, trace_index=trace_index)
    policy.reset()
    obs_model = reset.observation
    obs: Dict[str, object] = obs_model.model_dump()
    done = obs_model.done

    rollout = EpisodeRollout(
        episode_id=reset.episode_id,
        trace_id=reset.trace_id,
        family=reset.family,
        action_counts={a: 0 for a in ACTIONS},
    )
    while not done:
        raw = policy.choose_raw(obs)
        normalized = normalize_action_output(raw)
        chosen = normalized if normalized in LEGAL_ACTIONS else "hold"
        if normalized not in LEGAL_ACTIONS:
            rollout.invalid_output_count += 1
        step = client.step(chosen)
        next_obs = step.observation.model_dump()
        rollout.steps.append(
            StepTransition(
                timestep=int(obs.get("timestep", 0)),
                observation=dict(obs),
                chosen_action=chosen,
                raw_action_text=raw,
                normalized_action=normalized,
                reward=step.reward,
                next_observation=dict(next_obs),
                done=step.done,
            )
        )
        rollout.cumulative_reward += step.reward
        rollout.action_counts[chosen] = rollout.action_counts.get(chosen, 0) + 1
        obs = next_obs
        done = step.done
    return rollout
