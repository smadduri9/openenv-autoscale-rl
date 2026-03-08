from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, TypedDict

from hpa_policy import HeuristicHPAPolicy
from prompts import ACTIONS, format_observation_prompt
from rollout import normalize_action_output
from run_baseline import AGG_KEYS
from simulator import AutoscaleSimConfig, AutoscaleSimulator


class TraceRecord(TypedDict):
    trace_id: str
    family: str
    rps: List[float]


class TextActionModel:
    def __init__(self, model_path: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            from peft import AutoPeftModelForCausalLM

            self.model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="auto")
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        self.model.eval()

    def predict_raw(self, observation: Mapping[str, object]) -> str:
        prompt = format_observation_prompt(observation)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


def is_likely_hf_repo_id(model_path: str) -> bool:
    return "/" in model_path and not Path(model_path).exists()


def try_load_model(model_path: str, label: str) -> TextActionModel | None:
    path_obj = Path(model_path)
    if not path_obj.exists() and not is_likely_hf_repo_id(model_path):
        print(f"\nSkipping {label} policy: local model path not found: {model_path}")
        print(f"Tip: train it first or pass --skip-{label.lower()} for this run.")
        return None
    try:
        return TextActionModel(model_path)
    except Exception as exc:
        print(f"\nSkipping {label} policy: failed to load model {model_path!r}")
        print(f"Reason: {exc}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heuristic, SFT, and RL policies.")
    parser.add_argument("--trace-path", type=Path, default=Path("traces.jsonl"))
    parser.add_argument("--max-traces", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sft-model-path", type=str, default="sft_model")
    parser.add_argument("--rl-model-path", type=str, default="rl_model")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("policy_eval_summary.json"))
    return parser.parse_args()


def load_traces(path: Path) -> List[TraceRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")
    traces: List[TraceRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            for key in ("trace_id", "family", "rps"):
                if key not in raw:
                    raise ValueError(f"Missing key {key!r} at line {line_no}")
            traces.append(
                {
                    "trace_id": str(raw["trace_id"]),
                    "family": str(raw["family"]),
                    "rps": [float(v) for v in raw["rps"]],
                }
            )
    return traces


def aggregate_metrics(per_trace_metrics: List[Mapping[str, float | int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in AGG_KEYS:
        vals = [float(m[key]) for m in per_trace_metrics]
        out[key] = mean(vals) if vals else 0.0
    return out


def evaluate_heuristic(trace: TraceRecord) -> Dict[str, float | int]:
    sim = AutoscaleSimulator(
        config=AutoscaleSimConfig(episode_length=len(trace["rps"]), history_length=len(trace["rps"])),
        trace=trace["rps"],
        trace_family=trace["family"],
        seed=None,
    )
    policy = HeuristicHPAPolicy()
    obs = sim.reset()
    policy.reset()
    done = False
    while not done:
        action = policy.choose_action(obs)
        obs, _, done, _ = sim.step(action)
    return sim.get_metrics()


def evaluate_model(trace: TraceRecord, model: TextActionModel) -> Dict[str, float | int]:
    sim = AutoscaleSimulator(
        config=AutoscaleSimConfig(episode_length=len(trace["rps"]), history_length=len(trace["rps"])),
        trace=trace["rps"],
        trace_family=trace["family"],
        seed=None,
    )
    obs = sim.reset()
    done = False
    while not done:
        raw = model.predict_raw(obs)
        action = normalize_action_output(raw)
        if action not in ACTIONS:
            action = "hold"
        obs, _, done, _ = sim.step(action)
    return sim.get_metrics()


def print_summary(name: str, aggregate: Mapping[str, float]) -> None:
    print(f"\n{name} aggregate metrics")
    for key in AGG_KEYS:
        print(f"- {key}: {aggregate[key]:.6f}")


def main() -> None:
    args = parse_args()
    traces = load_traces(args.trace_path)[: args.max_traces]
    if not traces:
        raise RuntimeError("No traces available for evaluation.")

    heuristic_metrics: List[Dict[str, float | int]] = []
    for trace in traces:
        heuristic_metrics.append(evaluate_heuristic(trace))
    heuristic_agg = aggregate_metrics(heuristic_metrics)
    print_summary("Heuristic baseline", heuristic_agg)
    summary: Dict[str, Mapping[str, float]] = {"heuristic": heuristic_agg}

    if not args.skip_sft:
        sft_model = try_load_model(args.sft_model_path, "SFT")
        if sft_model is not None:
            sft_metrics: List[Dict[str, float | int]] = []
            for trace in traces:
                sft_metrics.append(evaluate_model(trace, sft_model))
            sft_agg = aggregate_metrics(sft_metrics)
            print_summary("SFT policy", sft_agg)
            summary["sft"] = sft_agg

    if not args.skip_rl:
        rl_model = try_load_model(args.rl_model_path, "RL")
        if rl_model is not None:
            rl_metrics: List[Dict[str, float | int]] = []
            for trace in traces:
                rl_metrics.append(evaluate_model(trace, rl_model))
            rl_agg = aggregate_metrics(rl_metrics)
            print_summary("RL policy", rl_agg)
            summary["rl"] = rl_agg

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary JSON to {args.output_json}")


if __name__ == "__main__":
    main()
