from __future__ import annotations

"""
Colab-friendly Unsloth + TRL GRPO trainer for the autoscaling OpenEnv-style environment.

Inputs:
- --base-url: environment server URL (or use --auto-launch-server)
- --init-model: SFT warm-start model/checkpoint
- --trace-path: trace file when auto-launching server

Outputs:
- RL adapter/checkpoint saved under --output-dir
- small run metadata at --output-dir/grpo_run_summary.json
"""

import argparse
import json
import os
import socket
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from datasets import Dataset

from client import OpenEnvClient
from prompts import ACTIONS, format_observation_prompt
from rollout import coerce_legal_action, normalize_action_output


def _configure_nonfatal_warning_filters() -> None:
    # Colab/Unsloth stacks sometimes emit optional-extension warnings (torchao, etc.).
    # Keep output readable unless execution actually fails.
    warnings.filterwarnings("ignore", message=".*torchao.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*torchao.*", category=RuntimeWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Colab-friendly Unsloth + TRL GRPO entrypoint for autoscaling.")
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--init-model", type=str, default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    p.add_argument("--output-dir", type=Path, default=Path("rl_model"))
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--prompt-seed", type=int, default=7)
    p.add_argument("--trace-index", type=int, default=None)
    p.add_argument("--auto-launch-server", action="store_true")
    p.add_argument("--trace-path", type=Path, default=Path("traces.jsonl"))
    p.add_argument("--server-host", type=str, default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--server-wait-seconds", type=float, default=20.0)
    p.add_argument(
        "--server-launch-mode",
        choices=["auto", "packaged", "legacy"],
        default="auto",
        help="Server launch path for --auto-launch-server. 'auto' prefers packaged env.",
    )
    p.add_argument(
        "--env-package-dir",
        type=Path,
        default=Path("envs/autoscale_env"),
        help="Packaged OpenEnv directory used when launch mode is packaged/auto.",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-completion-length", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def completion_to_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(str(v) for v in x)
    return str(x)


def ensure_trace_file(trace_path: Path) -> None:
    if trace_path.exists():
        print(f"[trace-check] Found existing trace file: {trace_path}")
        return
    print(f"[trace-check] Missing {trace_path}. Generating traces now...")
    cmd = [
        "python3",
        "generate_traces.py",
        "--num-traces",
        "30",
        "--output",
        str(trace_path),
        "--episode-length",
        "120",
        "--seed",
        "7",
    ]
    subprocess.run(cmd, check=True)
    print(f"[trace-check] Generated trace file: {trace_path}")


def is_port_in_use(host: str, port: int, timeout_s: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_s)
        return sock.connect_ex((host, port)) == 0


def maybe_launch_server(args: argparse.Namespace) -> subprocess.Popen[str] | None:
    if not args.auto_launch_server:
        return None
    if is_port_in_use(args.server_host, args.server_port):
        print(
            f"[server-launch] Port {args.server_host}:{args.server_port} is already in use. "
            "Assuming an existing server may already be running; skipping auto-launch."
        )
        return None
    ensure_trace_file(args.trace_path)
    trace_path_abs = str(args.trace_path.resolve())
    env_package_dir = args.env_package_dir.resolve()
    launch_mode = args.server_launch_mode
    use_packaged = False
    if launch_mode == "packaged":
        use_packaged = True
    elif launch_mode == "auto":
        use_packaged = env_package_dir.exists()

    if use_packaged:
        server_env = dict(os.environ)
        server_env["TRACE_PATH"] = trace_path_abs
        server_env["ENV_SEED"] = str(args.prompt_seed)
        server_env["HOST"] = args.server_host
        server_env["PORT"] = str(args.server_port)
        if not env_package_dir.exists():
            raise RuntimeError(
                f"Packaged environment directory not found: {env_package_dir}. "
                "Use --server-launch-mode legacy or fix --env-package-dir."
            )
        if (env_package_dir / "uv.lock").exists():
            print(f"[server-launch] Launching packaged env with uv_run from: {env_package_dir}")
            return subprocess.Popen(
                ["python3", "-m", "uv", "run", "server"],
                cwd=str(env_package_dir),
                env=server_env,
            )
        print(
            f"[server-launch] Launching packaged env with python_module from: {env_package_dir} "
            "(uv.lock missing, using fallback mode)."
        )
        return subprocess.Popen(
            ["python3", "-m", "server.app"],
            cwd=str(env_package_dir),
            env=server_env,
        )

    print("[server-launch] Launching legacy root server.py (compatibility mode).")
    cmd = [
        "python3",
        "server.py",
        "--trace-path",
        trace_path_abs,
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
        "--seed",
        str(args.prompt_seed),
    ]
    return subprocess.Popen(cmd)


def wait_for_health(client: OpenEnvClient, timeout_s: float) -> None:
    last_error: str | None = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            health = client.health()
            if health.ok:
                return
            last_error = f"/health returned ok={health.ok} message={health.message!r}"
        except Exception:
            last_error = "Could not connect to /health endpoint"
        time.sleep(0.5)
    detail = f" Last observed issue: {last_error}" if last_error else ""
    raise RuntimeError(
        f"Environment preflight failed: {client.base_url}/health not reachable/healthy within {timeout_s:.1f}s.{detail}"
    )


def build_seed_prompt_dataset(
    client: OpenEnvClient,
    num_prompts: int,
    prompt_seed: int,
    trace_index: int | None,
) -> Dataset:
    prompts: List[str] = []
    seeds: List[int] = []
    trace_indices: List[int] = []
    for i in range(num_prompts):
        seed = prompt_seed + i
        reset = client.reset(seed=seed, trace_index=trace_index)
        prompts.append(format_observation_prompt(reset.observation.model_dump()))
        seeds.append(seed)
        trace_indices.append(trace_index if trace_index is not None else -1)
    return Dataset.from_dict({"prompt": prompts, "seed": seeds, "trace_index": trace_indices})


def make_env_reward_func(base_url: str):
    client = OpenEnvClient(base_url)
    legal_actions = set(ACTIONS)

    def reward_func(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
        _ = prompts
        seeds_raw = kwargs.get("seed", [])
        trace_indices_raw = kwargs.get("trace_index", [])
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            raw_text = completion_to_text(completion)
            chosen, normalized, is_valid = coerce_legal_action(raw_text)

            step_seed = int(seeds_raw[idx]) if idx < len(seeds_raw) else idx
            step_trace_index = int(trace_indices_raw[idx]) if idx < len(trace_indices_raw) else -1
            if step_trace_index >= 0:
                client.reset(seed=step_seed, trace_index=step_trace_index)
            else:
                client.reset(seed=step_seed)

            step_resp = client.step(chosen)
            reward = float(step_resp.reward)
            if normalized not in legal_actions or not is_valid:
                reward -= 0.05
            rewards.append(reward)
        return rewards

    return reward_func


def resolve_init_model(init_model: str) -> str:
    default_hf_model = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    candidate = Path(init_model).expanduser()
    local_path_hint = init_model.startswith(("./", "../", "/", "~"))

    if candidate.exists() and candidate.is_dir():
        print(f"[model-init] Using local model directory: {candidate}")
        return str(candidate)
    if local_path_hint:
        raise RuntimeError(
            f"--init-model points to a local path but directory does not exist: {init_model}. "
            f"Use a valid local folder or an HF model id like {default_hf_model}."
        )
    if "/" in init_model:
        print(f"[model-init] Using Hugging Face model id: {init_model}")
        return init_model
    if init_model != default_hf_model:
        print(
            f"[model-init] Local model directory not found for '{init_model}'. "
            f"Falling back to default HF model id: {default_hf_model}"
        )
    return default_hf_model


def load_unsloth_model_and_tokenizer(args: argparse.Namespace, resolved_model: str):
    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except Exception as exc:
        raise RuntimeError(
            "Unsloth is required for this script. Install it in Colab, then retry."
        ) from exc

    PatchFastRL("GRPO", FastLanguageModel)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resolved_model,
        max_seq_length=args.max_prompt_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer, resolved_model


def load_grpo_classes():
    # Keep TRL import after Unsloth patching for compatibility.
    from trl import GRPOConfig, GRPOTrainer

    return GRPOConfig, GRPOTrainer


def ensure_cuda_or_die() -> tuple[bool, str]:
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        raise RuntimeError("GPU runtime is required. In Colab: Runtime -> Change runtime type -> GPU")
    gpu_name = str(torch.cuda.get_device_name(0))
    return cuda_available, gpu_name


def main() -> None:
    _configure_nonfatal_warning_filters()
    args = parse_args()
    cuda_available, gpu_name = ensure_cuda_or_die()
    resolved_init_model = resolve_init_model(args.init_model)
    print(
        "[startup] cuda_available={cuda_available} gpu_name={gpu_name} "
        "base_url={base_url} server_port={server_port} launch_mode={launch_mode} "
        "env_package_dir={env_package_dir} init_model={init_model} "
        "output_dir={output_dir} num_prompts={num_prompts} max_steps={max_steps}".format(
            cuda_available=cuda_available,
            gpu_name=gpu_name,
            base_url=args.base_url,
            server_port=args.server_port,
            launch_mode=args.server_launch_mode,
            env_package_dir=args.env_package_dir,
            init_model=resolved_init_model,
            output_dir=args.output_dir,
            num_prompts=args.num_prompts,
            max_steps=args.max_steps,
        )
    )
    server_proc = maybe_launch_server(args)
    client = OpenEnvClient(args.base_url)
    wait_for_health(client, timeout_s=args.server_wait_seconds)

    dataset = build_seed_prompt_dataset(
        client=client,
        num_prompts=args.num_prompts,
        prompt_seed=args.prompt_seed,
        trace_index=args.trace_index,
    )
    model, tokenizer, _ = load_unsloth_model_and_tokenizer(args, resolved_init_model)
    GRPOConfig, GRPOTrainer = load_grpo_classes()

    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        seed=args.seed,
        report_to=[],
    )

    try:
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=make_env_reward_func(args.base_url),
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        print(f"Saved RL model to {args.output_dir}")
        summary = {
            "num_prompts": len(dataset),
            "max_steps": args.max_steps,
            "base_url": args.base_url,
            "init_model": resolved_init_model,
            "normalization_example": normalize_action_output("scale_up_2\nextra"),
        }
        summary_path = args.output_dir / "grpo_run_summary.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved run summary to {summary_path}")
    finally:
        if server_proc is not None:
            server_proc.terminate()


if __name__ == "__main__":
    main()
