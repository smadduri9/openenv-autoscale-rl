# OpenEnv Autoscaling Environment

## Project Overview

This project implements a Kubernetes-like autoscaling simulator and packages it as a formal OpenEnv environment. It includes:

- A deterministic simulator for one autoscaled service under changing load
- A formal OpenEnv package at `envs/autoscale_env`
- Baseline heuristic policy + evaluation scripts
- SFT and Unsloth+GRPO training scaffolding
- Colab notebook and script for packaged-environment RL training

## Why This Environment Is Interesting

- **Delayed scaling dynamics**: scale-up actions have startup delays, so action quality depends on anticipation.
- **Real tradeoffs**: reward balances latency SLO violations, queue buildup, error rate, cost, and scaling flaps.
- **Policy-learning friendly**: deterministic traces and explicit action space make it suitable for imitation + RL experiments.
- **Hackathon practical**: includes end-to-end generation, baseline, packaging, smoke tests, and Colab training flow.

## Formal OpenEnv Package Status

The primary environment is now a formal OpenEnv package under `envs/autoscale_env` with:

- `openenv.yaml`
- `pyproject.toml`
- packaged server/client/models/environment modules
- validated deployment modes (`openenv_serve`, `uv_run`, `python_module`)

Validation command:

```bash
cd envs/autoscale_env
openenv validate --verbose
```

## Repo Structure

- `envs/autoscale_env/` - primary packaged OpenEnv environment (submission path)
- `simulator.py` - simulator source of truth (state dynamics + reward)
- `generate_traces.py` - synthetic workload traces
- `hpa_policy.py`, `run_baseline.py` - heuristic baseline + trajectory export
- `colab_train_rl.py` - packaged-environment Unsloth + TRL GRPO training script
- `notebooks/unsloth_openenv_autoscale_grpo.ipynb` - Colab workflow
- `rollout.py`, `train_rl.py`, `eval_policy.py` - rollout/training/evaluation utilities
- `server.py`, `client.py`, `models.py`, `environment.py` - legacy compatibility wrappers

## Quickstart (Packaged OpenEnv Primary Path)

Install deps:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install uv openenv-core
```

Generate traces:

```bash
python3 generate_traces.py --num-traces 30 --output traces.jsonl --episode-length 120 --seed 7
```

Validate package:

```bash
cd envs/autoscale_env
python3 -m uv lock
openenv validate --verbose
```

Launch packaged environment:

```bash
cd envs/autoscale_env
python3 -m uv run server
```

Smoke test from repo root:

```bash
python3 smoke_test_openenv.py --base-url http://127.0.0.1:8000 --policy random --seed 7 --trace-index 0
```

Legacy fallback launch (compatibility only):

```bash
python3 server.py --trace-path traces.jsonl --host 127.0.0.1 --port 8000 --seed 7
```

## Colab / Unsloth Training

Use `notebooks/unsloth_openenv_autoscale_grpo.ipynb` or run the script directly:

```bash
python3 colab_train_rl.py \
  --base-url http://127.0.0.1:8000 \
  --auto-launch-server \
  --server-launch-mode packaged \
  --env-package-dir envs/autoscale_env \
  --trace-path traces.jsonl \
  --init-model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
  --output-dir rl_model_unsloth_grpo \
  --num-prompts 48 \
  --prompt-seed 7 \
  --max-steps 30 \
  --learning-rate 5e-6 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --num-generations 4 \
  --max-prompt-length 512 \
  --max-completion-length 8 \
  --seed 7
```

## Results / Artifacts

This repo demonstrates that packaged OpenEnv launch and Unsloth GRPO training run end-to-end and produce artifacts.

Expected output directory:

- `rl_model_unsloth_grpo/`
  - model/adapters saved by trainer
  - tokenizer files
  - `grpo_run_summary.json`

Additional generated artifacts commonly used in this project:

- `expert_trajectories.jsonl`
- `rl_rollouts.jsonl`
- `plots/*.png`

## Limitations and Next Steps

- Training completion is verified; broad quantitative RL improvement claims are not asserted here.
- Current reward evaluation in GRPO is single-step (`reset -> step`) by design for MVP stability.
- Next steps:
  - expand multi-step rollout reward functions for stronger policy learning
  - run larger-scale policy comparisons across more traces
  - add Docker deployment mode for full multi-mode OpenEnv packaging completeness
