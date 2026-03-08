# Autoscaling Simulator + OpenEnv-Style Wrapper

Hackathon project for learning autoscaling policies with a deterministic single-service simulator, an OpenEnv-style HTTP wrapper, and SFT/RL scaffolding.

## What This Repo Includes

- Simulator core for autoscaling dynamics (`reset`, `step`, reward, metrics)
- Synthetic workload trace generation
- Heuristic baseline policy and evaluation runner
- OpenEnv-style HTTP environment (`/health`, `/reset`, `/step`, `/state`)
- SFT warm-start dataset/training/eval scripts
- RL rollout + minimal training scaffolding
- Plotting utilities for heuristic vs SFT comparison

## Project Structure

- `simulator.py` - core simulator and reward dynamics
- `generate_traces.py` - synthetic trace generator (JSONL)
- `hpa_policy.py`, `run_baseline.py` - heuristic baseline + metrics + trajectory export
- `models.py`, `environment.py`, `server.py`, `client.py` - OpenEnv-style wrapper and API
- `smoke_test_openenv.py` - end-to-end API smoke test
- `prompts.py`, `build_sft_dataset.py`, `train_sft.py`, `eval_sft.py` - SFT pipeline
- `rollout.py`, `train_rl.py`, `eval_policy.py`, `colab_train_rl.py` - RL/eval pipeline
- `plot_results.py` - PNG plot generation in `plots/`

## Quickstart

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Generate traces:

```bash
python3 generate_traces.py --num-traces 20 --episode-length 120 --output traces.jsonl --seed 7
```

Run baseline (and optionally export expert trajectories):

```bash
python3 run_baseline.py --trace-path traces.jsonl --max-traces 20 --compare-random --export-trajectories --trajectory-output expert_trajectories.jsonl
```

Run OpenEnv-style server:

```bash
python3 server.py --trace-path traces.jsonl --host 127.0.0.1 --port 8000 --seed 7
```

Run smoke test against server:

```bash
python3 smoke_test_openenv.py --base-url http://127.0.0.1:8000 --policy random
```

Run SFT evaluation:

```bash
python3 eval_sft.py --dataset-path sft_data/val.jsonl --model-path sft_model --max-examples 100 --print-examples 5
```

## Notes

- This repo currently provides an **OpenEnv-style** custom environment wrapper.
- If your local server is older than the latest models, `client.py` includes backward-compatible response normalization for health/reset/step/state payloads.
