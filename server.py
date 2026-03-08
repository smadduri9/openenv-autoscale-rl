from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
import uvicorn

from environment import AutoscaleOpenEnv
from models import HealthResponse, ResetRequest, ResetResponse, StateResponse, StepRequest, StepResponse
from simulator import AutoscaleSimConfig

app = FastAPI(title="Autoscale OpenEnv")
_ENV: AutoscaleOpenEnv | None = None


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True, message="autoscale-openenv-ready")


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    if _ENV is None:
        raise HTTPException(status_code=500, detail="Server environment not initialized.")
    try:
        return _ENV.reset(seed=req.seed, trace_id=req.trace_id, trace_index=req.trace_index)
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    if _ENV is None:
        raise HTTPException(status_code=500, detail="Server environment not initialized.")
    try:
        return _ENV.step(req.action.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    if _ENV is None:
        raise HTTPException(status_code=500, detail="Server environment not initialized.")
    try:
        return StateResponse(episode_id=_ENV._episode_id, state=_ENV.state())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run autoscale OpenEnv server.")
    p.add_argument("--trace-path", default="traces.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    global _ENV
    args = parse_args()
    _ENV = AutoscaleOpenEnv(trace_path=Path(args.trace_path), config=AutoscaleSimConfig(), seed=args.seed)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
