from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import uvicorn

from ..environment import AutoscaleOpenEnv
from ..models import (
    AutoscaleAction,
    AutoscaleObservation,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    StateResponse,
    StepRequest,
    StepResponse,
)
from .autoscale_environment import AutoscaleEnvironment

try:
    from openenv.core.env_server.http_server import create_app as create_openenv_app
except Exception:  # pragma: no cover - fallback path when openenv-core is unavailable
    create_openenv_app = None


def _env_factory() -> AutoscaleEnvironment:
    trace_path = os.getenv("TRACE_PATH", "traces.jsonl")
    seed = int(os.getenv("ENV_SEED", "7"))
    return AutoscaleEnvironment(trace_path=trace_path, seed=seed)


def _create_legacy_fastapi_app() -> FastAPI:
    from simulator import AutoscaleSimConfig

    app_obj = FastAPI(title="Autoscale OpenEnv")
    env = AutoscaleOpenEnv(
        trace_path=Path(os.getenv("TRACE_PATH", "traces.jsonl")),
        config=AutoscaleSimConfig(),
        seed=int(os.getenv("ENV_SEED", "7")),
    )

    @app_obj.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(ok=True, message="autoscale-openenv-ready")

    @app_obj.post("/reset", response_model=ResetResponse)
    def reset(req: ResetRequest) -> ResetResponse:
        try:
            return env.reset(seed=req.seed, trace_id=req.trace_id, trace_index=req.trace_index)
        except (ValueError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app_obj.post("/step", response_model=StepResponse)
    def step(req: StepRequest) -> StepResponse:
        try:
            return env.step(req.action.action)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app_obj.get("/state", response_model=StateResponse)
    def state() -> StateResponse:
        try:
            return StateResponse(episode_id=env._episode_id, state=env.state())
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app_obj


def build_app(trace_path: str | None = None, seed: int | None = None) -> FastAPI:
    if trace_path is not None:
        os.environ["TRACE_PATH"] = trace_path
    if seed is not None:
        os.environ["ENV_SEED"] = str(seed)
    if create_openenv_app is not None:
        return create_openenv_app(
            _env_factory,
            AutoscaleAction,
            AutoscaleObservation,
            env_name="autoscale_env",
        )
    return _create_legacy_fastapi_app()


app = build_app()


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(build_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
