from __future__ import annotations

import argparse

import uvicorn

from envs.autoscale_env.server.app import build_app

app = build_app()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run autoscale OpenEnv server.")
    p.add_argument("--trace-path", default="traces.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(trace_path=args.trace_path, seed=args.seed)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
