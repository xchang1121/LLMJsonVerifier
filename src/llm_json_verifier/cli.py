"""Small CLI shared by local development and the two deployment containers."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys

from . import __version__
from .config import VLLM_VERSION, load_settings
from .errors import VerifierError


def emit(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="llmjv")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("serve", "engine-command", "engine-start", "doctor"):
        command = commands.add_parser(name)
        command.add_argument("--config", default=None)
        if name == "doctor":
            command.add_argument(
                "--backend", action="store_true", help="also query a running vLLM server"
            )
        if name == "engine-command":
            command.add_argument("--json", action="store_true")
    for name in ("classify", "benchmark", "evaluate", "verify-cache"):
        command = commands.add_parser(name)
        command.add_argument("--url", default="http://127.0.0.1:8080")
        if name == "evaluate":
            command.add_argument("--dataset", required=True)
            command.add_argument("--rotate", action="store_true")
            command.add_argument("--seed", type=int)
        else:
            command.add_argument("--input", required=True)
        if name == "benchmark":
            command.add_argument("--repeats", type=int, default=20)
            command.add_argument("--warmup", type=int, default=1)
            command.add_argument("--cache-mode", choices=("warm", "cold"), default="warm")
        if name in {"benchmark", "evaluate"}:
            command.add_argument("--concurrency", type=int, default=1)
            command.add_argument("--records", help="write per-request JSONL to a new file")
        if name == "verify-cache":
            command.add_argument("--tolerance", type=float, default=1e-3)
    dataset = commands.add_parser("make-dataset")
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--context-chars", type=int, nargs="+", default=[0])
    dataset.add_argument(
        "--positions",
        nargs="+",
        choices=("start", "middle", "end"),
        default=["start", "middle", "end"],
    )
    return root


async def remote(args) -> int:
    from .client import GatewayClient, benchmark, evaluate, read_request, verify_cache

    async with GatewayClient(args.url) as client:
        if args.command == "evaluate":
            result = await evaluate(
                client, args.dataset, args.rotate, args.records, args.concurrency, args.seed
            )
        else:
            request = read_request(args.input)
            if args.command == "classify":
                result = (await client.classify(request)).model_dump()
            elif args.command == "benchmark":
                result = await benchmark(
                    client,
                    request,
                    args.repeats,
                    args.concurrency,
                    args.warmup,
                    args.cache_mode,
                    args.records,
                )
            else:
                result = await verify_cache(client, request, args.tolerance)
        emit(result)
        if args.command in {"benchmark", "evaluate"}:
            return int(
                bool(
                    result["failed"]
                    or result.get("rotation_outcomes", {}).get("failed")
                    or result.get("warmup_outcomes", {}).get("failed")
                )
            )
        return 1 if args.command == "verify-cache" and not result["passed"] else 0


async def doctor(settings, backend: bool) -> None:
    from .backend import VLLMBackend
    from .prompts import load_compiler
    from .service import ClassificationService

    compiler = await asyncio.to_thread(load_compiler, settings)
    result = {
        "gateway_version": __version__,
        "required_vllm_version": VLLM_VERSION,
        **compiler.registry_summary(),
        "model_weights_loaded": False,
    }
    if backend:
        service = ClassificationService(settings, compiler, VLLMBackend(settings))
        try:
            result["backend"] = await service.verify_backend()
        finally:
            await service.close()
    emit(result)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "make-dataset":
            from .datasets import write_regression_dataset

            emit(
                write_regression_dataset(
                    args.output, tuple(args.context_chars), tuple(args.positions)
                )
            )
            return 0
        if args.command in {"classify", "benchmark", "evaluate", "verify-cache"}:
            return asyncio.run(remote(args))
        settings = load_settings(args.config)
        if args.command == "serve":
            import uvicorn

            from .api import create_app

            uvicorn.run(
                create_app(settings),
                host=settings.service.host,
                port=settings.service.port,
                workers=1,
                access_log=False,
            )
        elif args.command in {"engine-command", "engine-start"}:
            from .launch import engine_command, start_engine

            if args.command == "engine-start":
                return start_engine(settings)
            if args.json:
                emit(engine_command(settings))
            else:
                print(shlex.join(engine_command(settings)))
        else:
            asyncio.run(doctor(settings, args.backend))
        return 0
    except (RuntimeError, ValueError, OSError, VerifierError) as exc:
        print(f"llmjv: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
