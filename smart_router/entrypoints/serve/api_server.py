import asyncio
import logging
import os
import platform

import sys
import tempfile
import uvicorn

from multiprocessing import Process
from typing import Optional

from starlette.applications import Starlette
from starlette.routing import Route

from smart_router.config import build_config, build_parser
from smart_router.engine.engine_client import EngineClient
from smart_router.engine.vllm_engine import start_engine
from smart_router.engine.sglang_engine import start_sglang_engine
from smart_router.entrypoints.serve.vllm_routes import VllmRoutes
from smart_router.entrypoints.serve.sglang_routes import SGLangRoutes
from smart_router.logger import init_logging

logger =logging.getLogger(__name__)

# Detect 0S
is_linux = platform.system() == "Linux"


def _get_zmq_addresses():
    """Generate ZMg addresses. Use unique IPc paths on Linux to avoid conflicts."""
    if is_linux:
        # Use temp directory with PID to avoid conflicts between instances
        # ipc_dir = os.path.join(tempfile.gettempdir(),f"smart-router-{os.getpid()}")
        main_pid = os.environ.get("_SMART_ROUTER_MAIN_PID")
        if main_pid is None:
            # We are in the main process - record our PID
            main_pid = str(os.getpid())
            os.environ["_SMART_ROUTER_MAIN_PID"] = main_pid
        ipc_dir = os.path.join(tempfile.gettempdir(), f"smart-router-{main_pid}")
        os.makedirs(ipc_dir, exist_ok=True)
        return (
            f"ipc://{os.path.join(ipc_dir, 'output.ipc')}",
            f"ipc://{os.path.join(ipc_dir, 'input.ipc')}"
        )
    else:
        return "tcp://127.0.0.1:5558", "tcp://127.0.0.1:5557"


# Module-level ZMQ addresses (resolved lazily, same for all workers in same process group)
output_addr: Optional[str] = None
input_addr: Optional[str] = None

# Global reference to receive_loop task for cleanup
_receive_task: Optional[asyncio.Task] = None

# Module-level config and app(populated by _init_app or main)
config = None
app: Starlette


def _build_app(config):
    """Build Starlette app with routes based on router_type."""
    router_type = config.router_type

    if router_type == "sglang-pd-disagg":
        sglang_routes = SGLangRoutes(bootstrap_ports=config.prefill_bootstrap_ports)
        routes = [
            Route("/v1/chat/completions", sglang_routes.chat_completions, methods=["POST"]),
            Route("/v1/completions", sglang_routes.completions, methods=["POST"]),
            Route("/generate", sglang_routes.generate, methods=["POST"]),
        ]

    else:
        # Default: vllm-pd-disagg
        vllm_routes = VllmRoutes()
        routes = [
            Route("/v1/chat/completions", vllm_routes.chat_completions, methods=["POST"])
        ]
    application = Starlette(
        routes=routes,
        on_startup=[startup],
        on_shutdown=[shutdown],
    )
    return application


def _init_app():
    """Initialize app from sys.argv. Called only when needed (not on import)."""
    global app, _config, output_addr, input_addr

    output_addr, input_addr =_get_zmq_addresses()

    _argv = sys.argv[1:]
    if _argv and _argv[0]== "serve":
        _argv = _argv[1:]
    _args = build_parser().parse_args(_argv)
    _config = build_config(_args)
    app = _build_app(_config)


async def startup():
    """Initialize Engineclient and start receive loop for each worker process."""
    global _receive_task
    app.state.engine_client = EngineClient(input_addr, output_addr)
    _receive_task = asyncio.create_task(app.state.engine_client.receive_loop())
    logger.info(f"Engineclient started with identity: {app.state.engine_client.identity}")


async def shutdown():
    """Gracefully shutdown Engineclient: close sockets first, then cancel receive loop."""
    global _receive_task

    engine_client = getattr(app.state, "engine_client", None)
    if engine_client is None:
        logger.warning("EngineClient not found during shutdown")
        return

    # Close sockets first -this causes receive_loop to exit naturally
    # on the next recv_multipart()instead of losing in-flight messages.
    await engine_client.shutdown()
    logger.info(f"Engineclient sockets closed: {engine_client.identity}")

    # Then cancel the receive task (it should exit on its own after socket close
    # but cancel as a safety net)
    if _receive_task is not None and not _receive_task.done():
        _receive_task.cancel()
        try:
            await _receive_task
        except asyncio.CancelledError:
            pass
        logger.info("Receive loop cancelled")

    logger.info(f"Engineclient shutdown complete: {engine_client.identity}")


def main(argv: list[str]|None = None) -> int:
    global app, _config, output_addr, input_addr
    parser = build_parser()
    args = parser.parse_args(argv)

    # Build config
    config = build_config(args)

    init_logging(args.log_level)

    # Resolve ZMQ addresses
    output_addr, input_addr = _get_zmq_addresses()

    # Select engine based on router_type
    if config.router_type == "sglang-pd-disagg":
        engine_target = start_sglang_engine
    else:
        engine_target = start_engine

    # Start engine process
    engine_process = Process(
        target=engine_target,
        args=(config,input_addr,output_addr),
        name="Router Engine",
    )
    engine_process.start()
    logger.info(f"Engine process started with PID: {engine_process.pid}")

    # Build app for uvicorn import path
    _config = config
    app = _build_app(config)

    # Track engine process for cleanup, avoid signal handler conflicts with uvicorn.
    # Instead of overriding signal handlers, use atexit + uvicorn's own signal handling.
    import atexit

    def cleanup_engine():
        if engine_process.is_alive():
            logger.info("Terminating engine process...")
            engine_process.terminate()
        engine_process.join(timeout=10)
        if engine_process.is_alive():
            logger.warning("Engine process did not terminate gracefully, killing...")
            engine_process.kill()
            engine_process.join()
        logger.info("Engine process stopped")

        # Cleanup IPc files on Linux
        if is_linux:
            # ipc_dir = os.path.join(tempfile.gettempdir(), f"smart-router-{os.getpid()}")
            main_pid = os.environ.get("_SMART_ROUTER_MAIN_PID", str(os.getpid()))
            ipc_dir = os.path.join(tempfile.gettempdir(), f"smart-router-{main_pid}")
            for fname in ("output.ipc", "input.ipc"):
                fpath = os.path.join(ipc_dir, fname)
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass

    atexit.register(cleanup_engine)

    try:
        uvicorn.run(
            "smart_router.entrypoints.serve.api_server:app",
            host=args.host,
            port=args.port,
            workers=args.apiserver_workers,
        )
    finally:
        # Ensure cleanup runs even if atexit doesn't(e.g. signal)
        cleanup_engine()
        atexit.unregister(cleanup_engine)

    return 0


# Lazy initialization: only parse argv and build app when actually needed.
# This prevents crashes on import (e.g. in tests or when importing for main()).
# uvicorn workers will trigger this via the module-level app access below,
# but only after the main process has already set things up.
def __getattr__(name):
    """Lazy module attribute access - build app on first access to 'app'."""
    if name == "app":
        _init_app()
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())