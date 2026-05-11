# -*- coding: utf-8 -*-

# Re-export all public types and functions for convenient access
from smart_router.worker.basic_worker import BasicWorker
from smart_router.worker.core import (
    WORKER_CLIENT,
    WorkerMetadata,
    WorkerType,
)
from smart_router.worker.dp_aware_worker import DPAwareWorker
from smart_router.worker.core import Worker
from smart_router.worker.worker_registry import (
    WorkerRegistry,
    get_available_workers,
    get_healthy_workers,
)

__all__ = [
    # Core types
    "Worker",
    "BasicWorker",
    "DPAwareWorker",
    "WORKER_CLIENT",
    "WorkerMetadata",
    "WorkerType",
    "WorkerRegistry",
    "get_healthy_workers",
    "get_available_workers",
    # Shared client
    "WORKER_CLIENT",
]
