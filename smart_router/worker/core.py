# -*- coding: utf-8 -*-
"""Worker base class and abstract interface."""

from __future__ import annotations

import asyncio

import enum
from dataclasses import dataclass, field
from smart_router.config.worker import HealthConfig

import httpx


# shared client (httpx) for workers
WORKER_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(30.0))

@dataclass
class WorkerMetadata:
    """Metadata describing a worker, including its URL, type, and health check configuration."""
    url: str
    worker_type: WorkerType
    health_config: HealthConfig = field(default_factory=HealthConfig)

class WorkerType(str, enum.Enum):
    REGULAR = "regular"
    PREFILL = "prefill"
    DECODE = "decode"

# ========================================================
## Worker interface and base implementation
# ========================================================
class Worker:
    """Abstract base class describing the worker interface.
    
    This class defines the contract that all worker implementations must follow.
    Concrete implementations include BasicWorker and DPAwareWorker.
    """
    def url(self) -> str:
        """Get the worker's URL."""
        raise NotImplementedError()

    def worker_type(self) -> WorkerType:
        """Get the worker's type (Regular, Prefill, or Decode)."""
        raise NotImplementedError()

    def is_healthy(self) -> bool:
        """Check if the worker is currently healthy."""
        raise NotImplementedError()

    def set_healthy(self, healthy: bool) -> None:
        """Set the worker's health status."""
        raise NotImplementedError()

    async def check_health_async(self) -> bool:
        """Perform an async health check on the worker and update its state."""
        raise NotImplementedError()

    def check_health(self) -> bool:
        """Synchronous health check wrapper (for compatibility)."""
        return asyncio.get_event_loop().run_until_complete(self.check_health_async())

    def load(self) -> int:
        """Get the current load (number of active requests)."""
        raise NotImplementedError()

    def increment_load(self, load: int = 1) -> None:
        """Increment the load counter."""
        raise NotImplementedError()

    def decrement_load(self, load: int = 1) -> None:
        """Decrement the load counter."""
        raise NotImplementedError()
    
    def decrement_load_with_context(self, load: int = 1):
        """Context manager to decrement load after the block, even if exceptions occur."""
        raise NotImplementedError()

    def reset_load(self) -> None:
        """Reset the load counter to 0 (for sync/recovery)."""
        pass

    def metadata(self) -> WorkerMetadata:
        """Get worker-specific metadata."""
        raise NotImplementedError()

    def is_available(self) -> bool:
        """Check if the worker is available for routing."""
        return self.is_healthy()

    # ===== Convenience helpers for metadata labels =====

    def model_id(self) -> str:
        """Get the model ID this worker serves."""
        return self.metadata().labels.get("model_id", "unknown")
    
    def endpoint_url(self, route: str) -> str:
        """Get the actual endpoint URL for requests (uses base URL without @rank)."""
        raise NotImplementedError()
    
    def base_url(self) -> str:
        """Get the base URL without DP suffix."""
        raise NotImplementedError()

    
    # ===== DP-aware specific methods (not part of Worker interface) =====
    def is_dp_aware(self) -> bool:
        """Check if this worker is DP-aware (data-parallel aware)."""
        return False

    def dp_rank(self) -> int:
        """Get the DP rank of this worker."""
        return -1

    def dp_size(self) -> int:
        """Get the total DP group size."""
        if self.is_dp_aware():
            raise NotImplementedError("DP-aware workers must implement dp_size()")

