from dataclasses import dataclass


@dataclass
class HealthConfig:
    timeout_secs: int = 5
    check_interval_secs: int = 60
    endpoint: str = "/health"
