from dataclasses import dataclass

@dataclass
class PolicyConfig:

    policy: str = "round_robin"  # default policy

    cache_threshold: float = 0.5

    balance_abs_threshold: int = 5

    balance_rel_threshold: float = 2.0

    prefix_cache_eviction_threshold_chars: int = 2_000_000

    prefix_cache_eviction_target_chars: int = 1_600_000

    prefix_cache_eviction_interval_secs: float = 120.0
