from dataclasses import dataclass, field
from smart_router.config.worker import HealthConfig
from smart_router.config.policy import PolicyConfig
from typing import List
from argparse import Namespace

@dataclass
class SmartRouterConfig:
    router_type: str = "vllm-pd-disagg"

    prefill_urls: List[str] = None
    prefill_intra_dp_size: int = 1
    prefill_bootstrap_ports: List[int] = None

    decode_urls: List[str] = None
    decode_intra_dp_size: int = 1

    health_config: HealthConfig = field(default_factory=HealthConfig)

    prefill_policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    decode_policy_config: PolicyConfig = field(default_factory=PolicyConfig)

def build_config(args: Namespace) -> SmartRouterConfig:
    """
    Build smart router config from args.
    """
    decode_policy_config = None
    if args.decode_policy != "":
        decode_policy_config: PolicyConfig = PolicyConfig(
            policy=args.decode_policy,
            cache_threshold=args.decode_cache_threshold,
            balance_abs_threshold=args.decode_balance_abs_threshold,
            balance_rel_threshold=args.decode_balance_rel_threshold,
        )

    prefill_policy_config = None
    if args.prefill_policy != "":
        prefill_policy_config: PolicyConfig = PolicyConfig(
            policy=args.prefill_policy,
            cache_threshold=args.prefill_cache_threshold,
            balance_abs_threshold=args.prefill_balance_abs_threshold,
            balance_rel_threshold=args.prefill_balance_rel_threshold,
        )

    # default policy config
    policy_config = PolicyConfig(
        policy=args.policy,
        cache_threshold=args.cache_threshold,
        balance_abs_threshold=args.balance_abs_threshold,
        balance_rel_threshold=args.balance_rel_threshold,
    )
   
    return SmartRouterConfig(
        router_type=args.router_type,
        prefill_urls=args.prefill_urls,
        prefill_intra_dp_size=args.prefill_intra_dp_size,
        prefill_bootstrap_ports=getattr(args, "prefill_bootstrap_ports", None),
        decode_urls=args.decode_urls, 
        decode_intra_dp_size=args.decode_intra_dp_size,
        health_config=HealthConfig(
            check_interval_secs=getattr(args, "health_check_interval", 60),
        ),
        decode_policy_config=decode_policy_config if decode_policy_config else policy_config,
        prefill_policy_config=prefill_policy_config if prefill_policy_config else policy_config,
    )
