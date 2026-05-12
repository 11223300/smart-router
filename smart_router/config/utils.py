
import argparse


SUPPORT_POLICIES = ["round_robin", "power_of_two", "prefix_aware", "consistent_hash", "minimum_load"]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # apis
    parser.add_argument("--host", default="0.0.0.0", help="The host to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, help="The port to bind the server to.")
    parser.add_argument("--apiserver-workers", type=int, default=8, help="The number of worker processes for the API server.")
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=60,
        help="Seconds between full worker health checks.",
    )
    parser.add_argument(
        "--enable-k8s-discovery",
        action="store_true",
        help="Discover prefill/decode workers from Kubernetes pods.",
    )
    parser.add_argument(
        "--k8s-prefill-port",
        type=int,
        help="Port used to build discovered prefill worker URLs.",
    )
    parser.add_argument(
        "--k8s-decode-port",
        type=int,
        help="Port used to build discovered decode worker URLs.",
    )
    parser.add_argument(
        "--k8s-namespace",
        help="Kubernetes namespace to watch. Defaults to the service account namespace.",
    )
    parser.add_argument(
        "--k8s-task-label-key",
        default="task_id",
        help="Pod label key used to group router and workers into one inference task.",
    )
    parser.add_argument(
        "--k8s-task-id",
        help="Task id label value to watch. Defaults to the router pod's own label value.",
    )
    parser.add_argument(
        "--k8s-url-scheme",
        default="http",
        choices=["http", "https"],
        help="URL scheme used for discovered worker URLs.",
    )

    # overview
    parser.add_argument(
        "--policy", 
        default="round_robin", 
        choices=SUPPORT_POLICIES, 
        help="The routing policy to use. This can be overridden by --prefill-policy and --decode-policy."   )
    parser.add_argument("--cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for prefill.")
    parser.add_argument("--balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefix-cache-eviction-threshold-chars", type=int, default=2_000_000, help="Per-worker prefix tree character high watermark for prefix-aware policy. Set <= 0 to disable eviction.")
    parser.add_argument("--prefix-cache-eviction-target-chars", type=int, default=1_600_000, help="Per-worker prefix tree character target after eviction for prefix-aware policy.")
    parser.add_argument("--prefix-cache-eviction-interval-secs", type=float, default=30.0, help="Seconds between prefix-aware tree eviction checks.")
    
    parser.add_argument(
        "--router_type",
        default="vllm-pd-disagg", 
        choices=["vllm-pd-disagg", "sglang-pd-disagg", "discovery"], 
        help="The routing type to use.")

    # prefill
    parser.add_argument("--prefill-urls", nargs="+")
    parser.add_argument("--prefill-intra-dp-size", type=int, default=1)
    parser.add_argument("--prefill-policy", default="", choices=[""]+ SUPPORT_POLICIES, help="The routing policy to use for prefill. Overrides --policy if set.")
    parser.add_argument("--prefill-cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-prefix-cache-eviction-threshold-chars", type=int, default=2_000_000, help="Per-prefill-worker prefix tree character high watermark for prefix-aware policy. Set <= 0 to disable eviction.")
    parser.add_argument("--prefill-prefix-cache-eviction-target-chars", type=int, default=1_600_000, help="Per-prefill-worker prefix tree character target after eviction for prefix-aware policy.")
    parser.add_argument("--prefill-prefix-cache-eviction-interval-secs", type=float, default=30.0, help="Seconds between prefill prefix-aware tree eviction checks.")

    # decode
    parser.add_argument("--decode-urls", nargs="+")
    parser.add_argument("--decode-intra-dp-size", type=int, default=1)
    parser.add_argument("--decode-policy", default="", choices=[""]+SUPPORT_POLICIES, help="The routing policy to use for decode. Overrides --policy if set.")
    parser.add_argument("--decode-cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-prefix-cache-eviction-threshold-chars", type=int, default=2_000_000, help="Per-decode-worker prefix tree character high watermark for prefix-aware policy. Set <= 0 to disable eviction.")
    parser.add_argument("--decode-prefix-cache-eviction-target-chars", type=int, default=1_600_000, help="Per-decode-worker prefix tree character target after eviction for prefix-aware policy.")
    parser.add_argument("--decode-prefix-cache-eviction-interval-secs", type=float, default=30.0, help="Seconds between decode prefix-aware tree eviction checks.")

    # logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser
