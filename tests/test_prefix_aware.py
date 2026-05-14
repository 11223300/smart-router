import pytest

from smart_router.config import PolicyConfig
from smart_router.policies.prefix_aware import PrefixAwarePolicy


def test_prefix_aware_policy_starts_eviction_loop_by_default():
    policy = PrefixAwarePolicy(PolicyConfig(policy="prefix_aware"))

    try:
        assert policy._eviction_started is True
        assert policy.tree._eviction_thread is not None
        assert policy.tree._eviction_thread.is_alive()
    finally:
        policy.stop()


def test_prefix_aware_policy_disables_eviction_when_threshold_is_not_positive():
    policy = PrefixAwarePolicy(
        PolicyConfig(
            policy="prefix_aware",
            prefix_cache_eviction_threshold_chars=0,
        )
    )

    try:
        assert policy._eviction_started is False
        assert policy.tree._eviction_thread is None
    finally:
        policy.stop()


def test_prefix_aware_policy_rejects_invalid_eviction_target():
    with pytest.raises(ValueError, match="target"):
        PrefixAwarePolicy(
            PolicyConfig(
                policy="prefix_aware",
                prefix_cache_eviction_threshold_chars=10,
                prefix_cache_eviction_target_chars=10,
            )
        )


def test_prefix_aware_policy_rejects_invalid_eviction_interval():
    with pytest.raises(ValueError, match="interval"):
        PrefixAwarePolicy(
            PolicyConfig(
                policy="prefix_aware",
                prefix_cache_eviction_threshold_chars=10,
                prefix_cache_eviction_target_chars=5,
                prefix_cache_eviction_interval_secs=0,
            )
        )
