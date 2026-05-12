import time

from smart_router.policies.prefix_tree import PrefixTree


def _wait_for(predicate, timeout_secs: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_eviction_loop_removes_lru_entries_until_under_target():
    tree = PrefixTree()
    tree.add_tenants(["worker-a"], time_s=1.0)
    tree.insert("a" * 12, "worker-a", time_s=2.0)
    tree.insert("b" * 12, "worker-a", time_s=3.0)

    assert tree.tenant_to_char_count["worker-a"] == 24
    assert tree.start_eviction_loop(
        eviction_threshold=10,
        eviction_target=5,
        interval_secs=0.01,
    )

    try:
        assert _wait_for(lambda: tree.tenant_to_char_count["worker-a"] <= 5)
    finally:
        tree.stop_eviction_loop()


def test_stop_eviction_loop_is_idempotent():
    tree = PrefixTree()

    assert tree.start_eviction_loop(
        eviction_threshold=10,
        eviction_target=5,
        interval_secs=30.0,
    )

    tree.stop_eviction_loop()
    tree.stop_eviction_loop()

    assert tree._eviction_thread is None


def test_remove_tenants_with_eviction_loop_running():
    tree = PrefixTree()
    tree.add_tenants(["worker-a"], time_s=1.0)
    tree.insert("a" * 20, "worker-a", time_s=2.0)
    assert tree.start_eviction_loop(
        eviction_threshold=10,
        eviction_target=5,
        interval_secs=0.01,
    )

    try:
        assert tree.remove_tenants(["worker-a"])["worker-a"] >= 0
        assert "worker-a" not in tree.tenant_to_char_count
        time.sleep(0.05)
        assert "worker-a" not in tree.tenant_to_char_count
    finally:
        tree.stop_eviction_loop()
