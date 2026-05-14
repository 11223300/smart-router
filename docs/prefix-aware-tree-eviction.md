# Prefix-Aware Tree Eviction

`prefix_aware` routing policy 会在 router 内存中维护一棵按 worker/tenant 区分的前缀树，用于把相似 prompt 尽量调度到已有前缀缓存的 worker。为了避免前缀树随着请求不断增长，策略默认启动一个后台 LRU 驱逐线程。

## 默认行为

默认每个 worker/tenant 的前缀树字符预算如下：

- 高水位：`2_000_000` characters
- 驱逐目标：`1_600_000` characters
- 检查间隔：`30` seconds

后台线程每隔 `prefix_cache_eviction_interval_secs` 检查一次。如果某个 tenant 的字符数超过高水位，就按该 tenant 的 LRU 顺序删除旧节点，直到降到目标水位附近。

## 配置参数

全局参数会同时作用于 prefill 和 decode policy，除非被各自的专用参数覆盖：

```bash
--prefix-cache-eviction-threshold-chars 2000000
--prefix-cache-eviction-target-chars 1600000
--prefix-cache-eviction-interval-secs 30
```

prefill 专用参数：

```bash
--prefill-prefix-cache-eviction-threshold-chars 2000000
--prefill-prefix-cache-eviction-target-chars 1600000
--prefill-prefix-cache-eviction-interval-secs 30
```

decode 专用参数：

```bash
--decode-prefix-cache-eviction-threshold-chars 2000000
--decode-prefix-cache-eviction-target-chars 1600000
--decode-prefix-cache-eviction-interval-secs 30
```

## 关闭驱逐

将 threshold 设置为 `0` 或负数即可关闭驱逐线程：

```bash
--prefix-cache-eviction-threshold-chars 0
```

不要用 `--prefix-cache-eviction-interval-secs 0` 表示关闭。`interval_secs` 必须大于 `0`；设置为 `0` 会被视为非法配置并抛出 `ValueError`。

## 调度影响

驱逐线程在 idle 时只会 sleep，不会阻塞调度。真正执行驱逐时，它会短暂持有 `PrefixTree` 的锁；同一时间的 prefix match、insert 或 worker remove 需要等待这把锁释放。

因此当前实现不是 lock-free，但默认高水位配置下，驱逐只会周期性发生，影响通常很小。如果线上需要更严格的尾延迟控制，可以后续把单次驱逐改成分批清理，每批释放一部分节点后主动让出锁。
