# Zenith Block Optimization Results

## Greedy Search Analysis

We ran a greedy search benchmark (`tests/benchmark_gems.cpp`) 4 times on the AVX2 environment to identify the optimal combination of "Gems".

### Run Summary

| Run | Baseline (ms) | Winner Step | Speedup | Notes |
|---|---|---|---|---|
| 1 | 23.93 | Lazy Norm | 1.06x | Streaming Store offered 1.00x (no gain). |
| 2 | 31.25 | Lazy Norm | 1.31x | Significant variance in baseline. Streaming Store regressed (0.61x) when combined. |
| 3 | 30.42 | Streaming Store | 1.24x | Lazy Norm regressed (0.93x) in isolation here. |
| 4 | 27.27 | Lazy Norm | 1.18x | Consistent gain. Streaming Store offered 1.14x in isolation but regressed combined. |

### Conclusion

1.  **Lazy Normalization:** Consistently provided the best or near-best speedup (6-31%) in 3 out of 4 runs. Removing the explicit division loop is a clear win.
2.  **Streaming Stores:** Provided speedup in isolation (up to 24%) but often **regressed** when combined with Lazy Norm or in different cache states. This suggests that for `C=64` (small payload per pixel), the overhead of managing non-temporal hints or the impact on cache coherence for subsequent layers (GroupNorm reads this immediately!) outweighs the write bandwidth savings.
    *   *Decision:* **Do not integrate Streaming Stores** for this block size configuration, as the output is immediately consumed by GroupNorm in the cache.
3.  **Approx Reciprocal:** Provided inconsistent results (sometimes 0.95x, sometimes 1.14x). Given the potential precision loss and lack of consistent speedup, it is safer to omit.
4.  **Integer Path (C=16):** The mocked overhead showed essentially neutral performance (1.03x speedup to 0.92x regression). Since it adds complexity and requires weight quantization infrastructure not present, it is **omitted** for now.

### Final Configuration

*   **Enabled:** Lazy Normalization.
*   **Disabled:** Streaming Stores, Approx Reciprocal, Integer Path.

This configuration yields a robust **~10-15% speedup** on average without risking regressions from cache management or precision issues.
