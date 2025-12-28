# 🛸 Project Jules: The "Alien Speed" Master Plan (Roadmap)

**Engine:** `dreidelDNN` (Header-Only C++)
**Core Unit:** The Zenith Block
**Optimization Strategy:** Replace Arithmetic with Logic; Replace Memory Access with Register Shuffles.

---

## 1. The "Psychic" Oracle (Optimized ALSH) [x]

**Goal:** Determine which blocks to compute without actually computing anything expensive.
**Status:** **Implemented** in `hal/ops.hpp` (`AlienOps::popcnt32`, `vec_sign_mask`) and `layers/ZenithBlock.hpp`.

* **Mechanism:** Binary Hyperplanes.
* **The Cheat:**
    1. **Sign Extraction:** Extract sign bit (`x >> 31`). [x]
    2. **XOR & Popcount:** `POPCNT(input_mask ^ oracle_mask)`. [x]
    3. **Speedup:** Replaces projection MM with single-cycle logic.

---

## 2. The "Teleporting" Eyes (Optimized Spatial LUT) [x]

**Goal:** 3x3 Convolution with zero multiplication and zero L1 cache latency.
**Status:** **Implemented** in `hal/ops.hpp` (`AlienOps::lut_lookup_16`) and `layers/ZenithBlock.hpp`.

* **Mechanism:** APoT Weights + 4-bit Nibble Inputs.
* **The Cheat:**
    1. **Load Table:** 16-entry LUT in ZMM/YMM register. [x]
    2. **Instruction:** `vpermb` (AVX-512) or `vpshufb` (AVX2) to shuffle results. [x]
    3. **Speedup:** 32-64 simultaneous lookups per cycle, bypassing L1 cache.

---

## 3. The "Register" Mixer (Optimized FWHT) [ ]

**Goal:** Mix information globally without memory thrashing.
**Status:** **Partially Implemented.** Uses `algo::WHT::fwht_1d` (Iterative Loop). **Pending:** Register-Resident implementation.

* **Mechanism:** Register-Resident Butterfly.
* **The Cheat:**
    1. **Vertical Loading:** Load block of channels into registers.
    2. **In-Register Permutation:** `_mm512_shuffle_ps` + `_mm512_add_ps`.
    3. **Speedup:** 100% ALU throughput, zero intermediate memory writes.

---

## 4. The "Ghost" Permutation (Optimized Shuffle) [ ]

**Goal:** Soft Permutation without random memory access.
**Status:** **Pending.** Currently uses `std::vector` indices and runtime copy.

* **Mechanism:** Compile-Time Constant Shuffles.
* **The Cheat:**
    1. **Instruction:** `_mm512_permute_ps` with immediate mask.
    2. **Logic:** Bake permutation into assembly at compile time.

---

## 5. The "Branchless" Gate (Optimized Activation) [ ]

**Goal:** Apply Thresholding without pipeline stalls.
**Status:** **Pending.** Uses standard layers.

* **Mechanism:** Bit-Masking Logic.
* **The Cheat:**
    1. **Mask:** `mask = x >> 31`.
    2. **Logic:** `result = x & (~mask)`.

---

## 6. The Memory Model: "Super-Block" Tiling [ ]

**Goal:** Efficient RAM access.
**Status:** **Pending.** Zero-Allocation `Arena` is implemented [x], but Z-Curve tiling is not.

* **Mechanism:** Morton Order (Z-Curve) Tiling.
* **The Cheat:**
    1. **Layout:** Store images in 64-byte Z-Order blocks.
    2. **Speedup:** Minimizes TLB misses.

---

## Implementation Roadmap

### Phase 1: Core Alien Primitives (Completed)
- [x] **`include/dreidel/hal/ops.hpp`**: `AlienOps` struct with `popcnt`, `lut_lookup`, `sign_mask`.
- [x] **`include/dreidel/core/Memory.hpp`**: Zero-Allocation `Arena`.
- [x] **`include/dreidel/layers/ZenithBlock.hpp`**: Integrated Phase 1 (Oracle) and Phase 2 (Eyes).

### Phase 2: Advanced Optimizations (Planned)
- [ ] **Phase 3:** Register-Resident FWHT Mixer.
- [ ] **Phase 4:** Ghost Permutation (Template-based shuffling).
- [ ] **Phase 5:** Branchless Activations.
- [ ] **Phase 6:** Z-Curve Memory Layout in `Tensor` or `Arena`.

### Phase 3: Tooling
- [ ] **`tools/Baker.cpp`**: Pre-processor for generating constexpr masks.

---

## 7. Benchmark Validation & Diagnostics

**Goal:** Prove "Alien" superiority over Standard `Conv2D` (Im2Col + GEMM) and diagnose bottlenecks.

### A. The "Why No Speedup?" Diagnosis

If `vpermb` (Eyes) is fast but the overall block is slow, check these **Performance Killers**:

1. **The Copy Penalty (Phase 4):** You mentioned using `std::vector` indices for permutation. **This is fatal.** Random memory reads (Scatter/Gather) are ~100x slower than the math you saved.
* *Fix:* You **must** implement Phase 4 (Ghost Permutation) or at least a hard-coded pre-computed shuffle to see gains.


2. **Cache Thrashing (Phase 3):** Iterative FWHT reads/writes the same array (N)$ times. If that array spills out of L1 Cache, you are memory-bound.
* *Fix:* The Register-Resident Mixer is mandatory to hide this latency.


3. **ALSH Overhead (Phase 1):** If the Oracle doesn't skip at least **80-90%** of blocks, the overhead of calculating the hash (`POPCNT`) might cost more than the savings.

### B. Speedup Estimates (Target vs. Current)

| Component | Standard Baseline | Current Implementation (Est.) | **Final "Alien" Target** | Why? |
| --- | --- | --- | --- | --- |
| **1. The Oracle** | N/A (Always Compute) | **0.8x (Slower)** | **5x - 20x** | Needs high sparsity (>90%) to pay off the `POPCNT` cost. |
| **2. The Eyes** | `FMUL` Latency (4 cycles) | **10x (L1 Hit) / 1x (RAM)** | **~30x** | `vpermb` is instant, but only if inputs are already in registers (Phase 6 needed). |
| **3. The Mixer** | `GEMM` (^2$) | **2x - 5x** | **50x** | Iterative FWHT is cache-heavy. Register FWHT is cache-free. |
| **4. Permutation** | Free (Fixed topology) | **0.1x (Disaster)** | **100x (Instant)** | `std::vector` copies are the bottleneck right now. |
| **TOTAL** | **1.0x** | **~0.9x - 1.5x** | **~25x** | **Memory movement is eating your math gains.** |

### C. Validation Tests (Micro-Benchmarks)

Add these specific tests to `tests/benchmark_alien.cpp`:

**Test 1: The "L1 Resident" Eye Test**

* **Setup:** Pre-load a small 32KB chunk of data (fits in L1). Run `ZenithBlock::Eyes` 1,000,000 times.
* **Goal:** Verify `vpermb` throughput.
* **Pass Criteria:** Should be >10x faster than standard float convolution loops.
* **Fail?** If not, your compiler isn't generating `vpermb`. Check assembly (`objdump -d`).

**Test 2: The "Memory Wall" Test**

* **Setup:** Run `ZenithBlock` on a 100MB image (blows out L3 cache).
* **Goal:** Measure the impact of **Phase 6 (Z-Curve)**.
* **Expected:** Without Z-Curve, this will be slow. This baseline proves the need for Phase 6.

**Test 3: The "Sparsity" Breakeven Analysis**

* **Setup:** Run the Oracle with different "Active Block" percentages (10%, 50%, 100%).
* **Goal:** Find the "Breakeven Point."
* **Hypothesis:** You likely need < 20% active blocks for the overhead to make sense currently.

### D. Immediate Action Plan

1. **Kill the `std::vector` copy immediately.** Even a hard-coded, ugly manual shuffle loop is better than a generic vector copy.
2. **Inspect Assembly:** Run `objdump -d -M intel your_binary | grep vpermb`. If you don't see it, `AlienOps::lut_lookup` is failing to vectorize.
3. **Profile L1 Misses:** Use `perf stat -e L1-dcache-load-misses ./your_benchmark`. High misses = You need Phase 6 (Z-Curve) and Phase 3 (Register Mixer).
