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
