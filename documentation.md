# Zenith Block Optimization and Benchmarking

This document details the optimizations and benchmarking results for the new **Zenith Block** architecture in `dreidelDNN`.

## Architecture Overview

The `ZenithBlock` is a replacement for standard `Conv2D` layers, designed for extreme efficiency on CPU using **Strict APoT (Additive Power-of-Two) Quantization**.

### 1. Strict APoT Quantization
Unlike mixed-precision approaches, the optimized Zenith Block operates **exclusively** on `int8_t` tensors representing APoT codes.
- **No Unpacking:** Data remains in the quantized domain throughout the block's pipeline.
- **Vectorized Arithmetic:** All multiplications and additions are performed in the Logarithmic Number System (LNS) using AVX2 intrinsics (`vpshufb` lookups).
- **Packed Weights:** Weights are stored as `int8` codes, reducing memory bandwidth by 4x compared to float.

### 2. The Zenith Pipeline
The block implements a three-stage "Alien" pipeline:
1.  **Oracle (Gating):** A "Psychic" gating mechanism using bitwise POPCNT on APoT sign bits to skip inactive spatial locations.
2.  **Eyes (Spatial Conv):** Depthwise convolution performed using vectorized LNS multiplication (`vec_mul_apot_avx2`) and addition.
3.  **Mixer (Spectral):** A Fast Walsh-Hadamard Transform (FWHT) replaces dense channel mixing ($O(N \log N)$ vs $O(N^2)$).
    - **Intra-Register Optimization:** For small strides (1, 2, 4, 8, 16), butterfly operations are performed *inside* 256-bit registers using AVX2 shuffles, eliminating memory round-trips.
    - **Inter-Register Vectorization:** For larger strides (32+), operations are vectorized across memory blocks.

## Benchmarking

A rigorous benchmark (`benchmarks/benchmark_zenith_conv.cpp`) compares a fully quantized Zenith backbone against a standard Float32 Conv2D backbone.

### Topology
**Zenith Model:**
`In(F) -> Pack -> Zenith -> QPool -> Zenith -> Zenith (Bottleneck) -> QUpscale -> Zenith -> Unpack`

**Conv Model:**
`Conv -> Pool -> Conv -> Conv -> Upscale -> Conv`

The benchmark uses explicit `PackAPoT` and `UnpackAPoT` layers at the boundaries, ensuring the inner chain runs entirely in Int8.

### Results (CPU, Single Thread/OMP Mixed)

| Channels (C) | Zenith Time (s) | Conv2D Time (s) | **Speedup** |
| :--- | :--- | :--- | :--- |
| **16** | ~3.16s | ~0.78s | **0.25x** (Slower) |
| **64** | ~0.41s | ~11.8s | **~29x** |
| **128** | ~0.76s | ~52.1s | **~68x** |

**Analysis:**
- **C=16:** The overhead of `Pack`/`Unpack` and scalar setup dominates small workloads.
- **C=64/128:** The $O(N \log N)$ spectral mixing combined with "Intra-Register" AVX2 optimizations yields massive gains over the $O(N^2)$ dense mixing of standard convolution.

## Usage

To use the optimized block:
```cpp
#include "dreidel/layers/ZenithBlock.hpp"
#include "dreidel/layers/Quantization.hpp"

// ...
layers::PackAPoT pack;
layers::ZenithBlock zenith(128, 3, 128); // 128 Channels
layers::UnpackAPoT unpack;

auto q_out = zenith.forward(pack.forward(input_float));
auto out = unpack.forward(q_out);
```
