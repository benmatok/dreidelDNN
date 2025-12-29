# Zenith Technology: Alien Speed on CPU

**Zenith Technology** is the core innovation behind dreidelDNN, enabling "Alien Speed" execution on standard CPUs without requiring GPUs. It achieves this by fundamentally rethinking how neural network operations are computed, moving away from standard floating-point arithmetic (GEMM) towards bitwise logic, table lookups, and spectral transforms.

## 1. APoT (Additive Power-of-Two) Quantization

Standard quantization uses linear integers (int8) to approximate floats. **APoT** uses a **Logarithmic Number System (LNS)** base-2 representation.

*   **Representation**: A value $x$ is represented as $(-1)^s \cdot 2^{k}$.
*   **Storage**: We store the integer exponent $k$ (plus a bias) in an 8-bit integer.
    *   `0x00` represents 0.
    *   `0x01` to `0x7F` represent exponents.
    *   `0x80` is the sign bit.
*   **Multiplication becomes Addition**:
    $$ x \cdot y = ((-1)^{s_x} 2^{k_x}) \cdot ((-1)^{s_y} 2^{k_y}) = (-1)^{s_x \oplus s_y} \cdot 2^{k_x + k_y} $$
    Multiplication is reduced to integer addition of exponents, which is extremely fast.

In `ZenithBlock`, weights are constrained to be exact powers of two, meaning no multiplication logic is ever needed—only bitwise XOR (for signs) and integer ADD (for exponents).

## 2. Shuffle AVX LUT (Lookup Tables)

A key bottleneck in LNS arithmetic is **Addition** ($z = x + y$). While multiplication is easy, addition is hard:
$$ x + y = x (1 + y/x) = x (1 + 2^{k_y - k_x}) $$

To solve this without converting back to float, we use **SIMD Shuffle Instructions** as parallel Lookup Tables (LUTs).

*   **Mechanism**:
    *   Compute difference $d = |k_x - k_y|$.
    *   Use `vpshufb` (AVX2) or `vpermb` (AVX-512) to look up a correction term $F(d) \approx \log_2(1 + 2^{-d})$ from a 16-entry table stored in a register.
    *   $z \approx \max(x, y) + F(d)$.
*   **Performance**: This allows performing 16 (AVX2) or 64 (AVX-512) non-linear function evaluations in a single cycle, bypassing the need for complex ALU operations.

## 3. In-Register FWHT (Spectral Mixing)

The **Zenith Block** replaces dense matrix multiplication ($O(N^2)$) with the **Fast Walsh-Hadamard Transform (FWHT)** ($O(N \log N)$) for channel mixing.

*   **The Problem**: Standard FWHT implementations are memory-bound. They recursively load/store data to RAM/L1 cache.
*   **The Zenith Solution**:
    *   We implement **Intra-Register FWHT**. For small strides (1, 2, 4, 8, 16), the butterfly operations are performed *entirely inside* the AVX-256 or AVX-512 registers using shuffle instructions.
    *   Data is loaded once, transformed in-place within registers, and stored once.
    *   This eliminates load/store traffic for the inner loops, hitting the theoretical compute peak of the CPU.

## 4. Soft Permutation

Spectral transforms like FWHT are fixed and linear. To add learnability and local interaction, we introduce **Soft Permutation**.

*   **Concept**: Instead of a hard index permutation (swapping wires), we use parameterized $2 \times 2$ block rotations (Givens rotations) or local mixing.
*   **Implementation**:
    *   In the `ZenithBlock`, this is often integrated into the "Mixer" stage.
    *   It allows the network to learn "ghost permutations"—virtual wirings that optimize the spectral mixing capability without the overhead of full dense matrices.
    *   Implemented via efficient APoT arithmetic ($x' = x + y, y' = x - y$ variants) directly in the spectral pipeline.

## Summary

The **Zenith Block** combines these technologies into a single, cohesive primitive:
1.  **Input** (int8 APoT)
2.  **Eyes** (Spatial Convolution via APoT Add/Mul)
3.  **Mixer** (In-Register FWHT + Soft Permutation)
4.  **Output** (int8 APoT)

This pipeline runs entirely on integer units, using vector shuffles for complex logic, realizing the "Alien Speed" vision.
