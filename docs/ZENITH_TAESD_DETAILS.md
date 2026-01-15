# Zenith-TAESD: Architecture & Performance Analysis

## Overview

**Zenith-TAESD** is a CPU-optimized Autoencoder architecture designed to replace the standard ResNet-based "Tiny AutoEncoder for Stable Diffusion" (TAESD). It leverages **Spectral Gating** via the **ZenithLite Block** to achieve global receptive fields with linear complexity ($O(N)$), significantly reducing the computational burden compared to dense convolutions ($O(N \cdot K^2)$).

We introduce two variants:
1.  **Zenith-TAESD**: A hybrid architecture using standard 3x3 convolutions for downsampling/upsampling and ZenithLite blocks for feature mixing.
2.  **Zenith-TAESD-Lite**: An ultra-lightweight variant replacing all 3x3 convolutions with **1x1 Convs** combined with **PixelShuffle/Unshuffle**, maximizing throughput on CPU.

---

## Architecture Comparison

### 1. Vanilla TAESD (Baseline)
*   **Structure**: Deep stack of `ResNetBlock` layers (Conv3x3 $\to$ ReLU $\to$ Conv3x3).
*   **Downsampling**: Strided Conv3x3.
*   **Upsampling**: Nearest Neighbor + Conv3x3 (or ConvTranspose).
*   **Characteristics**: High arithmetic intensity. Memory bandwidth usage is moderate, but FLOPs are very high due to spatial sliding windows.
*   **Complexity**: High ($O(H \cdot W \cdot C \cdot K^2)$).

### 2. Zenith-TAESD
*   **Core Component**: `ZenithLiteBlock`.
    *   **Compress**: 1x1 Group Conv (Groups=4).
    *   **Row Mix**: 1D Fast Walsh-Hadamard Transform (FWHT) $\to$ Spectral Gate $\to$ Inverse FWHT.
    *   **Col Mix**: Transpose $\to$ 1D FWHT $\to$ Spectral Gate $\to$ Inverse FWHT $\to$ Transpose.
    *   **Expand**: 1x1 Group Conv.
*   **Down/Up**: Uses `OptimizedConv2D` (3x3) for high-quality feature extraction at resolution changes.
*   **Advantage**: Replaces heavy ResNet blocks with efficient Spectral Mixers. Reduces FLOPs while maintaining global context.

### 3. Zenith-TAESD-Lite (Fastest)
*   **Structure**: Pure 1x1 Convolutions + Spectral Mixing.
*   **Downsampling**: `PixelUnshuffle` (Space-to-Depth) $\to$ `OptimizedConv2D` (1x1).
*   **Upsampling**: `OptimizedConv2D` (1x1) $\to$ `PixelShuffle` (Depth-to-Space).
*   **Advantage**: Eliminates all 3x3 convolutions. Extremely low FLOPs. Shifts bottleneck entirely to memory bandwidth (perfect for ZenithLite optimizations).

---

## Benchmarks (CPU - AVX2)

**Hardware Target**: Standard CPU (e.g., x86_64 with AVX2).
**Input Resolution**: 512x512x3.

| Model | Latency (ms) | Speedup vs Baseline | Est. GFLOPs | Actual GFLOPs/s |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla TAESD** | ~5422 ms | 1.0x | ~48.6 | ~9.0 |
| **Zenith-TAESD** | ~2051 ms | **2.6x** | ~48.6* | ~23.7 |
| **Zenith-TAESD-Lite** | **~1293 ms** | **4.2x** | **~17.4** | ~13.5 |

*\*Zenith-TAESD FLOPs are lower in practice due to spectral sparsity, but theoretical max is similar due to 3x3 downsamplers. Lite FLOPs are massively reduced.*

### Key Observations
1.  **Massive Speedup**: The `Lite` variant is **4.2x faster** than the baseline. Even the standard `Zenith` variant is **2.6x faster**.
2.  **Efficiency**: `Zenith-TAESD` achieves higher GFLOPs/s (~23.7) because `OptimizedConv2D` (3x3) is more compute-bound and cache-friendly than the memory-bound 1x1 convs in the Lite model.
3.  **Data Flow**: In `ZenithLiteBlock`, the computation is split:
    *   **Compute (Conv/Mix)**: ~76% of time.
    *   **Data Flow (Transpose)**: ~24% of time.
    *   This indicates the **Blocked Transpose** optimization (Tile=32) effectively mitigated the memory bandwidth bottleneck.

## Implementation Details

*   **OptimizedConv2D**: Uses a custom weight layout `[K, K, In, Out]` to enable vectorization over output channels, bypassing the inefficient `[Out, In, K, K]` standard stride.
*   **Persistent Scratchpads**: Intermediate buffers in `ZenithLiteBlock` are allocated once and reused, preventing memory fragmentation and allocation overhead during inference.
*   **Separable Gating**: Spectral mixing is performed independently on Rows and Columns, reducing complexity from $O(N^2)$ to $O(N)$ (linear).

## Layer Specifications

### 1. Zenith-TAESD (Standard)

| Stage | Layer Type | Input Res | Input Channels | Output Channels | Kernel | Stride |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Encoder** | | | | | | |
| Stem | Conv2D | 512x512 | 3 | 64 | 3x3 | 1 |
| Block 1 | ZenithLite | 512x512 | 64 | 64 | - | - |
| Down 1 | Conv2D | 512x512 | 64 | 128 | 3x3 | 2 |
| Block 2 | ZenithLite | 256x256 | 128 | 128 | - | - |
| Down 2 | Conv2D | 256x256 | 128 | 256 | 3x3 | 2 |
| Block 3 | ZenithLite | 128x128 | 256 | 256 | - | - |
| Out | Conv2D | 128x128 | 256 | 4 | 3x3 | 2 |
| **Decoder** | | | | | | |
| In | Conv2D | 64x64 | 4 | 256 | 1x1 | 1 |
| Block 1 | ZenithLite | 64x64 | 256 | 256 | - | - |
| Up 1 | Upscale + Conv | 64x64 | 256 | 128 | 3x3 | 1 |
| Block 2 | ZenithLite | 128x128 | 128 | 128 | - | - |
| Up 2 | Upscale + Conv | 128x128 | 128 | 64 | 3x3 | 1 |
| Block 3 | ZenithLite | 256x256 | 64 | 64 | - | - |
| Out | Upscale + Conv | 256x256 | 64 | 3 | 3x3 | 1 |

### 2. Zenith-TAESD-Lite (1x1 Only)

| Stage | Layer Type | Input Res | Input Channels | Output Channels | Kernel | Stride |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Encoder** | | | | | | |
| Stem | Conv2D | 512x512 | 3 | 64 | 1x1 | 1 |
| Block 1 | ZenithLite | 512x512 | 64 | 64 | - | - |
| Down 1 | Unshuffle + Conv | 512x512 | 64 | 128 | 1x1 | 1 |
| Block 2 | ZenithLite | 256x256 | 128 | 128 | - | - |
| Down 2 | Unshuffle + Conv | 256x256 | 128 | 256 | 1x1 | 1 |
| Block 3 | ZenithLite | 128x128 | 256 | 256 | - | - |
| Out | Unshuffle + Conv | 128x128 | 256 | 4 | 1x1 | 1 |
| **Decoder** | | | | | | |
| In | Conv + Shuffle | 64x64 | 4 | 256 | 1x1 | 1 |
| Block 1 | ZenithLite | 64x64 | 256 | 256 | - | - |
| Up 1 | Conv + Shuffle | 64x64 | 256 | 128 | 1x1 | 1 |
| Block 2 | ZenithLite | 128x128 | 128 | 128 | - | - |
| Up 2 | Conv + Shuffle | 128x128 | 128 | 64 | 1x1 | 1 |
| Block 3 | ZenithLite | 256x256 | 64 | 64 | - | - |
| Out | Conv2D | 512x512 | 64 | 3 | 1x1 | 1 |
