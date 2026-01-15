# Zenith Nano Architecture

The Zenith Nano AE architecture is built for speed. It deviates from standard CNN designs to maximize CPU throughput.

## Core Components

### 1. ZenithNanoBlock

The fundamental building block. It replaces the standard `Conv2D(3x3)` with a spectral mixing approach.

*   **1x1 Convolution**: Performs channel mixing. This is computationally dense and optimized with `OptimizedConv2D`.
*   **FWHT (Fast Walsh-Hadamard Transform)**: Performs spatial mixing. This is an $O(N \log N)$ operation, significantly faster than $O(N^2)$ spatial convolutions for large receptive fields.
*   **Pipelining**: The block is designed to allow overlapping execution of the integer-heavy FWHT and float-heavy 1x1 Conv.

### 2. SpaceToDepth / DepthToSpace

Instead of using strided convolutions for downsampling or transposed convolutions for upsampling, Zenith Nano uses shuffling.

*   **SpaceToDepth (PixelUnshuffle)**: Rearranges spatial data into channels. Reduces spatial resolution while increasing channel depth, allowing subsequent operations to work on smaller grids.
*   **DepthToSpace (PixelShuffle)**: The inverse operation for upsampling.

### 3. Architecture Layout

A typical Zenith Nano configuration might look like this:

1.  **Input**: Image (H, W, 3)
2.  **SpaceToDepth**: Converts to (H/8, W/8, 3*64)
3.  **Encoder Blocks**: Series of `ZenithNanoBlock`s to extract features.
4.  **Bottleneck**: Compression to latent space.
5.  **Decoder Blocks**: Series of `ZenithNanoBlock`s to reconstruct features.
6.  **DepthToSpace**: Converts back to (H, W, 3)
7.  **Output**: Reconstructed Image

## Implementation Details

*   **ZenithNano (Float32)**: Used for training. Supports gradient computation.
*   **ZenithNanoInfer (Float16)**: Used for inference. Stores weights and activations in `uint16_t` (half-precision) to reduce memory footprint and bandwidth usage, utilizing `AVX2 F16C` for on-the-fly conversion to Float32 arithmetic.

## Performance

*   **Latency**: < 7ms on AVX2 hardware for 512x512 inputs.
*   **Efficiency**: Designed to keep execution units saturated by mixing integer and floating-point instructions.
