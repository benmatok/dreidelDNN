# Introduction to Zenith Nano AE

Zenith Nano AE is a specialized autoencoder designed for extreme efficiency on CPU architectures. By leveraging the spectral properties of the Walsh-Hadamard Transform (FWHT) and optimized 1x1 convolutions, it achieves inference latencies suitable for real-time applications without requiring a GPU.

## Key Features

*   **Spectral Mixing**: Uses `ZenithNanoBlock` which incorporates FWHT for global information mixing, avoiding expensive large-kernel convolutions.
*   **AVX2/AVX-512 Optimization**: Heavily optimized kernels for x86 processors.
*   **Pipelining**: Supports instruction-level parallelism by interleaving floating-point and integer operations.
*   **Float16 Inference**: Uses `ZenithNanoInfer` for reduced memory bandwidth usage during inference.

## When to use Zenith Nano AE

*   **Edge Computing**: When GPU resources are unavailable or power-constrained.
*   **Real-Time Processing**: When latency is critical (< 10ms target).
*   **Embedded Systems**: Lightweight header-only integration.

## Next Steps

*   [Inference Guide](02_inference.md)
*   [Training Guide](03_training.md)
