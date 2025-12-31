# Optimizer Benchmark Results (Zenith Autoencoder)

## Experiment Setup
- **Task:** Wavelet Image Reconstruction (Autoencoder)
- **Data:** Synthetic 64x64 Wavelet/Gaussian Signals
- **Batch Size:** 4
- **Epochs:** 500
- **Optimizer:** Adam (LR=0.001) for comparisons, varied for optimizer test.
- **Hardware:** CPU (OpenMP + AVX2)

## Part 1: Optimizer Comparison (Zenith Autoencoder)

| Optimizer | Learning Rate | Final MSE Loss | Time (s) |
|-----------|---------------|----------------|----------|
| SGD       | 0.1           | 0.05337        | 12.90    |
| RMSProp   | 0.001         | 0.05337        | 12.95    |
| **Adam**  | **0.001**     | **0.05337**    | **53.33**|

*Note: Convergence failure persists with AVX2 enabled, likely due to pre-ReLU cache inconsistency or gradient instability in the optimized path.*

## Part 2: Model Comparison (Optimized Zenith vs Conv2D)

Both models were trained using **Adam (LR=0.001)** for 500 epochs. The Zenith model uses **implicit optimized upscaling** (fused within the layer) and AVX2 optimizations.

| Model | Architecture | Final MSE Loss | Time (s) | Speedup |
|-------|--------------|----------------|----------|---------|
| **Zenith** | Implicit Optimized (WHT) | 0.05337 | **53.3s** | **2.99x** |
| Conv2D | Standard Spatial | **0.01663** | 159.6s | 1.00x |

**Analysis:**
- **Speed:** The **Optimized Zenith** Autoencoder achieves a **~3x speedup** over Conv2D for training.
- **Accuracy:** The Zenith model currently fails to converge (MSE 0.053 vs 0.017) when using the AVX2 path, despite the corrected backward logic for SoftPerm. This suggests a subtle bug in how the AVX2 forward pass populates the `pre_relu` cache or how it interacts with the backward pass.
- **Next Steps:** Debugging the AVX2 `pre_relu` storage logic is critical to unlocking both speed and accuracy. The generic path (verified previously) converged to MSE 0.020, proving the architecture is sound.
