# Optimizer Benchmark Results (Zenith Autoencoder)

## Experiment Setup
- **Task:** Wavelet Image Reconstruction (Autoencoder)
- **Data:** Synthetic 64x64 Wavelet/Gaussian Signals
- **Batch Size:** 4
- **Epochs:** 500
- **Optimizer:** Adam (LR=0.001) for comparisons, varied for optimizer test.
- **Hardware:** CPU (OpenMP)

## Part 1: Optimizer Comparison (Zenith Autoencoder)

| Optimizer | Learning Rate | Final MSE Loss | Time (s) |
|-----------|---------------|----------------|----------|
| SGD       | 0.1           | 0.05337        | 12.90    |
| RMSProp   | 0.001         | 0.05337        | 12.95    |
| **Adam**  | **0.001**     | **0.02117**    | **54.39**|

*Note: Runtime variance expected between runs.*

## Part 2: Model Comparison (Optimized Zenith vs Conv2D)

Both models were trained using **Adam (LR=0.001)** for 500 epochs. The Zenith model uses **implicit optimized upscaling** (fused within the layer), while Conv2D uses explicit `Upscale2D` layers.

| Model | Architecture | Final MSE Loss | Time (s) | Speedup |
|-------|--------------|----------------|----------|---------|
| **Zenith** | Implicit Optimized (WHT) | 0.02117 | **54.4s** | **2.57x** |
| Conv2D | Standard Spatial | **0.01662** | 139.9s | 1.00x |

**Analysis:**
- **Speed:** The **Optimized Zenith** Autoencoder is **~2.57x faster** than the Conv2D baseline (previously 2.1x with explicit scaling). This confirms the efficiency of the implicit/fused design.
- **Accuracy:** Reconstruction quality remains competitive (MSE 0.021 vs 0.017), validating that the Zenith optimization provides substantial speed gains with acceptable accuracy trade-offs.
- **Convergence:** Adam effectively trains the deep spectral network, overcoming the initial plateaus seen with SGD.
