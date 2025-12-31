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
| SGD       | 0.1           | 0.05337        | 63.67    |
| RMSProp   | 0.001         | 0.02978        | 62.80    |
| **Adam**  | **0.001**     | **0.02033**    | **66.56**|

**Conclusion:** Adam is the most effective optimizer for the Zenith architecture, achieving significantly lower loss than SGD (which failed to converge) and RMSProp.

## Part 2: Model Comparison (Zenith vs Conv2D)

Both models were trained using **Adam (LR=0.001)** for 500 epochs.

| Model | Architecture | Final MSE Loss | Time (s) | Speedup |
|-------|--------------|----------------|----------|---------|
| **Zenith** | Spectral (WHT) | 0.02033 | **66.6s** | **2.14x** |
| Conv2D | Standard Spatial | **0.01663** | 142.4s | 1.00x |

**Analysis:**
- **Accuracy:** The standard **Conv2D** Autoencoder achieves slightly better reconstruction quality (MSE 0.0166 vs 0.0203), which is expected as full spatial convolution is more expressive than the factorized spectral mixing in ZenithBlock.
- **Speed:** The **Zenith** Autoencoder is **~2.1x faster** to train than the equivalent Conv2D model on CPU.
- **Trade-off:** Zenith offers a strong speed/accuracy trade-off, delivering 2x speedup with only a modest increase in reconstruction error (~20% higher MSE). This validates the "Alien Speed" hypothesis for spectral layers.
