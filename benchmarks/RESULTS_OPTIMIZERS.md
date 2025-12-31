# Optimizer Benchmark Results (Zenith Autoencoder)

## Experiment Setup
- **Model:** Deep Zenith Autoencoder (64x64x1 -> ... -> 1x1x16 -> ... -> 64x64x1)
- **Layers:** 6 layers (3 Encoder, 3 Decoder) using `ZenithBlock` with spectral mixing.
- **Data:** Synthetic Wavelet/Gaussian 2D signals.
- **Batch Size:** 4
- **Epochs:** 500
- **Hardware:** CPU (OpenMP)

## Results

| Optimizer | Learning Rate | Final MSE Loss | Time (s) |
|-----------|---------------|----------------|----------|
| SGD       | 0.1           | 0.05337        | 63.67    |
| RMSProp   | 0.001         | 0.02978        | 62.80    |
| Adam      | 0.001         | 0.02812        | 59.13    |

## Conclusion
**Adam** achieves the best performance for training the deep Zenith Autoencoder over 500 epochs.

- **Loss:** Adam reached the lowest final loss (0.0281). RMSProp also converged well (0.0298) but was slightly less stable and slower to start descending than Adam. SGD failed to improve from the initial loss (0.0534), indicating it cannot effectively navigate the optimization landscape of this deep spectral model without momentum/adaptive rates.
- **Speed:** Adam was the fastest (59.1s) in this benchmark run.
- **Recommendation:** **Adam** is the clear choice for training Zenith-based models.
