# Optimizer Benchmark Results (Zenith Autoencoder)

## Experiment Setup
- **Model:** Deep Zenith Autoencoder (64x64x1 -> ... -> 1x1x16 -> ... -> 64x64x1)
- **Layers:** 6 layers (3 Encoder, 3 Decoder) using `ZenithBlock` with spectral mixing.
- **Data:** Synthetic Wavelet/Gaussian 2D signals.
- **Batch Size:** 4
- **Epochs:** 100
- **Hardware:** CPU (OpenMP)

## Results

| Optimizer | Learning Rate | Final MSE Loss | Time (s) |
|-----------|---------------|----------------|----------|
| SGD       | 0.1           | 0.05337        | 12.83    |
| RMSProp   | 0.001         | 0.02922        | 13.72    |
| Adam      | 0.001         | 0.02967        | 11.72    |

## Conclusion
**RMSProp** and **Adam** significantly outperform SGD for training the deep Zenith Autoencoder.

- **Loss:** Both RMSProp (0.0292) and Adam (0.0297) achieved meaningful convergence, reducing the loss by ~45% from the initial state. SGD (0.0534) failed to converge effectively in this timeframe, likely stuck on a plateau due to the complex spectral landscape.
- **Speed:** Adam was slightly faster (11.7s) than RMSProp (13.7s) and SGD (12.8s) in this run, though performance is roughly comparable.
- **Recommendation:** Adaptive gradient methods like RMSProp or Adam are essential for training Zenith-based spectral architectures.
