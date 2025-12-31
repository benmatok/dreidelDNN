# Optimizer Benchmark Results

## Experiment Setup
- **Model:** Autoencoder (Input 64 -> Dense(32) -> Tanh -> Dense(16) -> Tanh -> Dense(32) -> Tanh -> Dense(64))
- **Data:** Synthetic Gaussian/Gabor-like signals
- **Batch Size:** 32
- **Epochs:** 1000
- **Hardware:** CPU (OpenMP)

## Results

| Optimizer | Learning Rate | Final MSE Loss | Time (s) |
|-----------|---------------|----------------|----------|
| SGD       | 0.01          | 0.0980055      | 0.303    |
| RMSProp   | 0.001         | 0.0001572      | 0.321    |
| Adam      | 0.001         | 0.0003766      | 0.365    |

## Conclusion
**RMSProp** is the best performing optimizer for this specific Autoencoder task.

- **Loss:** RMSProp achieved the lowest final loss (0.000157), slightly outperforming Adam (0.000377). Both significantly outperformed SGD (0.098), which failed to converge to a low error state.
- **Speed:** RMSProp is slightly faster than Adam (~12% faster in this run) as it tracks fewer moments (only squared average vs. mean and variance).
- **Recommendation:** For this class of small-scale dense autoencoders on CPU, RMSProp offers the best balance of convergence speed and computational efficiency.
