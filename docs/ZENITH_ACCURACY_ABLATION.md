# Zenith Accuracy Ablation Study

This document details the "Greedy Search" ablation study performed to quantify the impact of architectural improvements on the **Zenith Hierarchical Autoencoder**.

## Methodology

*   **Task**: Autoencoder reconstruction on synthetic 64x64 Wavelet data.
*   **Training**: 1500 steps, Batch Size 4, Adam Optimizer (LR=0.0005).
*   **Baseline**: `ZenithHierarchicalAE` with standard He Initialization and **No** Positional Embeddings.
*   **Strategy**: Greedy addition of improvements.

## Experiments

### 1. Initialization (He vs Identity)
First, we compared the baseline (He Init) against the "Identity Initialization" strategy often recommended for spectral networks.

| Step | Baseline (He) | Identity Init |
| :--- | :--- | :--- |
| 0 | 2.69 | 2.25 |
| 500 | 0.44 | 0.30 |
| 1000 | 0.32 | 0.58 |

*   **Observation**: Identity Initialization provides a significant **early convergence boost** (Loss 0.30 vs 0.44 at Step 500). However, it exhibits instability in later stages (Loss increases to 0.58), suggesting that while gradients flow better, the optimizer might be overshooting.
*   **Conclusion**: Identity Init accelerates initial learning but may require Learning Rate Decay or lower base rates to maintain stability in the long run.

### 2. Positional Embeddings (+ PE)
Next, we added **Fixed 2D Sinusoidal Positional Embeddings** to the Identity-initialized model.

| Step | Identity Init | Identity + PE |
| :--- | :--- | :--- |
| 0 | 2.25 | 3.77 |
| 500 | 0.30 | 1.17 |
| 1000 | 0.58 | 0.61 |

*   **Observation**: Adding PE significantly increases the initial loss (3.77), posing a harder optimization problem initially. By step 1000, it approaches the structure-agnostic model (0.61 vs 0.58) but has not yet surpassed it in this short training regime.
*   **Conclusion**: Positional Embeddings add complexity that requires longer training to pay off. For short training runs or simple wavelet tasks, the structural bias might not yield immediate accuracy gains compared to the raw capacity of the baseline.

## Visual Analysis

![Ablation Graph](ablation_benchmark.svg)

The graph highlights the trade-off:
1.  **Red (Baseline)**: Consistent, stable descent.
2.  **Yellow (Identity)**: Rapid initial drop, followed by volatility.
3.  **Blue (Identity + PE)**: Slow start due to increased complexity.
