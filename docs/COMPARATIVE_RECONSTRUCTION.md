# Comparative Reconstruction Analysis

This document details the reconstruction quality comparison between the **Zenith Hierarchical Autoencoder** (Spectral) and the **Convolutional Baseline Autoencoder** (Spatial).

## 1. Methodology

*   **Dataset:** Procedurally generated 2D wavelet images (32x32, 3 channels).
    *   Kernels: Gabor, Mexican Hat, Gaussian, Difference of Gaussians, Ridge, High-Freq Checkerboard.
*   **Training:**
    *   Optimizer: Adam (LR=1e-3).
    *   Loss: Mean Squared Error (MSE).
    *   Epochs: 10.
*   **Architecture:**
    *   **Zenith:** Hierarchical Spectral Mixing (PixelUnshuffle + ZenithBlock).
    *   **Conv:** Strided Convolution + Residual Blocks.
    *   **Bottleneck:** Both models compress input to H/32 spatial resolution (1x1 bottleneck).

## 2. Quantitative Results

Evaluation metrics on the validation batch:

| Metric | Zenith (Spectral) | Conv Baseline (Spatial) |
| :--- | :--- | :--- |
| **MSE** | 0.03728 | 0.04898 |
| **MAE** | 0.12555 | 0.14556 |

**Observation:** The Zenith architecture achieved lower reconstruction error (both MSE and MAE) compared to the convolutional baseline in the same training duration, indicating faster convergence or better representational capacity for the wavelet-based data.

## 3. Visual Ablation (Reconstruction Quality)

The following images illustrate the reconstruction capabilities of both models.

*   **Input:** `ablation_input.png` - The original procedurally generated wavelet image.
*   **Zenith Reconstruction:** `ablation_zenith.png` - Output from the Spectral Autoencoder.
*   **Conv Reconstruction:** `ablation_conv.png` - Output from the Spatial Autoencoder.

*(Note: Images are generated in the working directory during training execution.)*

### Analysis
The MAE metric provides a robust measure of reconstruction fidelity that is less sensitive to outliers than MSE. The lower MAE for Zenith suggests it preserves the structural details of the wavelets more effectively than the standard convolution operation, likely due to the spectral mixing's ability to capture global patterns efficiently even with aggressive spatial downsampling.
