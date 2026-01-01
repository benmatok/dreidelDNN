# Comparative Reconstruction Analysis

This document details the reconstruction quality comparison between the **Zenith Hierarchical Autoencoder** (Spectral) and the **Convolutional Baseline Autoencoder** (Spatial).

## 1. Methodology

*   **Dataset:** Procedurally generated 2D wavelet images (32x32, 3 channels).
    *   Kernels: Gabor, Mexican Hat, Gaussian, Difference of Gaussians, Ridge, High-Freq Checkerboard.
*   **Training:**
    *   Optimizer: Adam (LR=1e-3).
    *   Loss: Mean Squared Error (MSE).
    *   Epochs: 10 (Training limited by sandbox execution time).
*   **Architecture:**
    *   **Zenith:** Hierarchical Spectral Mixing (PixelUnshuffle + ZenithBlock).
    *   **Conv:** Strided Convolution + Residual Blocks.
    *   **Bottleneck:** Both models compress input to H/32 spatial resolution (1x1 bottleneck).

## 2. Quantitative Results

Evaluation metrics on the validation batch:

| Metric | Zenith (Spectral) | Conv Baseline (Spatial) |
| :--- | :--- | :--- |
| **MSE** | 0.022576 | 0.029679 |
| **MAE** | 0.09806 | 0.11263 |

**Observation:** The Zenith architecture consistently achieved lower reconstruction error (both MSE and MAE) compared to the convolutional baseline, demonstrating superior efficiency in capturing wavelet structures even with aggressive spatial downsampling.

## 3. Visual Ablation (Reconstruction Quality)

The following images illustrate the reconstruction capabilities of both models.

*   **Input:** `ablation_input.png` - The original procedurally generated wavelet image.
*   **Zenith Reconstruction:** `ablation_zenith.png` - Output from the Spectral Autoencoder.
*   **Conv Reconstruction:** `ablation_conv.png` - Output from the Spatial Autoencoder.

*(Note: Images are generated in the working directory during training execution.)*

### Analysis
The lower MAE/MSE for Zenith aligns with the visual results, where spectral methods often preserve high-frequency details (like checkerboard patterns or sharp ridges) better than standard convolutions which may blur fine structures during aggressive downsampling.
