# Legacy Documentation & Guides

This file contains documentation for older architectures and workflows, such as the original Spectral Whisper recasting pipeline. While these are still supported, the project's main focus has shifted to the **Zenith Block**.

---

## 🧪 Spectral Whisper Recasting Guide

### 1. Generate Weights and Data
Use the `tools/recast_whisper.py` script to export a pre-trained Whisper model (e.g., `openai/whisper-tiny`) to Spectral format. This process also generates synthetic distillation data (input/output pairs for each block).

```bash
# Export weights and generate validation data (Batch Size 1 recommended for CPU testing)
python3 tools/recast_whisper.py --model openai/whisper-tiny --batch-size 1
```

### 2. Run Block-wise Distillation
Compile and run the C++ distillation loop. This trains the `DeepSpectralLinear` layers to approximate the original Dense projections.

```bash
g++ -O3 -fopenmp -std=c++17 -Iinclude examples/train_spectral_whisper.cpp -o train_spectral_whisper
./train_spectral_whisper --epochs 5 --lr 0.1
```

### 🔍 Recasting Insights

*   **Learning Rate Sensitivity**: Unlike `SpectralViT` which tolerates high learning rates (`1.0`), `SpectralWhisper` (particularly the Decoder) requires a more conservative rate (`0.1`) to prevent divergence.
*   **Cross-Attention Shape Mismatch**: The backward pass for Multi-Head Attention must handle shape mismatches between Query and Key/Value paths. Our implementation checks shapes and freezes V-projection training if a mismatch (typical in Cross-Attention) is detected, ensuring stability.
*   **GELU Gradients**: Using an exact or high-quality approximation for GELU gradients (cached input) is crucial. Identity approximation leads to training instability in deeper networks.
*   **Block-wise Convergence**: The Encoder blocks converge robustly (~0.7 -> 0.67 MSE). Decoder blocks show faster initial convergence but are more sensitive to initialization variance.

---

## 🧪 Spectral Vision Transformers (ViT)

### Project Jules Overview
Recasting Vision Transformers (ViT) to Spectral Architectures.
- **DeepSpectralLinear**: Cascaded WHT layers for high expressivity.
- **Recast Tool**: Convert PyTorch ViT to Dreidel format (`tools/recast_pytorch.py`).
- **Distillation**: Block-wise training loop (`examples/train_spectral_vit.cpp`).
