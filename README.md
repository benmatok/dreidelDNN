# dreidelDNN
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/e7144ace-58d5-4a7a-83ef-9b90a043410b" />

**dreidelDNN** is a high-performance, header-only C++ deep learning framework designed for scalable CPU training. It is built upon the revolutionary **Zenith Technology**, enabling **Alien Speed** without GPUs.

> **Why dreidelDNN?**
> *   👽 **Alien Speed**: Replaces floating-point arithmetic with bitwise logic and table lookups.
> *   🚀 **Zenith Block**: The core building block achieving massive throughput via APoT quantization and In-Register Spectral Mixing.
> *   🛠️ **Zero Dependencies**: Just `#include <dreidel/dreidel.hpp>`.
> *   🧠 **Algorithmic Innovation**: Implements "Deep Learning with Hashes" (SLIDE) and "Spectral Learning" (LibWHT).

---

## 🏆 **SOTA: Zenith-Lasso** (New!)

The **Zenith-Lasso Autoencoder** is our latest breakthrough, offering **43x faster inference** than standard Conv2D models while maintaining stable convergence.

*   **Technology**: Uses Convex Group Lasso Regularization to induce structural sparsity in the spectral domain.
*   **Performance**: ~35ms / image (128x128) vs ~1500ms for Conv2D on CPU.
*   **Architecture**: Fully spectral (ZenithBlocks) with stabilized normalization.

👉 **[See the Benchmark Results](docs/COMPARATIVE_RECONSTRUCTION.md)** | **[Read the Tutorial](tutorials/13_zenith_lasso.md)**

---

## 🚀 Quick Start: The Zenith Way

dreidelDNN is header-only. Just clone and include.

### Installation
```bash
git clone https://github.com/yourusername/dreidelDNN.git
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$(pwd)/dreidelDNN/include
```

### Example: High-Performance Zenith Pipeline
This example demonstrates constructing a model using the `ZenithBlock`, operating entirely in the `int8` APoT domain for maximum efficiency.

```cpp
#include <dreidel/dreidel.hpp>

using namespace dreidel;

int main() {
    // 1. Prepare Data (Standard Float Tensors)
    Tensor<float> input({1, 64, 64, 128}); // Batch, H, W, Channels

    // 2. Define Zenith Model
    // Pipeline: Float -> PackAPoT -> Zenith -> Zenith -> UnpackAPoT -> Float
    Sequential model;

    // Convert Standard Float to Alien APoT (int8)
    model.add<layers::PackAPoT>();

    // Zenith Blocks: "Alien Speed" Processing
    // (Channels=128, Kernel=3, SpectralDim=128)
    model.add<layers::ZenithBlock>(128, 3, 128);
    model.add<layers::ZenithBlock>(128, 3, 128);

    // Convert back to Standard Float
    model.add<layers::UnpackAPoT>();

    // 3. Inference
    auto output = model.forward(input);

    return 0;
}
```

---

## 👽 The Zenith Technology

The core of dreidelDNN is the **Zenith Block**, a hyper-optimized primitive that outperforms standard layers by exploiting the "Alien Speed" master plan.

*   **APoT (Additive Power-of-Two)**: Replaces multiplication with integer addition in the Logarithmic Number System.
*   **In-Register FWHT**: Performs Spectral Mixing entirely within AVX registers, avoiding L1 cache latency.
*   **Soft Permutation**: Local mixing via specialized butterfly operations.
*   **Shuffle AVX LUT**: Uses `vpshufb` as a massive parallel lookup table for function approximation.

👉 **[Read the Full Zenith Technology Guide](docs/ZENITH_TECHNOLOGY.md)**

### "Secret Sauce" Optimizations
The Zenith architecture incorporates several advanced algorithmic tricks to ensure stability and performance in Deep Spectral Networks:

1.  **Delta-Orthogonal Initialization**: Depthwise kernels are initialized as Identity mappings to preserve signal variance across deep networks, preventing vanishing/exploding gradients in the spectral domain.
2.  **Spectral Dropout**: Regularization is applied by dropping random frequencies in the Hadamard domain, forcing the network to learn robust global features.
3.  **GroupNorm Integration**: Replaces standard BatchNorm with Group Normalization (32 groups), ensuring stable convergence even with small micro-batches typical of CPU training.
4.  **Coordinate-Wise Clipping**: The `SimpleAdam` optimizer supports clipping gradients per-element rather than by global norm, preventing high-frequency noise from destabilizing the DC component.

---

## 🗺️ Roadmap & Status

### 🏛️ Core Engine (Completed)
The foundation of the library is stable and verified.
- [x] **Zenith Block**: The primary computational unit.
- [x] **Tensor Core**: NHWC layout, AVX-512 SIMD, Custom Allocators.
- [x] **ALSH Engine**: Signed Random Projections & Mongoose LSH for sublinear retrieval.
- [x] **Spectral Kernel**: High-throughput `FWHT` (Fast Walsh-Hadamard Transform) reaching 80% RAM bandwidth.

### 🔭 Project Jules: Spectral Vision (In Progress)
Recasting Vision Transformers (ViT) to Spectral Architectures.
- [x] **Recast Tool**: Convert PyTorch ViT to Dreidel format (`tools/recast_pytorch.py`).
- [x] **Distillation**: Block-wise training loop (`examples/train_spectral_vit.cpp`).

### 🎙️ Project Ivrit: Spectral Speech (Planned)
**Focus:** Porting `ivrit-ai/whisper-large-v3` to a fully spectral C++ architecture for efficient CPU-based ASR/TTS.

---

## 🏛️ Standard & Legacy Architectures

While `ZenithBlock` is the future, dreidelDNN supports other spectral and standard layers for compatibility and research.

*   **LinearWHT**: The original O(N) spectral layer.
*   **DeepSpectralLinear**: Cascaded WHT layers for high expressivity (Used in SpectralViT).
*   **Standard Dense**: Baseline dense layers (O(N^2)).

### Spectral Whisper Recasting
Guides for recasting Whisper models can be found in the [Legacy Documentation](docs/LEGACY.md) or previous versions of this README.

---

## 📂 Architecture Overview

The framework is organized into the following components within `include/dreidel/`:

| Component | Description |
|-----------|-------------|
| **`layers/`** | **`ZenithBlock`** (Primary), `LinearWHT`, `DeepSpectralLinear`, `ALSHSparseDense`, `GroupNorm`. |
| **`core/`** | `Tensor`, `Allocator` (SIMD-aligned memory management). |
| **`algo/`** | `WHT` (Spectral Transforms), `ALSH` (Hashing Engine). |
| **`optim/`** | `DiagonalNewton` (2nd Order Spectral), `KFAC`. |
| **`models/`** | Pre-assembled architectures (`SpectralViT`, `SpectralWhisper`). |

## 🤝 Contributing

We welcome contributions! Please see `MODELZOO.md` for our current research directions.

**License**: MIT
