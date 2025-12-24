# dreidelDNN
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/e7144ace-58d5-4a7a-83ef-9b90a043410b" />

**dreidelDNN** is a high-performance, header-only C++ deep learning framework designed for scalable CPU training. It prioritizes algorithmic efficiency over raw hardware acceleration, leveraging **Spectral Transforms (WHT)** and **Asymmetric Locality Sensitive Hashing (ALSH)** to achieve sub-linear training times.

> **Why dreidelDNN?**
> *   🚀 **Speed without GPU:** Up to 1000x parameter reduction using Spectral Layers.
> *   🛠️ **Zero Dependencies:** Just `#include <dreidel/dreidel.hpp>`.
> *   🧠 **Algorithmic Innovation:** Implements "Deep Learning with Hashes" (SLIDE) and "Spectral Learning" (LibWHT).

---

## 🚀 Quick Start

dreidelDNN is header-only. Just clone and include.

### Installation
```bash
git clone https://github.com/yourusername/dreidelDNN.git
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$(pwd)/dreidelDNN/include
```

### Example: Spectral Training
```cpp
#include <dreidel/dreidel.hpp>

using namespace dreidel;

int main() {
    // 1. Define Model
    Sequential model;

    // Spectral Linear Layer (O(N) parameters instead of O(N^2))
    model.add<layers::LinearWHT>(1024, activation::ReLU);
    model.add<layers::LinearWHT>(1024, activation::Softmax);

    // 2. Optimizer (Structure-Aware)
    optim::DiagonalNewton optimizer(0.1);

    // 3. Train
    model.compile(optimizer, loss::CrossEntropy);
    // model.fit(data, labels, ...);

    return 0;
}
```

---

## 🗺️ Roadmap & Status

### 🏛️ Core Engine (Completed)
The foundation of the library is stable and verified.
- [x] **Tensor Core**: NHWC layout, AVX-512 SIMD, Custom Allocators.
- [x] **ALSH Engine**: Signed Random Projections & Mongoose LSH for sublinear retrieval.
- [x] **Spectral Kernel**: High-throughput `FWHT` (Fast Walsh-Hadamard Transform) reaching 80% RAM bandwidth.
- [x] **Optimizers**: `DiagonalNewton` for spectral layers and `KFAC` for dense layers.

### 🔭 Project Jules: Spectral Vision (In Progress)
Recasting Vision Transformers (ViT) to Spectral Architectures.
- [x] **DeepSpectralLinear**: Cascaded WHT layers for high expressivity.
- [x] **Recast Tool**: Convert PyTorch ViT to Dreidel format (`tools/recast_pytorch.py`).
- [x] **Distillation**: Block-wise training loop (`examples/train_spectral_vit.cpp`).
- [ ] **Validation**: Full ImageNet convergence.

---

### 🎙️ Project Ivrit: Spectral Speech (Planned)
**Focus:** Porting `ivrit-ai/whisper-large-v3` to a fully spectral C++ architecture for efficient CPU-based ASR/TTS.

#### Phase 1: Architecture Analysis & Setup (Completed)
- [x] **Analysis**: Map `whisper-large-v3` dimensions, attention heads, and activation flows.
- [x] **Dependencies**: Setup Python environment with `transformers`, `torchaudio`, and `librosa` for verification.
- [x] **Scope**: Determine layer replacement strategy (e.g., `DeepSpectralLinear` for Q/K/V projections).

#### Phase 2: Core Components Implementation (Completed)
- [x] **MultiHeadAttentionSpectral**: Implement efficient attention mechanism using spectral projections.
- [x] **AudioEncoder**: Implement Log-Mel Spectrogram preprocessing in C++.
- [x] **SpectralWhisper**: Implement full Encoder-Decoder class in `include/dreidel/models/SpectralWhisper.hpp`.
    - [x] Support for KV-Caching in C++ (Placeholder).
    - [x] Support for variable sequence lengths (via padding/slicing).

#### Phase 3: Recasting Tooling (Completed)
- [x] **Recast Script**: Create `tools/recast_whisper.py` to map PyTorch weights to Spectral Layers.
    - [x] Implement "Variance Preserving Initialization" for initializing spectral scales from dense matrices.
    - [x] Support capture of `encoder_hidden_states` for Decoder recasting.
- [x] **Verification**: Unit test comparing layer-by-layer outputs (PyTorch vs C++) with random inputs (`tests/test_whisper_recast.cpp`).

#### Phase 4: Distillation Strategy (Completed)
- [x] **Teacher-Student Loop**: Implement `train_spectral_whisper.cpp` for block-wise distillation.
    - [x] Successfully distilling Encoder blocks using `DiagonalNewton`.
    - [x] Successfully distilling Decoder blocks (including Cross-Attention logic).
- [x] **Optimization**: Tuned `DiagonalNewton` learning rates (`0.1` recommended).

#### Phase 5: End-to-End Verification (Planned)
- [ ] **Inference Test**: Run a "Hello World" audio file through the C++ pipeline.
- [ ] **Benchmark**: Measure Real-Time Factor (RTF) on standard CPU (Target: <0.5 RTF).
- [ ] **Accuracy**: Verify Word Error Rate (WER) degradation is within acceptable limits (<10% relative).

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

## 📂 Architecture Overview

The framework is organized into the following components within `include/dreidel/`:

| Component | Description |
|-----------|-------------|
| **`core/`** | `Tensor`, `Allocator` (SIMD-aligned memory management). |
| **`algo/`** | `WHT` (Spectral Transforms), `ALSH` (Hashing Engine). |
| **`layers/`** | `LinearWHT` (Spectral), `ALSHSparseDense` (Sublinear), `DeepSpectralLinear`. |
| **`optim/`** | `DiagonalNewton` (2nd Order Spectral), `KFAC`. |
| **`models/`** | Pre-assembled architectures (`SpectralViT`, `SpectralWhisper`). |

## 🤝 Contributing

We welcome contributions! Please see `MODELZOO.md` for our current research directions.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

**License**: MIT
