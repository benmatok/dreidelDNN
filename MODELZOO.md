# dreidelDNN Model Zoo & Porting Strategy

This document outlines the roadmap for implementing standard deep learning architectures within the `dreidelDNN` framework, leveraging its unique Spectral (`LibWHT`) and ALSH capabilities. It also details the strategy for porting pre-trained models (PyTorch/JAX/ONNX) to our efficient spectral format via distillation and recasting.

## 1. Native Model Implementations

We aim to provide reference implementations for three key architectures, optimized for CPU training/inference using our spectral layers.

### A. Spectral UNet (Medical/Imaging)
*Target: 3D Volumetric Segmentation (BraTS, KiTS)*

*   **Encoder:** Replace standard `Conv3D` with `Conv3DSpectral` (Spatial Depthwise + LinearWHT Mixing).
*   **Skip Connections:** Use simple element-wise addition (or concatenation followed by `LinearWHT` mixing).
*   **Decoder:** Spectral UpSampling (WHT-based interpolation or Transposed Conv3DSpectral).
*   **Optimization:**
    *   Use `BlockDiagonal` solver for the spatial components.
    *   Use `DiagonalNewton` for the mixing layers.

### B. Spectral ViT (Vision Transformer)
*Target: Image Classification (ImageNet)*

*   **Patch Embedding:** `Conv2D` projection -> `LinearWHT`.
*   **Self-Attention:**
    *   Replace dense projections ($Q, K, V$) with `LinearWHT`.
    *   Investigate replacing Softmax Attention with **Fast Walsh-Hadamard Attention** (linearized attention using WHT features).
*   **MLP Block:**
    *   Replace `Dense` layers with `ALSHSparseDense` for large-scale width scaling (MoE-like behavior) OR `LinearWHT` for dense efficiency.

### C. Spectral Llama (LLM)
*Target: Text Generation*

*   **Token Embeddings:** Standard lookup.
*   **Layers:**
    *   Replace `Linear` layers in Attention and FeedForward blocks with `LinearWHT`.
    *   **FFN:** Use `ALSHSparseDense` to activate only relevant neurons for a given token (Conditional Computation), effectively implementing an ultra-fast Mixture-of-Experts (MoE).
*   **Rotary Embeddings (RoPE):** Adapt to spectral domain if applicable, or keep in spatial domain.

---

## 2. Porting Strategy: Distillation & Recasting

Since `dreidelDNN` uses unique layers (`LinearWHT`, `ALSHSparseDense`) that do not map 1:1 to standard matrix multiplication weights, we cannot simply load PyTorch state dicts. We use a two-stage process.

### Stage 1: Recasting (Architecture Mapping)

We define a mapping from Source (Standard) layers to Target (Spectral) layers.

| Source Layer (PyTorch) | Target Layer (dreidelDNN) | initialization Strategy |
| :--- | :--- | :--- |
| `nn.Linear(in, out)` | `LinearWHT(in)` | **Projected Initialization**: Train `LinearWHT` to mimic `nn.Linear` output using least squares on a calibration dataset (activations). |
| `nn.Conv2d(in, out, k)` | `Conv3DSpectral` (adapted) | Decompose 2D conv into Spatial (Depthwise) + Mixing (WHT). Initialize Spatial with approx depthwise filters, Mixing via projection. |
| `nn.ReLU` | `ReLU` | Direct mapping. |

### Stage 2: Distillation (Knowledge Transfer)

Once the student model (dreidelDNN architecture) is initialized via Recasting, we perform **Knowledge Distillation** to recover accuracy.

1.  **Teacher:** Pre-trained PyTorch/HuggingFace model (frozen).
2.  **Student:** `dreidelDNN` model (e.g., Spectral Llama).
3.  **Data:** Unlabeled calibration dataset (subset of pre-training data).
4.  **Loss:**
    *   $L = \alpha L_{logits}(Teacher, Student) + \beta L_{hidden}(Teacher, Student)$
    *   Match logits and potentially intermediate hidden states.
5.  **Optimizer:** Use `DiagonalNewton` (Phase 7) for rapid convergence of the diagonal scale factors $D$ in `LinearWHT`.

### Workflow Tooling (Planned)

*   `tools/recast_pytorch.py`: Python script to load a PyTorch model, probe its activations, solve for initial `LinearWHT` scale factors, and export weights to `dreidelDNN` binary format.
*   `tools/distill_cpp`: C++ binary that loads the exported weights and fine-tunes them against the Teacher model's outputs (served via ONNX Runtime or LibTorch).
