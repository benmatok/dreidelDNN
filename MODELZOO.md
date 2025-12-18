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

*   **Patch Embedding:** `Conv2D` projection -> `DeepSpectralLinear`.
*   **Self-Attention:**
    *   Replace dense projections ($Q, K, V$) with `DeepSpectralLinear`.
    *   Investigate replacing Softmax Attention with **Fast Walsh-Hadamard Attention** (linearized attention using WHT features).
*   **MLP Block:**
    *   Replace `Dense` layers with `ALSHSparseDense` for large-scale width scaling (MoE-like behavior) OR `DeepSpectralLinear` for dense efficiency.

### C. Spectral Llama (LLM)
*Target: Text Generation*

*   **Token Embeddings:** Standard lookup.
*   **Layers:**
    *   Replace `Linear` layers in Attention and FeedForward blocks with `DeepSpectralLinear`.
    *   **FFN:** Use `ALSHSparseDense` to activate only relevant neurons for a given token (Conditional Computation), effectively implementing an ultra-fast Mixture-of-Experts (MoE).
*   **Rotary Embeddings (RoPE):** Adapt to spectral domain if applicable, or keep in spatial domain.

---

## 2. Theoretical Analysis & Recasting Strategy

### Analysis of `LinearWHT` Expressivity

The standard `LinearWHT` layer implements the transformation:
$$ y = \text{FWHT}(x \odot D) $$
where $D$ is a learnable diagonal vector. In matrix form, this is $y = x D H^T$ (assuming row vectors).

*   **Parameter Count:** $O(N)$ vs $O(N^2)$ for Dense.
*   **Constraint:** This effectively restricts the weight matrix $W$ to be diagonal in the Hadamard basis. This means $W$ must be a circulant-like convolution.
*   **Limitation:** A single `LinearWHT` layer cannot approximate arbitrary dense matrices (e.g., random projections or specific permutations). This explains the high relative error (~1.0) observed in initial recasting attempts.

### Solution: `DeepSpectralLinear` (Cascaded Spectral Layers)

To recover full expressivity while maintaining $O(N \log N)$ computational complexity (vs $O(N^2)$), we introduce the `DeepSpectralLinear` layer. This layer cascades multiple `LinearWHT` blocks interleaved with fixed random permutations.

$$ y = \text{Layer}_K( \dots \text{Layer}_1(x) \dots ) $$
where $\text{Layer}_k(x) = \text{FWHT}(P_k(x) \odot D_k)$.

*   $P_k$: A fixed random permutation (shuffles the vector).
*   $D_k$: A learnable diagonal scale.
*   **Expressivity:** By stacking $K$ such layers (typically $K=4$ to $\log N$), the network can approximate any unitary matrix and, by extension, any dense matrix (Fastfood transform theorem).
*   **Cost:** $K \times O(N \log N)$. For small $K$, this is still significantly faster than $O(N^2)$ for large $N$.

### Improved Recasting Strategy

Since `DeepSpectralLinear` has a non-convex structure involving multiple layers, simple least-squares recasting is no longer feasible.

1.  **Initialization:**
    *   Initialize $D_k$ to preserve variance (e.g., Identity flow or slightly noisy).
    *   Initialize $P_k$ as random fixed permutations.
2.  **Distillation (Mandatory):**
    *   We cannot analytically solve for weights. We must proceed directly to Stage 2 (Distillation).
    *   Use the "Teacher-Student" loop to train the `DeepSpectralLinear` parameters ($D_1 \dots D_K$) to match the output of the source `nn.Linear` layer using `DiagonalNewton` optimizer.

---

## 3. Implementation Status

### Spectral ViT (ViT-Base-Patch16-224 / DeiT-Tiny)

*   **Status:** Functional Prototype (DeiT-Tiny Validated).
*   **Previous Finding:** Simple `LinearWHT` recasting failed (Error ~1.0).
*   **Current Solution:**
    1.  Implemented `DeepSpectralLinear` (K=4) in `recast_pytorch.py` and C++ model.
    2.  Recasting tool now supports distillation data generation.
    3.  **Validation:** C++ Block-Wise Distillation (`train_spectral_vit.cpp`) converges successfully on DeiT-Tiny (Loss reduced from ~3.3 to ~0.82 on Block 0).
    4.  **Inference:** Verified correctness and speed (~290ms on CPU).
