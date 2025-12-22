# Project Ivrit: Architecture Analysis & Design

## 1. Architecture Analysis (`whisper-large-v3`)

The target model is `ivrit-ai/whisper-large-v3`, which matches the configuration of `openai/whisper-large-v3`.

### Key Dimensions
- **Encoder/Decoder Layers**: 32 each.
- **Hidden Size (`d_model`)**: 1280.
- **FFN Dimension**: 5120.
- **Attention Heads**: 20.
- **Vocab Size**: 51866.
- **Input**: 128 Mel bins.
- **Context**: 1500 (Source), 448 (Target).

### Spectral Dimension Mapping
Since the Spectral WHT kernel requires dimensions to be powers of 2 ($N=2^k$), we must pad the native dimensions.

| Dimension Type | Native Size | Spectral Size ($N$) | Padding | Efficiency Ratio |
|----------------|-------------|---------------------|---------|------------------|
| Hidden (`d_model`) | 1280 | **2048** | 768 | ~62.5% |
| FFN | 5120 | **8192** | 3072 | ~62.5% |
| Mel Bins | 128 | **128** | 0 | 100% |

**Note**: The padding overhead is significant (~37%).
*   **Strategy**: Use `pad_last_dim` before Spectral Layers and `slice_last_dim` after.
*   **Alternative**: Investigate if we can project 1280 -> 1024 (lossy) or if we should stick to 2048. For exact recasting, we must use 2048. For training from scratch, 1024 might suffice. Given "Recasting" is the goal, we assume 2048.

## 2. Layer Replacement Strategy

We aim to replace $O(N^2)$ Dense layers with $O(N)$ `DeepSpectralLinear` layers.

### Encoder Block (Repeated 32x)
1.  **Self-Attention (`MultiHeadAttentionSpectral`)**:
    -   **Q, K, V Projections**: 3x `Dense(1280, 1280)` $\rightarrow$ 3x `DeepSpectralLinear(2048, 2048)`.
        -   *Input*: Pad 1280 -> 2048.
        -   *Output*: Slice 2048 -> 1280 (to split into heads).
    -   **Output Projection**: `Dense(1280, 1280)` $\rightarrow$ `DeepSpectralLinear(2048, 2048)`.
2.  **Feed Forward (`MLP`)**:
    -   **fc1**: `Dense(1280, 5120)` $\rightarrow$ `DeepSpectralLinear(2048, 8192)`.
        -   *Note*: The expansion ratio changes from 4x to 4x (2048->8192). This aligns well.
    -   **fc2**: `Dense(5120, 1280)` $\rightarrow$ `DeepSpectralLinear(8192, 2048)`.

### Decoder Block (Repeated 32x)
Identical to Encoder, with addition of **Cross-Attention**:
1.  **Cross-Attention**:
    -   **Q (from Decoder)**: `DeepSpectralLinear`.
    -   **K, V (from Encoder)**: `DeepSpectralLinear`.
    -   **Out**: `DeepSpectralLinear`.

## 3. Implementation Plan (Phase 2 & 3)

### Components needed
-   **`MultiHeadAttentionSpectral`**: Must handle the padding/slicing internally to expose a standard interface.
-   **`SpectralWhisperEncoder`**: Composed of Conv1d frontend + 32 Spectral Blocks.
-   **`SpectralWhisperDecoder`**: Composed of Embedding + 32 Spectral Blocks.

### Activation Flows
-   **GELU**: Standard.
-   **LayerNorm**: Applied before blocks (Pre-Norm).

### Audio Frontend
-   **Log-Mel Spectrogram**: Need a C++ implementation of Librosa's melspectrogram logic or reuse `torchaudio` equivalent logic in C++.
-   **Conv1d**: Two layers of 1D convolution with stride.
    -   Layer 1: 80 -> ? (Whisper specific, usually stride 1 or 2).
    -   Layer 2: ? -> 1280.
    -   *Action*: Can implement as `Conv1d` or recast as matrix multiplication if sequence length is fixed (but it's not). Need a generic `Conv1d` layer in `dreidelDNN`.

## 4. Dependencies
-   **Python**: `transformers`, `torchaudio` (Verified).
-   **C++**: `dreidel/layers/DeepSpectralLinear.hpp` (Existing), `dreidel/layers/Embedding.hpp` (Need check), `dreidel/layers/Conv1d.hpp` (Need check).
