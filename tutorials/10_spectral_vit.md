# Vision Transformers (SpectralViT)

`SpectralViT` is a complete model architecture implemented in dreidelDNN that replaces the standard Multi-Head Attention and MLP blocks with Spectral layers.

## Structure

A Vision Transformer consists of:
1. **Patch Embedding**: Converts image patches to vectors.
2. **Encoder Blocks**: A sequence of Attention and MLP layers.
3. **Head**: Classification layer.

In `SpectralViT`, we replace the quadratic complexity of Attention ($N^2$) and the dense MLPs with $O(N \log N)$ spectral operations.

## Running Inference

Use the `examples/test_spectral_vit.cpp` as a reference.

```cpp
#include <dreidel/models/SpectralViT.hpp>

// 1. Initialize Model
// (embed_dim, hidden_dim, num_heads, num_layers)
auto vit = std::make_shared<models::SpectralViT<float>>(768, 1024, 12, 12);

// 2. Load Weights
// Load weights distilled from a PyTorch model
vit->load_weights("path/to/weights/");

// 3. Forward Pass
Tensor<float> image({1, 3, 224, 224});
// ... fill image ...

Tensor<float> logits = vit->forward(image);
```

## Dimension Mismatches

Standard ViT uses embedding dimension 768. `LinearWHT` requires powers of 2 (e.g., 512 or 1024).
`SpectralViT` handles this by:
- **Padding**: 768 -> 1024 before spectral layers.
- **Slicing**: 1024 -> 768 after spectral layers.

This is handled automatically inside the `SpectralBlock` implementation.
