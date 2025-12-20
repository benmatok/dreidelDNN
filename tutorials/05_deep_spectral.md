# Deep Spectral Networks

While `LinearWHT` is efficient, a single diagonal scaling in the spectral domain corresponds to a circulant convolution, which cannot represent all linear transformations. To approximate arbitrary dense matrices, we stack multiple spectral blocks, creating a **Deep Spectral Linear** layer.

## Architecture

A `DeepSpectralLinear` layer consists of a cascade of $K$ blocks. Each block performs:
$$ x_{i+1} = \text{FWHT}( \text{BlockMix}(\text{Permute}(x_i)) \odot D_i ) $$

- **Permutation**: Randomly shuffles elements to mix information globally.
- **BlockMix**: A local $2 \times 2$ rotation (learnable or fixed) to mix adjacent elements.
- **Scale ($D$)**: Learnable diagonal weights.
- **FWHT**: Global mixing.

This structure allows the network to learn complex dependencies with $O(K \cdot N \log N)$ complexity instead of $O(N^2)$.

## Usage

```cpp
#include <dreidel/layers/DeepSpectralLinear.hpp>

using namespace dreidel;
using namespace dreidel::layers;

// Define parameters
size_t dim = 1024;
size_t num_blocks = 4; // Cascade depth

// Create the layer
auto deep_spectral = std::make_shared<DeepSpectralLinear<float>>(dim, num_blocks);
```

## Recasting Strategy

It is difficult to train Deep Spectral layers from scratch due to the complex optimization landscape. The standard strategy is **Recasting**:
1. Take a pre-trained Dense layer (e.g., from a PyTorch ViT).
2. Initialize a `DeepSpectralLinear` layer.
3. Use **Distillation** to train the spectral layer to mimic the dense layer's output (minimize L2 distance).

## Configuration

You can control the internal structure:
- **Permutations**: Loaded deterministically to match saved models.
- **BlockMix**: Can be disabled for simpler architectures.

```cpp
// Example: loading permutations
std::vector<int> perm = ...;
deep_spectral->set_permutation(block_index, perm);
```

This layer is the workhorse of the Spectral ViT implementation.
