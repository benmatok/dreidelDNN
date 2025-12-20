# LinearWHT: The Spectral Layer

The `LinearWHT` layer is a key component of dreidelDNN. It replaces standard dense matrix multiplications with a spectral transformation, significantly reducing parameter count and computational complexity.

## Concept

A standard Dense layer with input size $N$ and output size $N$ has $N^2$ parameters.
A **LinearWHT** layer has only $N$ parameters.

It implements the operation:
$$ y = \text{FWHT}(x \odot D) $$

Where:
- $x$ is the input vector.
- $D$ is a learnable diagonal scale vector ($N$ parameters).
- $\text{FWHT}$ is the Fast Walsh-Hadamard Transform (parameter-free).

Optionally, it can apply a **TopK** sparsity mask to output only the most significant frequencies.

## Usage

```cpp
#include <dreidel/layers/LinearWHT.hpp>

using namespace dreidel;
using namespace dreidel::layers;

// Create a LinearWHT layer
// Dimension must be a power of 2 (e.g., 512, 1024)
size_t dim = 1024;
auto spectral_layer = std::make_shared<LinearWHT<float>>(dim);

// Optional: TopK sparsity (keep only top 100 coeffs)
auto sparse_layer = std::make_shared<LinearWHT<float>>(dim, 100);
```

## Input Requirements

- The last dimension of the input tensor must match `dim`.
- If the input dimension is smaller, `LinearWHT` will automatically pad it with zeros (during forward) but slicing back is often manual if needed.
- Ideally, design your network such that dimensions are powers of 2.

## Parameter Efficiency

For $N=1024$:
- **Dense**: $1024 \times 1024 = 1,048,576$ parameters.
- **LinearWHT**: $1024$ parameters.
- **Reduction**: ~1000x.

## Performance

- **Dense**: $O(N^2)$ compute.
- **LinearWHT**: $O(N \log N)$ compute.

This makes `LinearWHT` extremely efficient for large layers, though it is less expressive than a full dense matrix. To recover expressivity, we use `DeepSpectralLinear`.
