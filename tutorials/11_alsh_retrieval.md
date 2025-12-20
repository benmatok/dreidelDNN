# ALSH: Approximate Nearest Neighbors

dreidelDNN includes an Asymmetric Locality Sensitive Hashing (ALSH) engine for fast retrieval, useful for large-scale classification or memory augmentation.

## Concepts

- **MIPS**: Maximum Inner Product Search. Finding vectors $x$ that maximize $q \cdot x$.
- **SRP**: Signed Random Projections. A hashing scheme that preserves cosine similarity.
- **Transformation**: ALSH maps MIPS to Cosine Similarity using a specific norm transformation.

## Usage

```cpp
#include <dreidel/algo/alsh.hpp>

using namespace dreidel::algo;

// 1. Configure ALSH
size_t K = 10; // Number of bits per hash
size_t L = 5;  // Number of hash tables
size_t dim = 128;

ALSH<float> engine(dim, K, L);

// 2. Add Database Items
std::vector<Tensor<float>> items;
// ... load items ...

for (size_t i = 0; i < items.size(); ++i) {
    engine.add(items[i], i); // Store ID 'i'
}

// 3. Query
Tensor<float> query_vec({1, dim});
// ... fill query ...

// Retrieve top candidates (approximate)
std::vector<size_t> candidates = engine.query(query_vec);

// 4. Re-rank
// Compute exact distance only on candidates to find the true top-1
```

## Performance

- **Speed**: Sublinear time complexity $O(L \cdot N^{1/c})$.
- **Recall**: Dependent on $K$ and $L$. Higher $L$ improves recall, higher $K$ improves precision (speeds up by reducing candidates).
