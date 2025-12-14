# dreidelDNN

**dreidelDNN** is a high-performance, header-only C++ deep learning framework designed for scalable CPU training. It emphasizes algorithmic efficiency over hardware acceleration by leveraging **Asymmetric Locality Sensitive Hashing (ALSH)** to achieve sub-linear training times, inspired by works like [SLIDE](https://arxiv.org/abs/1903.03129) and [Mongoose](https://arxiv.org/abs/2006.07064).

## Key Features

1.  **Header-Only C++**: Easy integration, no complex build systems.
2.  **Scalable CPU Training**: Optimized for multi-node CPU clusters.
3.  **Sublinear Deep Learning**: Implements **SLIDE** mechanisms to sparsely activate neurons using LSH, reducing computational cost from $O(N)$ to $O(k)$ where $k \ll N$.
4.  **KFAC Optimization**: Built-in support for Kronecker-Factored Approximate Curvature for faster convergence.
5.  **x86 Optimization**: Designed for SIMD (AVX2/AVX-512) vectorization.

## Architecture

The framework is organized into the following components within `include/dreidel/`:

### 1. Core (`core/`)
*   **Tensor**: N-dimensional array supporting float32/float64. Will include SIMD intrinsics for fast linear algebra.
*   **Allocator**: Custom memory allocators to ensure cache alignment (critical for CPU performance).

### 2. Algorithms (`algo/`)
*   **ALSH Engine**: Implements the Maximum Inner Product Search (MIPS) using hashing.
    *   **Transformation**: Converts MIPS to Nearest Neighbor Search (NNS) (e.g., $P(x) = [x, 0]$, $Q(y) = [y, \sqrt{M^2 - ||y||^2}]$).
    *   **Hash Families**: Signed Random Projections (SRP) and Learnable LSH (Mongoose).
    *   **Reservoir Sampling**: For selecting active neurons.

### 3. Layers (`layers/`)
*   **Base Layer**: Standard forward/backward interfaces.
*   **Dense**: Standard fully connected layer.
*   **ALSHSparseDense**: The core innovation. It does not perform full matrix multiplication. instead:
    1.  Hashes the input activation.
    2.  Queries the ALSH Engine to find the top-k neurons.
    3.  Computes outputs/gradients *only* for those active neurons.

### 4. Optimizers (`optim/`)
*   **SGD / Adam**: Standard first-order optimizers.
*   **KFAC**: Second-order optimizer approximating the Fisher Information Matrix using Kronecker factorization.

### 5. Distributed (`dist/`)
*   **Communicator**: Abstract interface for data parallel training (Ring AllReduce, Parameter Server).
*   **Backend**: Reference implementation (can be hooked to MPI or TCP).

## Usage (Planned)

```cpp
#include <dreidel/dreidel.hpp>

using namespace dreidel;

int main() {
    // 1. Define Model
    Sequential model;

    // Add standard layers
    model.add<layers::Dense>(784, 1024, activation::ReLU);

    // Add ALSH Sparse Layer (sublinear)
    // 1024 inputs, 10000 outputs, LSH parameters
    model.add<layers::ALSHSparseDense>(1024, 10000, activation::ReLU, ALSHParams{/*...*/});

    model.add<layers::Dense>(10000, 10, activation::Softmax);

    // 2. Optimizer with KFAC
    optim::KFAC optimizer(0.001);

    // 3. Train
    model.compile(optimizer, loss::CrossEntropy);
    model.fit(train_data, train_labels, epochs=10, batch_size=32);

    return 0;
}
```

## Implementation Roadmap

This roadmap breaks down the development into verifiable, testable steps.

### Phase 1: Foundation (Architecture & Mocks)
- [x] Define directory structure.
- [x] Create abstract interfaces for `Tensor`, `Layer`, `Optimizer`, and `Communicator`.
- [x] **Verification**: Compile a "Hello World" that includes the headers and instantiates a mock model.

### Phase 2: Core Tensor & Basic Math
- [x] Implement `Tensor<T>` class with basic storage.
- [x] Implement naive Matrix Multiplication (GEMM) and Element-wise ops.
- [x] Add Basic SIMD support (autovectorization hints).
- [x] **Verification**: Unit tests for Tensor operations (Add, Mul, Dot).

### Phase 3: Basic DNN Flow
- [ ] Implement `Dense` layer (forward/backward).
- [ ] Implement `ReLU` and `Softmax`.
- [ ] Implement `SGD` optimizer.
- [ ] Implement `Sequential` model runner.
- [ ] **Verification**: Train a small network on XOR or MNIST (subset) using standard Dense layers.

### Phase 4: ALSH Engine (The "Brain")
- [ ] Implement Signed Random Projections (SRP) hashing.
- [ ] Implement Hash Tables (array of buckets).
- [ ] Implement MIPS transformation logic.
- [ ] **Verification**: Test retrieval accuracy. Given a query vector, does it retrieve vectors with high dot products? Compare against exact brute force.

### Phase 5: Sparse Training (SLIDE)
- [ ] Implement `ALSHSparseDense` layer.
    -   Connects `ALSH` engine to weight matrix.
    -   Forward: Hash input -> Query -> Sparse Dot.
    -   Backward: Sparse Gradient update.
- [ ] **Verification**: Replace a Dense layer with ALSHSparseDense in the MNIST test. Verify accuracy is comparable but operations are fewer (counting FLOPs).

### Phase 6: KFAC Optimization
- [ ] Implement storage for activation covariance ($A$) and gradient covariance ($G$).
- [ ] Implement block-diagonal inversion logic.
- [ ] Integrate into `Optimizer` step.
- [ ] **Verification**: Check convergence speed (epochs to reach X% accuracy) vs SGD.

### Phase 7: Scalability & Distribution
- [ ] Implement `Communicator` interface for gradient averaging.
- [ ] Mock MPI backend.
- [ ] **Verification**: Simulate 2 nodes in a single process (two models exchanging gradients).

### Phase 8: Mongoose (Adaptive LSH)
- [ ] Implement scheduler for re-hashing or learning hash functions.
- [ ] **Verification**: Long-running training test. Ensure accuracy doesn't degrade as weights shift.
