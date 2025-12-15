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
- [x] Implement `Dense` layer (forward/backward).
- [x] Implement `ReLU` and `Softmax`.
- [x] Implement `SGD` optimizer.
- [x] Implement `Sequential` model runner.
- [x] **Verification**: Train a small network on XOR or MNIST (subset) using standard Dense layers.

### Phase 4: ALSH Engine (The "Brain")
- [x] Implement Signed Random Projections (SRP) hashing.
- [x] Implement Hash Tables (array of buckets).
- [x] Implement MIPS transformation logic.
- [x] **Verification**: Test retrieval accuracy. Given a query vector, does it retrieve vectors with high dot products? Compare against exact brute force.

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

## Test Outputs

### Tensor Tests
```
Testing Tensor Creation...
PASS
Testing Tensor Addition...
PASS
Testing Tensor Matmul (GEMM)...
PASS
All Tensor tests passed!
```

### XOR Verification (Phase 3)
```
Training XOR Network...
Epoch 0, Loss: 0.716317
Epoch 1000, Loss: 0.0152441
Epoch 2000, Loss: 0.00573388
Epoch 3000, Loss: 0.00346887
Epoch 4000, Loss: 0.00246702
Epoch 5000, Loss: 0.00190503
Epoch 6000, Loss: 0.00154734
Epoch 7000, Loss: 0.00130031
Epoch 8000, Loss: 0.00111961
Epoch 9000, Loss: 0.000982013
Predictions:
Input [0, 0] -> Prob [0]: 0.997368, [1]: 0.00263221 | Pred: 0 True: 0
Input [0, 1] -> Prob [0]: 0.000322151, [1]: 0.999678 | Pred: 1 True: 1
Input [1, 0] -> Prob [0]: 0.000356529, [1]: 0.999643 | Pred: 1 True: 1
Input [1, 1] -> Prob [0]: 0.999819, [1]: 0.000181153 | Pred: 0 True: 0
Accuracy: 4/4
PASS
```

### ALSH Verification (Phase 4)
```
Running Comprehensive ALSH Validation...
[TEST] Identity Retrieval... PASS
[TEST] Structured Data Recall... (Recall: 8/10) PASS
[TEST] High Noise Robustness... PASS
[TEST] Orthogonal Query... PASS
All validation tests passed.
```

#### Benchmark (Phase 4)
*Config: Dim=128, K=10, L=5*

| Items  | Build Time (ms) | BF Query (us) | ALSH Query (us) | Speedup | Recall (%) |
|--------|-----------------|---------------|-----------------|---------|------------|
| 1000   | 9               | 232           | 14              | 15.47   | 20         |
| 10000  | 95              | 3622          | 168             | 21.43   | 70         |
| 50000  | 473             | 19830         | 231             | 85.47   | 40         |
| 100000 | 925             | 36406         | 1138            | 31.96   | 80         |

*Note: Benchmarks utilize noisy clustered data (std=0.5) to test robustness. Speedups > 20x achieved for large datasets.*
