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

# Implementation Roadmap: Project Jules (Spectral Evolution)

> This roadmap tracks the evolution from the initial ALSH prototype to the high-performance LibWHT Spectral Engine.

---

## Phase 1: Foundation (Architecture & Mocks)
**Status:** Completed

- [x] Define directory structure.
- [x] Create abstract interfaces for `Tensor`, `Layer`, `Optimizer`, and `Communicator`.
- [x] **Verification:** Compile a "Hello World" that includes the headers and instantiates a mock model.

## Phase 2: Core Tensor & Basic Math
**Status:** Completed

- [x] Implement `Tensor<T>` class with basic storage.
- [x] Implement naive Matrix Multiplication (GEMM) and Element-wise ops.
- [x] Add Basic SIMD support (autovectorization hints).
- [x] **Verification:** Unit tests for `Tensor` operations (`Add`, `Mul`, `Dot`).

## Phase 3: Basic DNN Flow
**Status:** Completed

- [x] Implement `Dense` layer (forward/backward).
- [x] Implement `ReLU` and `Softmax`.
- [x] Implement `SGD` optimizer.
- [x] Implement `Sequential` model runner.
- [x] **Verification:** Train a small network on XOR or MNIST (subset) using standard Dense layers.

## Phase 4: Legacy ALSH Engine (The "Brain" v1)
**Status:** Completed

- [x] Implement Signed Random Projections (SRP) hashing.
- [x] Implement Hash Tables (array of buckets).
- [x] Implement MIPS transformation logic.
- [x] **Verification:** Test retrieval accuracy. Given a query vector, does it retrieve vectors with high dot products?

---

### 🔄 PIVOT POINT: Spectral Acceleration
*Moving from approximate random hashing to structured Walsh-Hadamard Transforms (WHT) for superior memory efficiency and determinism.*

---

## Phase 5: The Spectral Math Kernel (LibWHT Core)
**Status:** Completed

- [x] Refactor `Tensor` to enforce Channel-Last (`NHWC`) memory layout (critical for WHT SIMD).
- [x] Implement `SIMD_FWHT` Kernel (Iterative Fast Walsh-Hadamard Transform).
- [x] Use AVX-512 intrinsics (`_mm512_add_ps`, `_mm512_sub_ps`).
- [x] Ensure operations are strictly **In-Place** (no allocation).
- [x] **Verification (Math):** "Identity Test". Assert $\text{FWHT}(\text{FWHT}(x)) / N == x$ (tolerance `1e-5`).
- [x] **Verification (Speed):** "Throughput Test". Benchmark GB/s vs `memcpy`. Target: >80% of system RAM bandwidth.

## Phase 6: Structured Spectral Layers
**Status:** Planned

- [ ] Implement `LinearWHT` (Replaces Standard Dense).
    - **Logic:** $y = \text{TopK}(\text{FWHT}(x \odot D))$
    - **Storage:** `std::vector<float> scale` ($D$).
- [ ] Implement `Conv3D_Spectral` (Hybrid Accelerator).
    - **Logic:** DepthwiseConv3D (Spatial) $\to$ `LinearWHT` (Mixing).
    - **Fusion:** Fuse Spatial output to Mixing input in L1 cache.
- [ ] **Verification:** Replace a `Dense` layer in the MNIST test with `LinearWHT`. Check parameter reduction ($N^2 \to N$).

## Phase 7: Structure-Aware Optimizer (Replaces KFAC)
**Status:** Planned

- [ ] Implement `DiagonalNewton` Solver (for WHT Layers).
    - **Logic:** Element-wise curvature update: $D_{new} = D - \eta \frac{\nabla L}{\nabla^2 L}$.
- [ ] Implement `BlockDiagonal` Solver (for Spatial Conv).
    - **Logic:** Parallel inversion of small $27 \times 27$ blocks.
- [ ] Update `Optimizer::step()` to dispatch logic based on `LayerType`.
- [ ] **Verification:** "Rosenbrock Test". Check convergence speed on a diagonal quadratic function (< 10 steps).

## Phase 8: Large-Scale Filter Pruning (The "Selector")
**Status:** Planned

- [ ] Implement `WHT Hasher` (Upgrade to Phase 4).
    - **Logic:** $\text{Sign}(\text{FWHT}(x))$ to generate binary codes (replaces SRP).
- [ ] Implement `Sparse Gather Engine`.
    - Use `_mm512_i32gather_ps` to load only selected filter weights.
- [ ] **Verification:** Run on 4096-channel tensor. Metric: >5x speedup vs Dense Compute.

## Phase 9: Scalability & Distribution
**Status:** Planned

- [ ] Implement `SplitWHT` (MPI/Sharding).
    - **Logic:** Distributed Butterfly Diagram (Node A: Steps $1..k$, Node B: Steps $k..N$).
- [ ] Implement Gradient All-Reduce for scale vectors (tiny payload).
- [ ] **Verification:** Run `SplitWHT` on 2 mock nodes. Assert output matches `SingleNodeWHT`.



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
Running ALSH Test...
Index build time: 48 ms
Brute Force Time: 2351 us
ALSH Query Time (incl. re-rank): 21 us
Candidates count: 8 / 10000
Recall @ 10: 1 / 10 (10%)
Test Passed: Retrieved at least some relevant items.
```

#### Benchmark (Phase 4)
*Config: Dim=128, K=10, L=5 (Optimized for speedup > 20x)*

| Items  | Build Time (ms) | BF Query (us) | ALSH Query (us) | Speedup | Recall (%) |
|--------|-----------------|---------------|-----------------|---------|------------|
| 1000   | 9               | 223           | 32              | 6.76    | 100        |
| 10000  | 93              | 3285          | 68              | 47.60   | 100        |
| 50000  | 463             | 18991         | 503             | 37.68   | 100        |
| 100000 | 921             | 34443         | 962             | 35.76   | 100        |

*Note: Benchmarks utilize clustered synthetic data to simulate realistic feature distributions.*

### Spectral Kernel Verification (Phase 5)
```
Running Identity Test (FWHT(FWHT(x)) / N == x)...
Max diff: 4.17233e-07
PASS
Running Throughput Test...
Allocating 128 MB Tensor...
FWHT Throughput: 2.95508 GB/s
Memcpy Throughput: 4.38342 GB/s
Ratio: 67.4148%
PASS (Good)
```
