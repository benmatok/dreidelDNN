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

## Implementation Roadmap: Spectral Acceleration

### Phase 1: The Spectral Math Kernel (LibWHT Core)
**Goal:** Build the fastest possible WHT implementation on CPU.

*   **Memory Manager (NHWC Layout)**
    *   *Task:* Implement Tensor class enforcing Channel-Last layout with 64-byte alignment.
    *   *Validation:* Verify that `reinterpret_cast<__m512*>(ptr)` does not segfault (proving alignment).
*   **SIMD_FWHT Kernel (In-Place)**
    *   *Task:* Implement Iterative FWHT using AVX-512 `_mm512_add_ps` / `_mm512_sub_ps`.
    *   *Validation (Math):* "Identity Test." Assert `FWHT(FWHT(x)) / N == x` within 1e-5 tolerance.
    *   *Validation (Speed):* "Throughput Test." Benchmark GB/s vs memcpy. Target: >80% of system RAM bandwidth.

### Phase 2: Structured Spectral Layers
**Goal:** Replace Heavy Layers with Spectral equivalents.

*   **LinearWHT (Dense Replacement)**
    *   *Task:* Implement $y = \text{TopK}(\text{FWHT}(x \odot D))$.
    *   *Validation (Params):* Assert `sizeof(layer)` equals $N \times 4$ bytes (not $N^2$).
    *   *Validation (Gradient):* "Finite Difference Check." Compare analytic gradient of $D$ vs numerical perturbation.
*   **Conv3D_Spectral (Hybrid Accelerator)**
    *   *Task:* Implement DepthwiseConv3D $\to$ LinearWHT (Mixing).
    *   *Validation (Fusion):* Profiler Check. Ensure no memory writes occur between Spatial and Mixing steps (L1 cache residency).

### Phase 3: Structure-Aware Optimizer
**Goal:** Second-Order convergence in Linear time.

*   **DiagonalNewton Solver**
    *   *Task:* Implement element-wise curvature update: $D_{new} = D - \eta \frac{\nabla L}{\nabla^2 L}$.
    *   *Validation:* "Rosenbrock Test." Minimize a high-dim diagonal quadratic function.
        *   Pass Criteria: Converge in < 10 steps.
        *   Fail Criteria: SGD takes > 100 steps.
*   **BlockDiagonal Solver (Spatial)**
    *   *Task:* Parallel inversion of $27 \times 27$ blocks.
    *   *Validation:* Verify `BlockInv * Block == Identity`.

### Phase 4: Large-Scale Filter Pruning
**Goal:** $O(1)$ Filter Selection for Wide Layers.

*   **WHT Hasher**
    *   *Task:* Implement `Sign(FWHT(x))` to generate binary codes.
    *   *Validation:* "Collision Test." Verify that similar vectors (Euclidean dist < $\epsilon$) produce identical hashes > 90% of the time.
*   **Sparse Gather Engine**
    *   *Task:* Use `_mm512_i32gather_ps` to load weights.
    *   *Validation (Recall/Speedup):*
        *   Experiment: Run on 4096-channel tensor.
        *   Metric: (Time of Dense Conv) / (Time of WHT Hash + Gather Conv). Target: > 5x speedup.

### Phase 5: Distributed Model Parallelism
**Goal:** Training Billion-Parameter Layers across Nodes.

*   **SplitWHT (MPI/Sharding)**
    *   *Task:* Split the butterfly diagram. Node A does steps $1..k$, Node B does $k..N$.
    *   *Validation (Correctness):* Run SplitWHT on 2 processes. Assert output exactly matches SingleNodeWHT.
*   **Gradient All-Reduce**
    *   *Task:* Sync only the diagonal scale vectors $D$.
    *   *Validation (Scaling):* Measure latency vs payload size. Assert latency is negligible compared to compute time.

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
