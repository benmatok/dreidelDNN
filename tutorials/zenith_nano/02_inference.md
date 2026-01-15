# Inference with Zenith Nano AE

This guide explains how to perform inference using the Zenith Nano AE model.

## Basics

For inference, we typically use the `ZenithNanoInfer` class. This class uses `uint16_t` (Float16) for storage to minimize memory bandwidth, while performing computations in Float32 using AVX2 F16C intrinsics.

### Code Example

```cpp
#include <dreidel/dreidel.hpp>
#include <dreidel/models/ZenithNanoInfer.hpp>

using namespace dreidel;

int main() {
    // 1. Instantiate the Inference Model
    models::ZenithNanoInfer model;

    // 2. Prepare Input
    // Input is expected to be in NHWC format.
    // For ZenithNanoInfer, inputs are typically converted to F16 (uint16_t).
    Tensor<uint16_t> input({1, 512, 512, 3});

    // ... Populate input data ...

    // 3. Run Inference
    auto output = model.forward(input);

    // output is now a Tensor<uint16_t> containing the reconstructed image.

    return 0;
}
```

## Pipelined Inference

To achieve maximum performance (< 3ms per block), we use a pipelined approach that interleaves operations.

See `examples/ZenithNano_Pipelined.cpp` for a detailed implementation of this technique. The core idea is to run the integer-based Spatial FWHT on different execution ports than the floating-point Channel Mixing operations.

### Compilation

Ensure you compile with the necessary flags:

```bash
g++ -O3 -mavx2 -mfma -mf16c -fopenmp examples/ZenithNano_Pipelined.cpp -o zenith_pipelined
```

*   `-mavx2`: Enables AVX2 instructions.
*   `-mf16c`: Enables hardware conversion between Float16 and Float32.
*   `-mfma`: Enables Fused Multiply-Add.
