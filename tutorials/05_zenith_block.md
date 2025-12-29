# Tutorial 05: The Zenith Block

This tutorial introduces the **Zenith Block**, the high-performance core of dreidelDNN that uses "Alien Speed" technology (APoT, Spectral Mixing) to run efficiently on CPUs.

## 1. What is the Zenith Block?

The `ZenithBlock` is a drop-in replacement for standard Convolutional blocks (Conv2D + BatchNorm + Activation). It is strictly quantized:
*   **Input**: `Tensor<int8_t>` (APoT encoded)
*   **Weights**: APoT encoded (Powers of Two)
*   **Output**: `Tensor<int8_t>` (APoT encoded)

Because it requires APoT inputs, you must sandwich it between `PackAPoT` and `UnpackAPoT` layers if you are interfacing with standard floating-point data.

## 2. Basic Usage

Here is a minimal example of constructing a model with Zenith Blocks.

```cpp
#include <dreidel/dreidel.hpp>
#include <iostream>

using namespace dreidel;

int main() {
    // 1. Create a Sequential Model
    Sequential model;

    // 2. Input Adapter: Float -> APoT (int8)
    // The ZenithBlock expects int8 inputs.
    model.add<layers::PackAPoT>();

    // 3. Add Zenith Blocks
    // Parameters:
    // - Channels: 64 (Must be Power of 2 for Spectral Mixing)
    // - Kernel Size: 3 (3x3 Convolution)
    // - Spectral Dim: 64 (Usually same as channels)
    model.add<layers::ZenithBlock>(
        64,  // Channels
        3,   // Kernel Size
        64   // Spectral Dimension
    );

    // Stack another block (Deep Network)
    model.add<layers::ZenithBlock>(64, 3, 64);

    // 4. Output Adapter: APoT (int8) -> Float
    model.add<layers::UnpackAPoT>();

    // 5. Run Forward Pass
    Tensor<float> input({1, 32, 32, 64}); // B, H, W, C
    input.randomize_normal();

    Tensor<float> output = model.forward(input);

    std::cout << "Input Shape: " << input.shape() << std::endl;
    std::cout << "Output Shape: " << output.shape() << std::endl;

    return 0;
}
```

## 3. Configuration Options

### Gating (The Oracle)
The Zenith Block supports an optional "Oracle" gating mechanism to skip computation for inactive spatial patches.

```cpp
// Enable Gating (Last parameter = true)
model.add<layers::ZenithBlock>(64, 3, 64, 1024*1024, true);
```

### Memory Arena
The block uses a scratchpad arena for intermediate calculations to avoid malloc overhead. You can configure the size (default 1MB).

```cpp
size_t arena_size = 2 * 1024 * 1024; // 2MB
model.add<layers::ZenithBlock>(64, 3, 64, arena_size);
```

## 4. Performance Tips

1.  **Powers of Two**: Always use channel counts that are powers of two (e.g., 64, 128, 256). The In-Register FWHT relies on this structure.
2.  **Batching**: While `ZenithBlock` supports batching, the lowest latency is achieved with `Batch=1` for real-time applications.
3.  **Compilation**: Ensure you compile with AVX2 or AVX-512 flags to enable the In-Register optimizations.
    ```bash
    g++ -O3 -mavx2 ...
    ```

## 5. Advanced: Manual Pipeline

If you don't want to use `Sequential`, you can manage the tensors manually:

```cpp
layers::PackAPoT pack;
layers::ZenithBlock block(128, 3, 128);
layers::UnpackAPoT unpack;

Tensor<int8_t> packed = pack.forward(float_input);
Tensor<int8_t> processed = block.forward(packed);
Tensor<float> result = unpack.forward(processed);
```

This gives you full control over memory management and layout.
