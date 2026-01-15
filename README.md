# dreidelDNN: Zenith Nano AE

**dreidelDNN** is a high-performance, header-only C++ deep learning framework. It currently hosts **Zenith Nano AE**, an ultra-fast CPU autoencoder designed for real-time inference on edge devices.

## 🚀 Zenith Nano AE

**Zenith Nano AE** is a lightweight autoencoder that achieves state-of-the-art inference speeds on standard CPUs.

*   **Ultra-Fast Inference**: < 7ms for 512x512 images on AVX2 hardware.
*   **Header-Only**: Zero dependencies, just `#include <dreidel/dreidel.hpp>`.
*   **Training & Inference**: Supports full backpropagation and pipelined inference.
*   **Architecture**: Built with `ZenithNanoBlock`, utilizing 1x1 Convolutions and FWHT for spectral mixing.

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/yourusername/dreidelDNN.git
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$(pwd)/dreidelDNN/include
```

### Inference Example

See `examples/ZenithNano_Pipelined.cpp` for a complete example of high-performance pipelined inference.

```cpp
#include <dreidel/dreidel.hpp>
#include <dreidel/models/ZenithNanoInfer.hpp>

int main() {
    using namespace dreidel;

    // Load Model (Weights can be loaded from disk)
    models::ZenithNanoInfer model;

    // Create Input Tensor (Batch=1, H=512, W=512, C=3)
    Tensor<uint16_t> input({1, 512, 512, 3});

    // ... Fill input ...

    // Inference
    auto output = model.forward(input);

    return 0;
}
```

## 📚 Documentation & Tutorials

*   **[Introduction to Zenith Nano](tutorials/zenith_nano/01_introduction.md)**
*   **[Inference Guide](tutorials/zenith_nano/02_inference.md)**
*   **[Training Guide](tutorials/zenith_nano/03_training.md)**
*   **[Architecture Details](docs/ZENITH_NANO_ARCHITECTURE.md)**

## 📂 Repository Structure

*   `include/dreidel/`: Core library headers.
*   `examples/`: Example codes for training and inference.
*   `tutorials/`: Step-by-step guides.
*   `docs/`: Technical documentation.

## 🤝 Contributing

Contributions are welcome!

**License**: MIT
