# Training Zenith Nano AE

This guide explains how to train the Zenith Nano AE model.

## Basics

For training, we use the `ZenithNano` class (not `ZenithNanoInfer`). This class uses standard `float` (Float32) for both storage and computation to ensure numerical stability during backpropagation.

### Code Example

```cpp
#include <dreidel/dreidel.hpp>
#include <dreidel/models/ZenithNano.hpp>
#include <dreidel/optim/Adam.hpp>

using namespace dreidel;

int main() {
    // 1. Instantiate the Training Model
    models::ZenithNano model;
    model.set_training(true); // Enable training mode (gradient buffers)

    // 2. Setup Optimizer
    optim::Adam optimizer(model.parameters(), 1e-3);

    // 3. Training Loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        // ... Load batch ...
        Tensor<float> input = ...;
        Tensor<float> target = input; // Autoencoder reconstruction

        // Forward Pass
        auto output = model.forward(input);

        // Compute Loss (e.g., MSE)
        float loss = ...;

        // Backward Pass
        model.zero_grad();
        // ... propagate gradients ...
        model.backward(grad_output);

        // Update Weights
        optimizer.step();

        std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
    }

    return 0;
}
```

## Running the Example

An example training script is provided in `examples/train_zenith_nano.cpp`. It trains the model on synthetic wavelet data.

To compile and run:

```bash
g++ -O3 -mavx2 -mfma -fopenmp examples/train_zenith_nano.cpp -o train_zenith
./train_zenith
```

This example demonstrates convergence to a low error (MAE ~0.014) in a short amount of time.
