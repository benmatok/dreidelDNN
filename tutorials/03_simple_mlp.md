# Building a Simple MLP

This tutorial guides you through building a simple Multi-Layer Perceptron (MLP) using standard `Dense` layers. While dreidelDNN specializes in Spectral layers, standard layers are fully supported.

## Components

A typical neural network in dreidelDNN consists of:
1. **Layers**: `Dense`, `ReLU`, `Softmax`, etc.
2. **Model container**: `Sequential` (stores layers).
3. **Loss function**: `MSE`, `CrossEntropy` (often implemented manually or via helper functions in tests).
4. **Optimizer**: `SGD`, `DiagonalNewton`.

## Defining the Model

Let's build a network for a simple regression task: Input (10) -> Hidden (32) -> Output (1).

```cpp
#include <dreidel/dreidel.hpp>
#include <vector>
#include <memory>

using namespace dreidel;
using namespace dreidel::layers;

int main() {
    // 1. Define Layers
    // Dense(input_dim, output_dim)
    auto fc1 = std::make_shared<Dense<float>>(10, 32);
    auto relu = std::make_shared<ReLU<float>>();
    auto fc2 = std::make_shared<Dense<float>>(32, 1);

    // 2. Create Sequential Model
    models::Sequential<float> model;
    model.add(fc1);
    model.add(relu);
    model.add(fc2);

    // 3. Create Dummy Data
    // Batch size 5
    Tensor<float> input({5, 10});
    input.random(0.0f, 1.0f);

    Tensor<float> target({5, 1});
    target.random(0.0f, 1.0f);

    // 4. Optimizer
    optim::SGD<float> optimizer(0.01f); // Learning rate 0.01

    // 5. Training Loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Forward
        Tensor<float> output = model.forward(input);

        // Compute Loss (MSE)
        // Loss = 0.5 * (output - target)^2
        Tensor<float> diff = output - target;
        // In a real loop you'd sum this for logging

        // Backward
        // dLoss/dOutput = (output - target)
        model.backward(diff);

        // Update
        // Collect parameters and gradients
        auto params = model.parameters();
        auto grads = model.gradients();

        optimizer.step(params, grads);
    }

    std::cout << "Training complete." << std::endl;
    return 0;
}
```

## Explanation

- **Dense Layer**: Performs $y = xW + b$. It initializes weights automatically using Xavier initialization.
- **ReLU**: Applied element-wise.
- **Sequential**: Manages the forward and backward pass order.
- **Backward**: The gradient of the loss function is passed to `model.backward()`.
- **Optimizer**: `optimizer.step()` updates the weights in-place using the gradients.

This pattern forms the basis for more complex models.
