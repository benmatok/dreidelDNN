# Custom Layers

Extending dreidelDNN with custom layers is straightforward. You simply inherit from the `Layer` base class.

## The `Layer` Interface

```cpp
template <typename T, BackendType B = BackendType::CPU>
class Layer {
public:
    virtual Tensor<T, B> forward(const Tensor<T, B>& input) = 0;
    virtual Tensor<T, B> backward(const Tensor<T, B>& grad_output) = 0;

    // Optional: Return parameters for optimization
    virtual std::vector<Tensor<T, B>*> parameters() { return {}; }
    virtual std::vector<Tensor<T, B>*> gradients() { return {}; }

    virtual std::string name() const = 0;
};
```

## Example: A "Square" Layer

Let's implement a layer that computes $y = x^2$.

```cpp
#include <dreidel/layers/Layer.hpp>

template <typename T>
class SquareLayer : public layers::Layer<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) override {
        // Cache input for backward pass
        input_ = input;

        // y = x * x
        return input * input;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // dL/dx = dL/dy * dy/dx
        // dy/dx = 2x

        Tensor<T> two_x = input_ * 2.0f;

        // Element-wise multiplication
        return grad_output * two_x;
    }

    std::string name() const override { return "SquareLayer"; }

private:
    Tensor<T> input_;
};
```

## Integrating

You can now use `SquareLayer` in a `Sequential` model just like any built-in layer.

```cpp
auto sq = std::make_shared<SquareLayer<float>>();
model.add(sq);
```

This flexibility allows you to implement custom activations, normalization schemes, or novel spectral transformations.
