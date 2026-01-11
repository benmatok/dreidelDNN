#ifndef DREIDEL_LAYERS_LAYER_HPP
#define DREIDEL_LAYERS_LAYER_HPP

#include "../core/Tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class Layer {
public:
    virtual ~Layer() = default;

    // Forward pass: Input -> Output
    virtual Tensor<T, B> forward(const Tensor<T, B>& input) = 0;

    // Backward pass: Gradient of Output -> Gradient of Input
    // Also computes gradients for weights if applicable
    virtual Tensor<T, B> backward(const Tensor<T, B>& grad_output) = 0;

    // Get parameters (weights, biases) for optimizer
    virtual std::vector<Tensor<T, B>*> parameters() { return {}; }

    // Get gradients for optimizer
    virtual std::vector<Tensor<T, B>*> gradients() { return {}; }

    // Get curvatures (diagonal hessian approximation) for optimizer (optional)
    virtual std::vector<Tensor<T, B>*> curvatures() { return {}; }

    // Get activations (inputs) for KFAC (optional)
    virtual std::vector<Tensor<T, B>*> activations() { return {}; }

    // Get output gradients for KFAC (optional)
    virtual std::vector<Tensor<T, B>*> grad_outputs() { return {}; }

    // Set training mode (e.g., for Dropout/BatchNorm)
    virtual void set_training(bool training) { (void)training; }

    virtual std::string name() const = 0;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_LAYER_HPP
