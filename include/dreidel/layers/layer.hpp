#pragma once

#include "../core/tensor.hpp"

namespace dreidel {
namespace layers {

/**
 * @brief Abstract Base Layer.
 */
template <typename T>
class Layer {
public:
    virtual ~Layer() = default;

    /**
     * @brief Forward pass.
     *
     * @param input Input tensor.
     * @return Output tensor.
     */
    virtual core::Tensor<T> forward(const core::Tensor<T>& input) = 0;

    /**
     * @brief Backward pass.
     *
     * @param grad_output Gradient w.r.t output.
     * @return Gradient w.r.t input.
     */
    virtual core::Tensor<T> backward(const core::Tensor<T>& grad_output) = 0;

    /**
     * @brief Get parameters (weights, biases).
     */
    virtual std::vector<core::Tensor<T>*> parameters() { return {}; }
    virtual std::vector<core::Tensor<T>*> gradients() { return {}; }
};

} // namespace layers
} // namespace dreidel
