#pragma once

#include "layer.hpp"
#include "../algo/alsh.hpp"

namespace dreidel {
namespace layers {

/**
 * @brief Sparse Dense Layer using ALSH (SLIDE implementation).
 *
 * Instead of computing y = Wx + b for all neurons, it identifies
 * a subset of active neurons and computes only for them.
 */
template <typename T>
class ALSHSparseDense : public Layer<T> {
public:
    ALSHSparseDense(int input_dim, int output_dim, algo::ALSHParams alsh_params)
        : input_dim_(input_dim), output_dim_(output_dim), alsh_engine_(alsh_params) {

        // TODO: Initialize weights and bias
        // weights_.resize({input_dim, output_dim});
        // bias_.resize({output_dim});

        // TODO: Initialize ALSH index with random weights
        // alsh_engine_.build_index(weights_);
    }

    core::Tensor<T> forward(const core::Tensor<T>& input) override {
        // 1. Hash input to find active neurons
        auto active_indices = alsh_engine_.query(input);

        // 2. Compute sparse dot product
        // Only compute W[i] * x for i in active_indices

        // TODO: Return result (likely a sparse representation or a masked dense tensor)
        return core::Tensor<T>();
    }

    core::Tensor<T> backward(const core::Tensor<T>& grad_output) override {
        // Update only active weights
        // TODO: Implement sparse backprop
        return core::Tensor<T>();
    }

private:
    int input_dim_;
    int output_dim_;
    core::Tensor<T> weights_;
    core::Tensor<T> bias_;
    algo::ALSH<T> alsh_engine_;
};

} // namespace layers
} // namespace dreidel
