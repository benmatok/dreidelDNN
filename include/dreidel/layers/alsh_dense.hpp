#pragma once

#include "Layer.hpp"
#include "../algo/alsh.hpp"
#include <vector>
#include <iostream>

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
        : input_dim_(input_dim), output_dim_(output_dim),
          alsh_engine_(alsh_params),
          weights_({static_cast<size_t>(input_dim), static_cast<size_t>(output_dim)}),
          bias_({1, static_cast<size_t>(output_dim)}),
          grad_weights_({static_cast<size_t>(input_dim), static_cast<size_t>(output_dim)}),
          grad_bias_({1, static_cast<size_t>(output_dim)})
    {
        // Initialization
        T stddev = std::sqrt(2.0 / (input_dim + output_dim));
        weights_.random(0, stddev);
        bias_.fill(0);

        grad_weights_.fill(0);
        grad_bias_.fill(0);

        // Build ALSH Index
        alsh_engine_.build_index(weights_);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Input: (Batch, InDim)
        input_ = input;

        size_t batch_size = input.shape()[0];
        Tensor<T> output({batch_size, static_cast<size_t>(output_dim_)});
        output.fill(0); // Default to 0 for inactive

        T* out_ptr = output.data();
        const T* in_ptr = input.data();
        const T* w_ptr = weights_.data();
        const T* b_ptr = bias_.data();

        // Cache active indices for backward pass
        active_indices_cache_.resize(batch_size);

        // Sparse Forward
        for(size_t b=0; b<batch_size; ++b) {
            // Extract single sample
            Tensor<T> sample({static_cast<size_t>(input_dim_)});
            T* s_ptr = sample.data();
            for(size_t i=0; i<input_dim_; ++i) s_ptr[i] = in_ptr[b*input_dim_ + i];

            std::vector<int> active_indices = alsh_engine_.query(sample);
            active_indices_cache_[b] = active_indices; // Cache

            // Compute for active indices
            for(int idx : active_indices) {
                if (idx >= output_dim_) continue;

                T val = b_ptr[idx];
                for(size_t k=0; k<input_dim_; ++k) {
                    val += s_ptr[k] * w_ptr[k*output_dim_ + idx]; // Column-major access if weights are (In, Out)
                }
                out_ptr[b*output_dim_ + idx] = val;
            }
        }

        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Sparse Backward using active indices

        // Ensure grad_output is cached for KFAC if needed (though KFAC might need dense)
        // If KFAC is used, it will read this. But if we sparsify, 'grad_outputs()' returns sparse grad?
        // KFAC usually expects full grad for stats.
        // For now, let's just cache it.
        grad_output_ = grad_output;

        // Reset gradients
        grad_weights_.fill(0);
        grad_bias_.fill(0);

        size_t batch_size = input_.shape()[0];
        const T* go_ptr = grad_output.data();
        const T* in_ptr = input_.data();
        const T* w_ptr = weights_.data();

        // dL/dX = dL/dY * W^T
        // Sparse calculation: only active output neurons contribute to dL/dX
        Tensor<T> grad_input({batch_size, static_cast<size_t>(input_dim_)});
        grad_input.fill(0);
        T* gi_ptr = grad_input.data();

        T* gw_ptr = grad_weights_.data();
        T* gb_ptr = grad_bias_.data();

        for(size_t b=0; b<batch_size; ++b) {
            const auto& active_indices = active_indices_cache_[b];

            for(int idx : active_indices) {
                if (idx >= output_dim_) continue;

                T dy = go_ptr[b*output_dim_ + idx]; // Scalar gradient for this neuron

                // Update grad_bias
                gb_ptr[idx] += dy;

                // Update grad_weights and grad_input
                for(size_t k=0; k<input_dim_; ++k) {
                    T x_val = in_ptr[b*input_dim_ + k];

                    // dL/dW[k, idx] += x[b, k] * dy
                    gw_ptr[k*output_dim_ + idx] += x_val * dy;

                    // dL/dX[b, k] += dy * W[k, idx]
                    gi_ptr[b*input_dim_ + k] += dy * w_ptr[k*output_dim_ + idx];
                }
            }
        }

        // Periodically update ALSH index?
        // alsh_engine_.build_index(weights_);

        return grad_input;
    }

    // Explicitly rebuild index
    void rebuild_index() {
         alsh_engine_.build_index(weights_);
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&weights_, &bias_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_weights_, &grad_bias_};
    }

    std::vector<Tensor<T>*> activations() override {
        return {&input_};
    }

    std::vector<Tensor<T>*> grad_outputs() override {
        return {&grad_output_};
    }

    std::string name() const override { return "ALSHSparseDense"; }

private:
    int input_dim_;
    int output_dim_;
    Tensor<T> weights_;
    Tensor<T> bias_;

    Tensor<T> grad_weights_;
    Tensor<T> grad_bias_;

    algo::ALSH<T> alsh_engine_;

    Tensor<T> input_;
    Tensor<T> grad_output_;

    std::vector<std::vector<int>> active_indices_cache_;
};

} // namespace layers
} // namespace dreidel
