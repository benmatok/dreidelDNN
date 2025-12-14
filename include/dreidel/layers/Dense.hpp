#ifndef DREIDEL_LAYERS_DENSE_HPP
#define DREIDEL_LAYERS_DENSE_HPP

#include "Layer.hpp"
#include <random>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class Dense : public Layer<T, B> {
public:
    Dense(size_t input_dim, size_t output_dim)
        : input_dim_(input_dim), output_dim_(output_dim),
          weights_({input_dim, output_dim}), bias_({1, output_dim}),
          grad_weights_({input_dim, output_dim}), grad_bias_({1, output_dim})
    {
        // Xavier/Glorot Initialization (Normal)
        // stddev = sqrt(2 / (in + out))
        T stddev = std::sqrt(2.0 / (input_dim + output_dim));

        weights_.random(0, stddev);
        bias_.fill(0);

        grad_weights_.fill(0);
        grad_bias_.fill(0);
    }

    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        // Y = X * W + b
        // Input: (Batch, InputDim)
        // W: (InputDim, OutputDim)
        // b: (1, OutputDim)
        // Output: (Batch, OutputDim)

        // Store input for backward pass
        input_ = input;

        Tensor<T, B> output = input.matmul(weights_);
        output = output + bias_; // Broadcasting add
        return output;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // grad_output: (Batch, OutputDim) aka dL/dY

        // dL/dW = X^T * dL/dY
        // (InputDim, Batch) * (Batch, OutputDim) = (InputDim, OutputDim)
        grad_weights_ = input_.transpose().matmul(grad_output);

        // dL/db = sum(dL/dY, axis=0)
        // Sum over batch dimension
        grad_bias_ = grad_output.sum(0);

        // dL/dX = dL/dY * W^T
        // (Batch, OutputDim) * (OutputDim, InputDim) = (Batch, InputDim)
        Tensor<T, B> grad_input = grad_output.matmul(weights_.transpose());

        return grad_input;
    }

    std::vector<Tensor<T, B>*> parameters() override {
        return {&weights_, &bias_};
    }

    std::vector<Tensor<T, B>*> gradients() override {
        return {&grad_weights_, &grad_bias_};
    }

    std::string name() const override { return "Dense"; }

private:
    size_t input_dim_;
    size_t output_dim_;

    Tensor<T, B> weights_;
    Tensor<T, B> bias_;

    Tensor<T, B> grad_weights_;
    Tensor<T, B> grad_bias_;

    // Cache for backward
    Tensor<T, B> input_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_DENSE_HPP
