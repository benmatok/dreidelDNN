#ifndef DREIDEL_LAYERS_SOFTMAX_HPP
#define DREIDEL_LAYERS_SOFTMAX_HPP

#include "Layer.hpp"
#include <cmath>
#include <limits>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class Softmax : public Layer<T, B> {
public:
    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        // Softmax is usually applied along the last dimension (axis 1 for 2D)
        // y_i = exp(x_i) / sum(exp(x_j))

        if (input.shape().size() != 2) {
             throw std::runtime_error("Softmax only supports 2D tensors");
        }

        size_t rows = input.shape()[0];
        size_t cols = input.shape()[1];

        Tensor<T, B> output(input.shape());

        // Loop over rows (samples in batch)
        for (size_t i = 0; i < rows; ++i) {
            // Find max for numerical stability
            T max_val = std::numeric_limits<T>::lowest();
            for (size_t j = 0; j < cols; ++j) {
                if (input.data()[i * cols + j] > max_val) {
                    max_val = input.data()[i * cols + j];
                }
            }

            // Compute exp and sum
            T sum_exp = 0;
            for (size_t j = 0; j < cols; ++j) {
                T val = std::exp(input.data()[i * cols + j] - max_val);
                output.data()[i * cols + j] = val;
                sum_exp += val;
            }

            // Normalize
            for (size_t j = 0; j < cols; ++j) {
                output.data()[i * cols + j] /= sum_exp;
            }
        }

        output_ = output;
        return output;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // The backward pass for Softmax is often combined with CrossEntropyLoss for simplicity: y_pred - y_true.
        // If we treat it as a standalone layer:
        // dL/dx_i = sum_j (dL/dy_j * dy_j/dx_i)
        // dy_j/dx_i = y_i * (delta_ij - y_j)
        // So dL/dx_i = y_i * (dL/dy_i - sum_k(dL/dy_k * y_k))

        size_t rows = grad_output.shape()[0];
        size_t cols = grad_output.shape()[1];

        Tensor<T, B> grad_input(grad_output.shape());

        for (size_t i = 0; i < rows; ++i) {
            // Compute dot product of grad_output[i] and output[i]
            T sum_grad_y = 0;
            for (size_t j = 0; j < cols; ++j) {
                sum_grad_y += grad_output.data()[i * cols + j] * output_.data()[i * cols + j];
            }

            for (size_t j = 0; j < cols; ++j) {
                T y = output_.data()[i * cols + j];
                T g = grad_output.data()[i * cols + j];
                grad_input.data()[i * cols + j] = y * (g - sum_grad_y);
            }
        }

        return grad_input;
    }

    std::string name() const override { return "Softmax"; }

private:
    Tensor<T, B> output_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_SOFTMAX_HPP
