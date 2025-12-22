#ifndef DREIDEL_LAYERS_SOFTMAX_HPP
#define DREIDEL_LAYERS_SOFTMAX_HPP

#include "Layer.hpp"
#include <cmath>
#include <limits>
#include <numeric>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class Softmax : public Layer<T, B> {
public:
    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        // Softmax is applied along the last dimension.
        // We treat the tensor as a collection of 1D vectors along the last dim.

        size_t total_elements = input.size();
        size_t last_dim = input.shape().back();
        size_t num_vectors = total_elements / last_dim;

        Tensor<T, B> output(input.shape());
        const T* in_ptr = input.data();
        T* out_ptr = output.data();

        #pragma omp parallel for
        for (long i = 0; i < (long)num_vectors; ++i) {
            size_t offset = i * last_dim;

            // Find max for numerical stability
            T max_val = std::numeric_limits<T>::lowest();
            for (size_t j = 0; j < last_dim; ++j) {
                if (in_ptr[offset + j] > max_val) {
                    max_val = in_ptr[offset + j];
                }
            }

            // Compute exp and sum
            T sum_exp = 0;
            for (size_t j = 0; j < last_dim; ++j) {
                T val = std::exp(in_ptr[offset + j] - max_val);
                out_ptr[offset + j] = val;
                sum_exp += val;
            }

            // Normalize
            T inv_sum = 1.0 / sum_exp;
            for (size_t j = 0; j < last_dim; ++j) {
                out_ptr[offset + j] *= inv_sum;
            }
        }

        output_ = output;
        return output;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        size_t total_elements = grad_output.size();
        size_t last_dim = grad_output.shape().back();
        size_t num_vectors = total_elements / last_dim;

        Tensor<T, B> grad_input(grad_output.shape());
        const T* g_out_ptr = grad_output.data();
        const T* y_ptr = output_.data();
        T* g_in_ptr = grad_input.data();

        #pragma omp parallel for
        for (long i = 0; i < (long)num_vectors; ++i) {
            size_t offset = i * last_dim;

            // Compute dot product of grad_output[i] and output[i]
            T sum_grad_y = 0;
            for (size_t j = 0; j < last_dim; ++j) {
                sum_grad_y += g_out_ptr[offset + j] * y_ptr[offset + j];
            }

            for (size_t j = 0; j < last_dim; ++j) {
                T y = y_ptr[offset + j];
                T g = g_out_ptr[offset + j];
                g_in_ptr[offset + j] = y * (g - sum_grad_y);
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
