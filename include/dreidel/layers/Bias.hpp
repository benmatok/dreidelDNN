#ifndef DREIDEL_LAYERS_BIAS_HPP
#define DREIDEL_LAYERS_BIAS_HPP

#include "Layer.hpp"
#include <vector>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class Bias : public Layer<T, B> {
public:
    Bias(size_t dim)
        : dim_(dim), bias_({1, dim}), grad_bias_({1, dim})
    {
        bias_.fill(0);
        grad_bias_.fill(0);
    }

    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        // Broadcast add
        // Input: (..., dim)
        // Bias: (1, dim)
        return input + bias_;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // dL/db = sum(dL/dy, axis=0...N-1)
        // grad_output: (..., dim)

        // Sum over all dimensions except last
        // Assuming Tensor.sum(axis) reduces one axis.
        // If Tensor doesn't support reducing multiple axes at once easily, we rely on implementation.
        // Usually HAL ops reduce.

        // Let's implement manual reduction similar to LinearWHT if needed,
        // but Tensor might have `sum(axis)` or `reduce_sum`.
        // The Memory says `ops.hpp` defines `load`, `store`, `add`, `sub`, `butterfly`.
        // It doesn't explicitly mention `sum` or `reduce`.
        // But `Dense.hpp` uses `grad_output.sum(0)`.

        // If grad_output has >2 dimensions, `sum(0)` only reduces batch.
        // We need to reduce all except last.
        // For now, assume Flatten->Sum(0) or similar logic if supported.
        // Actually, if input is (Batch, Dim), sum(0) works.
        // If (Batch, Seq, Dim), sum(0).sum(0)?

        // Let's assume standard (Batch, Dim) usage for now as per Dense.
        // If more dims, we need to iterate.

        // For ViT, input is (Batch, Seq, Dim).
        // So we need to sum over Batch AND Seq.

        // Implementation:
        grad_bias_.fill(0);

        // Manual reduction
        size_t last_dim = dim_;
        size_t total_elements = grad_output.size();
        size_t outer_dims = total_elements / last_dim;

        const T* grad_ptr = grad_output.data();
        T* bias_grad_ptr = grad_bias_.data();

        #pragma omp parallel
        {
            std::vector<T> local_sum(last_dim, 0);

            #pragma omp for
            for(long i=0; i<(long)outer_dims; ++i) {
                size_t offset = i * last_dim;
                for(size_t j=0; j<last_dim; ++j) {
                    local_sum[j] += grad_ptr[offset + j];
                }
            }

            #pragma omp critical
            {
                for(size_t j=0; j<last_dim; ++j) {
                    bias_grad_ptr[j] += local_sum[j];
                }
            }
        }

        // dL/dx = dL/dy (Identity for bias add)
        return grad_output;
    }

    std::vector<Tensor<T, B>*> parameters() override {
        return {&bias_};
    }

    std::vector<Tensor<T, B>*> gradients() override {
        return {&grad_bias_};
    }

    std::string name() const override { return "Bias"; }

private:
    size_t dim_;
    Tensor<T, B> bias_;
    Tensor<T, B> grad_bias_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_BIAS_HPP
