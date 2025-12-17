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
        return input + bias_;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        grad_bias_.fill(0);

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

        return grad_output;
    }

    std::vector<Tensor<T, B>*> parameters() override {
        return {&bias_};
    }

    std::vector<Tensor<T, B>*> gradients() override {
        return {&grad_bias_};
    }

    std::vector<Tensor<T, B>*> curvatures() override {
        return {nullptr};
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
