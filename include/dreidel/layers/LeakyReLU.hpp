#ifndef DREIDEL_LAYERS_LEAKYRELU_HPP
#define DREIDEL_LAYERS_LEAKYRELU_HPP

#include "Layer.hpp"

namespace dreidel {
namespace layers {

template <typename T>
class LeakyReLU : public Layer<T> {
public:
    LeakyReLU(float negative_slope = 0.2f) : negative_slope_(negative_slope) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        input_ = input;
        return input.apply([this](T val) { return val > 0 ? val : static_cast<T>(val * negative_slope_); });
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> grad_input(grad_output.shape());
        size_t n = grad_output.size();
        const T* grad_data = grad_output.data();
        const T* input_data = input_.data();
        T* res_data = grad_input.data();

        #pragma omp parallel for
        for(size_t i=0; i<n; ++i) {
            res_data[i] = (input_data[i] > 0) ? grad_data[i] : static_cast<T>(grad_data[i] * negative_slope_);
        }

        return grad_input;
    }

    std::string name() const override { return "LeakyReLU"; }

private:
    float negative_slope_;
    Tensor<T> input_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_LEAKYRELU_HPP
