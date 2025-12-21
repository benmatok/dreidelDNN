#ifndef DREIDEL_LAYERS_GELU_HPP
#define DREIDEL_LAYERS_GELU_HPP

#include "Layer.hpp"
#include <cmath>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class GELU : public Layer<T, B> {
public:
    GELU() {}

    std::string name() const override { return "GELU"; }

    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        saved_input_ = input;
        Tensor<T, B> output(input.shape());
        const T* in_ptr = input.data();
        T* out_ptr = output.data();
        size_t n = input.size();

        // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float c1 = 0.7978845608; // sqrt(2/pi)
        const float c2 = 0.044715;

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_ptr[i]);
            float cubic = x * x * x;
            float inner = c1 * (x + c2 * cubic);
            float tanh_val = std::tanh(inner);
            out_ptr[i] = static_cast<T>(0.5f * x * (1.0f + tanh_val));
        }

        return output;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // d/dx = 0.5 * (1 + tanh(y)) + 0.5 * x * (1 - tanh^2(y)) * dy/dx
        // where y = sqrt(2/pi) * (x + 0.044715 * x^3)
        // dy/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)

        Tensor<T, B> grad_input(saved_input_.shape());
        const T* in_ptr = saved_input_.data();
        const T* grad_out_ptr = grad_output.data();
        T* grad_in_ptr = grad_input.data();
        size_t n = saved_input_.size();

        const float c1 = 0.7978845608;
        const float c2 = 0.044715;
        const float c3 = 0.134145; // 3 * c2

        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_ptr[i]);
            float x_sq = x * x;
            float cubic = x * x_sq;
            float inner = c1 * (x + c2 * cubic);
            float tanh_val = std::tanh(inner);

            float dy_dx = c1 * (1.0f + c3 * x_sq);
            float secl_sq = 1.0f - tanh_val * tanh_val;

            float d_gelu = 0.5f * (1.0f + tanh_val) + 0.5f * x * secl_sq * dy_dx;

            grad_in_ptr[i] = static_cast<T>(static_cast<float>(grad_out_ptr[i]) * d_gelu);
        }

        return grad_input;
    }

    std::vector<Tensor<T, B>*> parameters() override { return {}; }
    std::vector<Tensor<T, B>*> gradients() override { return {}; }
    std::vector<Tensor<T, B>*> curvatures() override { return {}; }

private:
    Tensor<T, B> saved_input_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_GELU_HPP
