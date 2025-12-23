#ifndef DREIDEL_LAYERS_GELU_HPP
#define DREIDEL_LAYERS_GELU_HPP

#include "Layer.hpp"
#include <cmath>
#include <vector>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class GELU : public Layer<T, B> {
public:
    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        cached_input_ = input;
        return input.apply([](T x) {
            return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) * (x + 0.044715 * std::pow(x, 3))));
        });
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // d/dx (0.5x(1+tanh(y))) where y = sqrt(2/pi)(x + 0.044715x^3)
        // This is complex.
        // Approximation: sigmoid(1.702*x) * (1 + exp(-1.702*x) + 1.702*x*exp(-1.702*x)) / (1+exp(-1.702*x))^2 ??
        // Standard approx:
        // CDF(x) + x * PDF(x)
        // using the tanh approx:
        // let k = sqrt(2/pi)
        // y = k * (x + 0.044715 * x^3)
        // tanh_y = tanh(y)
        // out = 0.5 * x * (1 + tanh_y)
        // d_out/d_x = 0.5 * (1 + tanh_y) + 0.5 * x * (1 - tanh_y^2) * dy/dx
        // dy/dx = k * (1 + 3 * 0.044715 * x^2)

        if (cached_input_.size() != grad_output.size()) {
            // If shapes mismatch (e.g. not cached), return identity as fallback or zeros
             return grad_output;
        }

        const T k = std::sqrt(2.0 / 3.14159265358979323846);
        const T c = 0.044715;

        Tensor<T, B> grad_input(grad_output.shape());
        const T* x_ptr = cached_input_.data();
        const T* g_ptr = grad_output.data();
        T* out_ptr = grad_input.data();
        size_t size = grad_input.size();

        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            T x = x_ptr[i];
            T g = g_ptr[i];
            T x3 = x * x * x;
            T y = k * (x + c * x3);
            T tanh_y = std::tanh(y);
            T dy_dx = k * (1.0 + 3.0 * c * x * x);
            T d_gelu = 0.5 * (1.0 + tanh_y) + 0.5 * x * (1.0 - tanh_y * tanh_y) * dy_dx;
            out_ptr[i] = g * d_gelu;
        }
        return grad_input;
    }

    std::vector<Tensor<T, B>*> parameters() override { return {}; }
    std::vector<Tensor<T, B>*> gradients() override { return {}; }
    std::vector<Tensor<T, B>*> curvatures() override { return {}; }
    std::string name() const override { return "GELU"; }

private:
    Tensor<T, B> cached_input_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_GELU_HPP
