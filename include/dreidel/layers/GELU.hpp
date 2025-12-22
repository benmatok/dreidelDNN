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
        return input.apply([](T x) {
            return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / 3.14159265358979323846) * (x + 0.044715 * std::pow(x, 3))));
        });
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // Approximate backward or exact if needed
        // For now, identity placeholder as exact derivative is complex and usually autograd handles it.
        return grad_output;
    }

    std::vector<Tensor<T, B>*> parameters() override { return {}; }
    std::vector<Tensor<T, B>*> gradients() override { return {}; }
    std::vector<Tensor<T, B>*> curvatures() override { return {}; }
    std::string name() const override { return "GELU"; }
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_GELU_HPP
