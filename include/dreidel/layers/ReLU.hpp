#ifndef DREIDEL_LAYERS_RELU_HPP
#define DREIDEL_LAYERS_RELU_HPP

#include "Layer.hpp"

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class ReLU : public Layer<T, B> {
public:
    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        input_ = input;
        return input.apply([](T val) { return val > 0 ? val : 0; });
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // dL/dx = dL/dy * dy/dx
        // dy/dx = 1 if x > 0 else 0

        // Element-wise multiplication
        // Since we don't have direct element-wise multiplication between two tensors in apply easily
        // without more complex Tensor iterators, we can do it manually loop or upgrade Tensor.

        // Let's implement element-wise gradient computation.
        Tensor<T, B> grad_input(grad_output.shape());
        size_t n = grad_output.size();
        const T* grad_data = grad_output.data();
        const T* input_data = input_.data();
        T* res_data = grad_input.data();

        // TODO: Move this to SIMD loop in Tensor class or accessible helper
        for(size_t i=0; i<n; ++i) {
            res_data[i] = (input_data[i] > 0) ? grad_data[i] : 0;
        }

        return grad_input;
    }

    std::string name() const override { return "ReLU"; }

private:
    Tensor<T, B> input_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_RELU_HPP
