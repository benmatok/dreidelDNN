#ifndef DREIDEL_OPTIM_DIAGONAL_NEWTON_HPP
#define DREIDEL_OPTIM_DIAGONAL_NEWTON_HPP

#include "Optimizer.hpp"
#include <vector>
#include <cmath>
#include <iostream>

namespace dreidel {
namespace optim {

template <typename T, BackendType B = BackendType::CPU>
class DiagonalNewton : public Optimizer<T, B> {
public:
    DiagonalNewton(T learning_rate, T epsilon = 1e-8)
        : learning_rate_(learning_rate), epsilon_(epsilon) {}

    void add_parameters(std::vector<Tensor<T, B>*> params,
                        std::vector<Tensor<T, B>*> grads,
                        std::vector<Tensor<T, B>*> curvs) {
        if (params.size() != grads.size()) {
             throw std::invalid_argument("Params and grads size mismatch");
        }
        // Curvatures might be empty if layer doesn't support it or if we treat empty as Identity
        // But here we assume we are using it for DiagonalNewton compliant layers.

        for(auto* p : params) parameters_.push_back(p);
        for(auto* g : grads) gradients_.push_back(g);

        if (curvs.size() == params.size()) {
            for(auto* c : curvs) curvatures_.push_back(c);
        } else if (curvs.empty()) {
            // Assume no curvature info provided, treat as 1 (SGD-like but we need to handle index mismatch)
            // We will store nullptrs
            for(size_t i=0; i<params.size(); ++i) curvatures_.push_back(nullptr);
        } else {
             throw std::invalid_argument("Curvatures size mismatch with params");
        }
    }

    void step() override {
        for (size_t i = 0; i < parameters_.size(); ++i) {
            Tensor<T, B>* param = parameters_[i];
            Tensor<T, B>* grad = gradients_[i];
            Tensor<T, B>* curv = curvatures_[i];

            // p = p - lr * grad / (hess + epsilon)
            // hess approx = curv

            // We need to iterate elements.
            // Assuming flat layout is fine for element-wise ops.

            T* p_ptr = param->data();
            T* g_ptr = grad->data();

            size_t size = param->size();
            T lr = learning_rate_;
            T eps = epsilon_;

            if (curv) {
                T* c_ptr = curv->data();
                if (curv->size() != size) throw std::runtime_error("Curvature size mismatch");

                DREIDEL_SIMD_LOOP
                for (size_t j = 0; j < size; ++j) {
                    // Update rule: D_new = D - eta * g / (h + eps)
                    // Note: For Rosenbrock test or general Newton, curvature could be negative?
                    // Usually in NN we use abs(curvature) or dampening to ensure positive definite.
                    // Or we trust it's positive (like x^2 sum).

                    T h = c_ptr[j];
                    // Simple dampening
                    p_ptr[j] -= lr * g_ptr[j] / (std::abs(h) + eps);
                }
            } else {
                // Fallback to SGD if no curvature
                DREIDEL_SIMD_LOOP
                for (size_t j = 0; j < size; ++j) {
                    p_ptr[j] -= lr * g_ptr[j];
                }
            }
        }
    }

    void zero_grad() override {
        for (auto* grad : gradients_) {
            grad->fill(0);
        }
        // Should we zero curvatures? Yes, they are accumulated.
        for (auto* curv : curvatures_) {
            if (curv) curv->fill(0);
        }
    }

private:
    T learning_rate_;
    T epsilon_;
    std::vector<Tensor<T, B>*> parameters_;
    std::vector<Tensor<T, B>*> gradients_;
    std::vector<Tensor<T, B>*> curvatures_;
};

} // namespace optim
} // namespace dreidel

#endif // DREIDEL_OPTIM_DIAGONAL_NEWTON_HPP
