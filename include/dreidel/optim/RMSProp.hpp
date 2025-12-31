#ifndef DREIDEL_OPTIM_RMSPROP_HPP
#define DREIDEL_OPTIM_RMSPROP_HPP

#include "Optimizer.hpp"
#include <vector>
#include <cmath>
#include <iostream>

namespace dreidel {
namespace optim {

template <typename T, BackendType B = BackendType::CPU>
class RMSProp : public Optimizer<T, B> {
public:
    RMSProp(T learning_rate, T alpha = 0.99, T epsilon = 1e-8)
        : learning_rate_(learning_rate), alpha_(alpha), epsilon_(epsilon) {}

    void add_parameters(std::vector<Tensor<T, B>*> params, std::vector<Tensor<T, B>*> grads) {
        if (params.size() != grads.size()) {
             throw std::invalid_argument("Params and grads size mismatch");
        }
        for(auto* p : params) parameters_.push_back(p);
        for(auto* g : grads) gradients_.push_back(g);

        // Initialize square average accumulators
        for(auto* p : params) {
            square_avg_.push_back(Tensor<T, B>(p->shape()));
            square_avg_.back().fill(0);
        }
    }

    void step() override {
        for (size_t i = 0; i < parameters_.size(); ++i) {
            Tensor<T, B>* param = parameters_[i];
            Tensor<T, B>* grad = gradients_[i];
            Tensor<T, B>& sq_avg = square_avg_[i];

            T* p_ptr = param->data();
            T* g_ptr = grad->data();
            T* s_ptr = sq_avg.data();
            size_t size = param->size();

            DREIDEL_SIMD_LOOP
            for (size_t j = 0; j < size; ++j) {
                T g = g_ptr[j];
                // s = alpha * s + (1 - alpha) * g^2
                s_ptr[j] = alpha_ * s_ptr[j] + (1.0 - alpha_) * g * g;

                // p = p - lr * g / (sqrt(s) + eps)
                p_ptr[j] -= learning_rate_ * g / (std::sqrt(s_ptr[j]) + epsilon_);
            }
        }
    }

    void zero_grad() override {
        for (auto* grad : gradients_) {
            grad->fill(0);
        }
    }

private:
    T learning_rate_;
    T alpha_;
    T epsilon_;
    std::vector<Tensor<T, B>*> parameters_;
    std::vector<Tensor<T, B>*> gradients_;
    std::vector<Tensor<T, B>> square_avg_;
};

} // namespace optim
} // namespace dreidel

#endif // DREIDEL_OPTIM_RMSPROP_HPP
