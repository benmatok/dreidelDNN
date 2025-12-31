#ifndef DREIDEL_OPTIM_ADAM_HPP
#define DREIDEL_OPTIM_ADAM_HPP

#include "Optimizer.hpp"
#include <vector>
#include <cmath>
#include <iostream>

namespace dreidel {
namespace optim {

template <typename T, BackendType B = BackendType::CPU>
class Adam : public Optimizer<T, B> {
public:
    Adam(T learning_rate, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

    void add_parameters(std::vector<Tensor<T, B>*> params, std::vector<Tensor<T, B>*> grads) {
        if (params.size() != grads.size()) {
             throw std::invalid_argument("Params and grads size mismatch");
        }
        for(auto* p : params) parameters_.push_back(p);
        for(auto* g : grads) gradients_.push_back(g);

        // Initialize moments
        for(auto* p : params) {
            m_.push_back(Tensor<T, B>(p->shape()));
            m_.back().fill(0);
            v_.push_back(Tensor<T, B>(p->shape()));
            v_.back().fill(0);
        }
    }

    void step() override {
        t_++;
        T beta1_t = std::pow(beta1_, t_);
        T beta2_t = std::pow(beta2_, t_);

        for (size_t i = 0; i < parameters_.size(); ++i) {
            Tensor<T, B>* param = parameters_[i];
            Tensor<T, B>* grad = gradients_[i];
            Tensor<T, B>& m_tensor = m_[i];
            Tensor<T, B>& v_tensor = v_[i];

            T* p_ptr = param->data();
            T* g_ptr = grad->data();
            T* m_ptr = m_tensor.data();
            T* v_ptr = v_tensor.data();
            size_t size = param->size();

            DREIDEL_SIMD_LOOP
            for (size_t j = 0; j < size; ++j) {
                T g = g_ptr[j];

                // m = beta1 * m + (1 - beta1) * g
                m_ptr[j] = beta1_ * m_ptr[j] + (1.0 - beta1_) * g;

                // v = beta2 * v + (1 - beta2) * g^2
                v_ptr[j] = beta2_ * v_ptr[j] + (1.0 - beta2_) * g * g;

                T m_hat = m_ptr[j] / (1.0 - beta1_t);
                T v_hat = v_ptr[j] / (1.0 - beta2_t);

                // p = p - lr * m_hat / (sqrt(v_hat) + eps)
                p_ptr[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
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
    T beta1_;
    T beta2_;
    T epsilon_;
    size_t t_;
    std::vector<Tensor<T, B>*> parameters_;
    std::vector<Tensor<T, B>*> gradients_;
    std::vector<Tensor<T, B>> m_;
    std::vector<Tensor<T, B>> v_;
};

} // namespace optim
} // namespace dreidel

#endif // DREIDEL_OPTIM_ADAM_HPP
