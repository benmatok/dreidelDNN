#ifndef DREIDEL_OPTIM_SIMPLE_ADAM_HPP
#define DREIDEL_OPTIM_SIMPLE_ADAM_HPP

#include "../core/Tensor.hpp"
#include <vector>
#include <cmath>
#include <map>
#include <iostream>

namespace dreidel {
namespace optim {

template <typename T>
class SimpleAdam {
public:
    SimpleAdam(T lr = 1e-3, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0), use_coord_clipping_(false), clip_threshold_(1.0) {}

    void set_coordinate_wise_clipping(bool enable, T threshold = 1.0) {
        use_coord_clipping_ = enable;
        clip_threshold_ = threshold;
    }

    void add_parameters(const std::vector<Tensor<T>*>& params, const std::vector<Tensor<T>*>& grads) {
        if (params.size() != grads.size()) throw std::invalid_argument("Params and Grads size mismatch");
        for (size_t i = 0; i < params.size(); ++i) {
            params_.push_back(params[i]);
            grads_.push_back(grads[i]);

            // Init moments
            m_.emplace_back(params[i]->shape());
            m_.back().fill(0);
            v_.emplace_back(params[i]->shape());
            v_.back().fill(0);
        }
    }

    void step() {
        t_++;
        T lr_t = lr_ * std::sqrt(1.0 - std::pow(beta2_, t_)) / (1.0 - std::pow(beta1_, t_));

        for (size_t i = 0; i < params_.size(); ++i) {
            Tensor<T>* p = params_[i];
            Tensor<T>* g = grads_[i];
            Tensor<T>& m = m_[i];
            Tensor<T>& v = v_[i];

            T* p_ptr = p->data();
            T* g_ptr = g->data(); // Not const because we might clip in-place
            T* m_ptr = m.data();
            T* v_ptr = v.data();

            size_t size = p->size();

            #pragma omp parallel for
            for (size_t k = 0; k < size; ++k) {
                T grad = g_ptr[k];

                // Coordinate-Wise Clipping (The "Gem")
                // Clip gradients per frequency band / element.
                if (use_coord_clipping_) {
                    if (std::abs(grad) > clip_threshold_) {
                        grad = (grad > 0) ? clip_threshold_ : -clip_threshold_;
                        // We modify the gradient used for update.
                        // Optionally write back to g_ptr if we want to persist clipping?
                        // Usually clipping modifies the gradient tensor.
                        g_ptr[k] = grad;
                    }
                }

                // Update moments
                m_ptr[k] = beta1_ * m_ptr[k] + (1.0 - beta1_) * grad;
                v_ptr[k] = beta2_ * v_ptr[k] + (1.0 - beta2_) * grad * grad;

                // Update param
                p_ptr[k] -= lr_t * m_ptr[k] / (std::sqrt(v_ptr[k]) + epsilon_);
            }
        }
    }

    void zero_grad() {
        for (auto* g : grads_) {
            g->fill(0);
        }
    }

private:
    T lr_, beta1_, beta2_, epsilon_;
    size_t t_;
    bool use_coord_clipping_;
    T clip_threshold_;
    std::vector<Tensor<T>*> params_;
    std::vector<Tensor<T>*> grads_;
    std::vector<Tensor<T>> m_;
    std::vector<Tensor<T>> v_;
};

} // namespace optim
} // namespace dreidel

#endif
