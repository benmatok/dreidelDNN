#ifndef DREIDEL_OPTIM_SGD_HPP
#define DREIDEL_OPTIM_SGD_HPP

#include "Optimizer.hpp"
#include <vector>

namespace dreidel {
namespace optim {

template <typename T, BackendType B = BackendType::CPU>
class SGD : public Optimizer<T, B> {
public:
    SGD(T learning_rate) : learning_rate_(learning_rate) {}

    void add_parameters(std::vector<Tensor<T, B>*> params, std::vector<Tensor<T, B>*> grads) override {
        if (params.size() != grads.size()) {
             throw std::invalid_argument("Params and grads size mismatch");
        }
        for(auto* p : params) parameters_.push_back(p);
        for(auto* g : grads) gradients_.push_back(g);
    }

    void step() override {
        for (size_t i = 0; i < parameters_.size(); ++i) {
            Tensor<T, B>* param = parameters_[i];
            Tensor<T, B>* grad = gradients_[i];

            Tensor<T, B> update = (*grad) * (-learning_rate_);
            *param = (*param) + update;
        }
    }

    void zero_grad() override {
        for (auto* grad : gradients_) {
            grad->fill(0);
        }
    }

private:
    T learning_rate_;
    std::vector<Tensor<T, B>*> parameters_;
    std::vector<Tensor<T, B>*> gradients_;
};

} // namespace optim
} // namespace dreidel

#endif // DREIDEL_OPTIM_SGD_HPP
