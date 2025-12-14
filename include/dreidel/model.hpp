#ifndef DREIDEL_MODEL_HPP
#define DREIDEL_MODEL_HPP

#include <vector>
#include <memory>
#include <string>
#include <iostream>

#include "layers/Layer.hpp"
#include "optim/Optimizer.hpp"
#include "optim/SGD.hpp"

namespace dreidel {

template <typename T, BackendType B = BackendType::CPU>
class Sequential {
public:
    Sequential() = default;

    void add(std::shared_ptr<layers::Layer<T, B>> layer) {
        layers_.push_back(layer);
    }

    Tensor<T, B> forward(const Tensor<T, B>& input) {
        Tensor<T, B> output = input;
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        return output;
    }

    void backward(const Tensor<T, B>& grad_output) {
        Tensor<T, B> grad = grad_output;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
    }

    // A simple compile method to setup optimizer
    // In real frameworks, this might do more.
    // For now, we assume user manages optimizer registration or we do it here.
    void compile(std::shared_ptr<optim::SGD<T, B>> optimizer) {
        optimizer_ = optimizer;
        // Register parameters
        for (auto& layer : layers_) {
            auto params = layer->parameters();
            auto grads = layer->gradients();
            if (!params.empty()) {
                optimizer_->add_parameters(params, grads);
            }
        }
    }

    // Train step
    // Returns loss value (needs loss function passed in, or as arg)
    // Let's assume user passes a loss lambda or object to train_step?
    // Or we stick to the requested "Sequential model runner" which usually implies .fit()
    // For basic flow, let's expose train_step taking input and target and loss function.

    // Minimal Loss interface
    template<typename LossFunc>
    T train_step(const Tensor<T, B>& input, const Tensor<T, B>& target, LossFunc loss_fn) {
        // Forward
        Tensor<T, B> prediction = forward(input);

        // Loss
        T loss_val = loss_fn.compute(prediction, target);
        Tensor<T, B> grad_loss = loss_fn.gradient(prediction, target);

        // Backward
        optimizer_->zero_grad();
        backward(grad_loss);

        // Update
        optimizer_->step();

        return loss_val;
    }

private:
    std::vector<std::shared_ptr<layers::Layer<T, B>>> layers_;
    std::shared_ptr<optim::SGD<T, B>> optimizer_;
};

} // namespace dreidel

#endif // DREIDEL_MODEL_HPP
