#pragma once

#include <vector>
#include <memory>
#include "layers/layer.hpp"
#include "optim/kfac.hpp"

namespace dreidel {

/**
 * @brief Sequential Model container.
 */
template <typename T>
class Sequential {
public:
    void add(std::shared_ptr<layers::Layer<T>> layer) {
        layers_.push_back(layer);
    }

    // Helper for adding by type
    template <typename LayerType, typename... Args>
    void add(Args&&... args) {
        layers_.push_back(std::make_shared<LayerType>(std::forward<Args>(args)...));
    }

    core::Tensor<T> forward(const core::Tensor<T>& input) {
        core::Tensor<T> out = input;
        for (auto& layer : layers_) {
            out = layer->forward(out);
        }
        return out;
    }

    void backward(const core::Tensor<T>& loss_grad) {
        core::Tensor<T> grad = loss_grad;
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
    }

    // Mock compilation
    void compile(optim::KFAC& optimizer) {
        optimizer_ = &optimizer;
    }

    // Mock fit
    void fit() {
        // Training loop would go here
    }

private:
    std::vector<std::shared_ptr<layers::Layer<T>>> layers_;
    optim::KFAC* optimizer_ = nullptr;
};

} // namespace dreidel
