#ifndef DREIDEL_CORE_MODEL_HPP
#define DREIDEL_CORE_MODEL_HPP

#include "../layers/Layer.hpp"
#include <vector>

namespace dreidel {

template <typename T>
class Model : public layers::Layer<T> {
public:
    virtual ~Model() = default;

    void add_layer(layers::Layer<T>* layer) {
        layers_.push_back(layer);
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        for(auto layer : layers_) {
            auto l_params = layer->parameters();
            params.insert(params.end(), l_params.begin(), l_params.end());
        }
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        for(auto layer : layers_) {
            auto l_grads = layer->gradients();
            grads.insert(grads.end(), l_grads.begin(), l_grads.end());
        }
        return grads;
    }

    // Default backward passes through layers in reverse?
    // ZenithNano implements its own forward/backward.
    // ZenithDiscriminator implements its own forward.
    // We need a default backward if we just use add_layer?
    // But Layer requires pure virtual forward/backward.

    // ZenithDiscriminator implements forward explicitly.
    // It should implement backward explicitly if not using a Sequential container logic.
    // But wait, Model here is just a helper to aggregate parameters?

    std::string name() const override { return "Model"; }

protected:
    std::vector<layers::Layer<T>*> layers_;
};

} // namespace dreidel

#endif // DREIDEL_CORE_MODEL_HPP
