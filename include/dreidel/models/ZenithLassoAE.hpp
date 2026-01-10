#pragma once

#include "../layers/Layer.hpp"
#include "../layers/ZenithBlock.hpp"
#include "../layers/Upscale2D.hpp"
#include <vector>
#include <memory>

namespace dreidel {
namespace models {

/**
 * @brief ZenithLassoAE: The SOTA Spectral Autoencoder optimized for stability and speed.
 *
 * Features:
 * - ZenithBlock layers with stabilized GroupNorm (eps=1.0) to prevent divergence.
 * - Deep 6-layer architecture with 1x1 bottleneck.
 * - Designed for training with Group Lasso Regularization to induce >70% sparsity.
 */
template <typename T>
class ZenithLassoAE : public layers::Layer<T> {
public:
    ZenithLassoAE() {
        // Encoder
        // 1. 1 -> 128 (Stride 4)
        layers_.push_back(std::make_unique<layers::ZenithBlock<T>>(1, 128, 3, 128, true, true, false, 4, 1, "he", false, false, 1.0f));
        // 2. 128 -> 128 (Stride 4)
        layers_.push_back(std::make_unique<layers::ZenithBlock<T>>(128, 128, 3, 128, true, true, false, 4, 1, "he", false, false, 1.0f));
        // 3. 128 -> 64 (Stride 4)
        layers_.push_back(std::make_unique<layers::ZenithBlock<T>>(128, 64, 3, 128, true, true, false, 4, 1, "he", false, false, 1.0f));

        // Decoder
        // 4. Upscale 4 -> 64 -> 128
        layers_.push_back(std::make_unique<layers::Upscale2D<T>>(4));
        layers_.push_back(std::make_unique<layers::ZenithBlock<T>>(64, 128, 3, 128, true, true, false, 1, 1, "he", false, false, 1.0f));

        // 5. Upscale 4 -> 128 -> 128
        layers_.push_back(std::make_unique<layers::Upscale2D<T>>(4));
        layers_.push_back(std::make_unique<layers::ZenithBlock<T>>(128, 128, 3, 128, true, true, false, 1, 1, "he", false, false, 1.0f)); // Note: train_zenith_lasso had stride 4 typo in search block, verified stride 1

        // 6. Upscale 4 -> 128 -> 1
        layers_.push_back(std::make_unique<layers::Upscale2D<T>>(4));
        layers_.push_back(std::make_unique<layers::ZenithBlock<T>>(128, 1, 3, 128, true, true, false, 1, 1, "he", false, false, 1.0f));
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = input;
        for(auto& layer : layers_) {
            x = layer->forward(x);
        }
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> g = grad_output;
        for(auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            g = (*it)->backward(g);
        }
        return g;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        for(auto& layer : layers_) {
            auto p = layer->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        for(auto& layer : layers_) {
            auto g = layer->gradients();
            grads.insert(grads.end(), g.begin(), g.end());
        }
        return grads;
    }

    std::string name() const override { return "ZenithLassoAE"; }

private:
    std::vector<std::unique_ptr<layers::Layer<T>>> layers_;
};

} // namespace models
} // namespace dreidel
