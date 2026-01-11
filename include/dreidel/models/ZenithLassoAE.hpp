#pragma once

#include "../layers/Layer.hpp"
#include "../layers/ZenithBlock.hpp"
#include "../layers/Upscale2D.hpp"
#include "../optim/ZenithRegularizer.hpp"
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
        layers_.push_back(std::make_unique<layers::ZenithBlock<T>>(128, 128, 3, 128, true, true, false, 1, 1, "he", false, false, 1.0f));

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

    // Expose encoder for latent space analysis
    Tensor<T> forward_encoder(const Tensor<T>& input) {
        Tensor<T> x = input;
        // Run first 3 layers (Indices 0, 1, 2)
        for(size_t i=0; i<3 && i<layers_.size(); ++i) {
            x = layers_[i]->forward(x);
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

    // Apply Group Lasso to Mixing Weights of all Zenith Blocks
    void apply_lasso(float lambda, float lr) {
        if (lambda <= 0.0f) return;
        for(auto& layer_base : layers_) {
            auto* layer = dynamic_cast<layers::ZenithBlock<T>*>(layer_base.get());
            if(layer) {
                 auto params = layer->parameters();
                 // mixing_weights_ is params[2] (Left, Center, Right stacked 3xChannels)
                 // params[2] is Tensor<T>*, we need data pointer.
                 // Ensure we have enough params (ZenithBlock returns at least 3)
                 if (params.size() >= 3) {
                     optim::apply_group_lasso_avx(params[2]->data(), params[2]->size(), lambda, lr);
                 }
            }
        }
    }

    // Helper to calculate sparsity statistics
    void get_sparsity_stats(size_t& total_blocks, size_t& zero_blocks) {
        total_blocks = 0;
        zero_blocks = 0;
        for(auto& layer_base : layers_) {
            auto* layer = dynamic_cast<layers::ZenithBlock<T>*>(layer_base.get());
            if(layer) {
                auto params = layer->parameters();
                if (params.size() >= 3) {
                    float* w = params[2]->data();
                    size_t s = params[2]->size();
                    for(size_t i=0; i<s; i+=8) {
                        if (i + 8 > s) break;
                        float energy = 0;
                        for(int j=0; j<8; j++) energy += std::abs(w[i+j]);
                        total_blocks++;
                        if (energy < 1e-9f) zero_blocks++;
                    }
                }
            }
        }
    }

    void set_epsilon(float eps) {
        for(auto& layer_base : layers_) {
            auto* layer = dynamic_cast<layers::ZenithBlock<T>*>(layer_base.get());
            if(layer) {
                layer->set_epsilon(eps);
            }
        }
    }

    std::string name() const override { return "ZenithLassoAE"; }

private:
    std::vector<std::unique_ptr<layers::Layer<T>>> layers_;
};

} // namespace models
} // namespace dreidel
