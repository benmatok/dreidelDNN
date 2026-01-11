#pragma once

#include "ZenithOverhaulAE.hpp"
#include "../layers/Layer.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/GELU.hpp"
#include "../layers/ZenithBlock.hpp"
#include <vector>
#include <memory>
#include <tuple>

namespace dreidel {
namespace models {

template <typename T>
class GhostProjection : public layers::Layer<T> {
public:
    GhostProjection(size_t in_channels, size_t out_channels) {
        // Conv 1x1 -> GELU -> Conv 1x1
        // We use ZenithBlock(kernel=1) instead of Conv2D for "Alien Speed" optimization.
        // args: in, out, k, spec_dim, ifwht, dilated, gating, stride, upscale, init, slm, seq, eps
        block1_ = std::make_unique<layers::ZenithBlock<T>>(in_channels, out_channels, 1, in_channels, true, false, false, 1, 1, "he", false, false, 1.0f);
        act_ = std::make_unique<layers::GELU<T>>();
        block2_ = std::make_unique<layers::ZenithBlock<T>>(out_channels, out_channels, 1, out_channels, true, false, false, 1, 1, "he", false, false, 1.0f);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = block1_->forward(input);
        x = act_->forward(x);
        x = block2_->forward(x);
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> g = block2_->backward(grad_output);
        g = act_->backward(g);
        g = block1_->backward(g);
        return g;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto p1 = block1_->parameters(); params.insert(params.end(), p1.begin(), p1.end());
        auto p2 = block2_->parameters(); params.insert(params.end(), p2.begin(), p2.end());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto g1 = block1_->gradients(); grads.insert(grads.end(), g1.begin(), g1.end());
        auto g2 = block2_->gradients(); grads.insert(grads.end(), g2.begin(), g2.end());
        return grads;
    }

    void set_training(bool training) override {
        block1_->set_training(training);
        act_->set_training(training);
        block2_->set_training(training);
    }

    std::string name() const override { return "GhostProjection"; }

private:
    std::unique_ptr<layers::ZenithBlock<T>> block1_;
    std::unique_ptr<layers::GELU<T>> act_;
    std::unique_ptr<layers::ZenithBlock<T>> block2_;
};

template <typename T>
class ZenithGhostAE : public layers::Layer<T> {
public:
    struct TrainOutput {
        Tensor<T> reconstruction;
        std::vector<Tensor<T>> ghost_preds;    // Predictions from ghosts
        std::vector<Tensor<T>> encoder_targets; // Real encoder features
    };

    ZenithGhostAE() {
        // 1. Stem: 3 -> 32
        stem_ = std::make_unique<layers::Conv2D<T>>(3, 32, 3, 1, 1);

        // 2. Encoders (7 blocks)
        // Enc1: 32 -> 64
        encoders_.push_back(std::make_unique<ZenithCompressBlock<T>>(32));
        // Enc2: 64 -> 128
        encoders_.push_back(std::make_unique<ZenithCompressBlock<T>>(64));
        // Enc3: 128 -> 256
        encoders_.push_back(std::make_unique<ZenithCompressBlock<T>>(128));
        // Enc4: 256 -> 512
        encoders_.push_back(std::make_unique<ZenithCompressBlock<T>>(256));
        // Enc5: 512 -> 1024
        encoders_.push_back(std::make_unique<ZenithCompressBlock<T>>(512));
        // Enc6: 1024 -> 2048
        encoders_.push_back(std::make_unique<ZenithCompressBlock<T>>(1024));
        // Enc7: 2048 -> 4096 (Bottleneck)
        encoders_.push_back(std::make_unique<ZenithCompressBlock<T>>(2048));

        // 3. Decoders (7 blocks)
        // Dec7: 4096 -> 2048
        decoders_.push_back(std::make_unique<ZenithExpandBlock<T>>(4096));
        // Dec6: 2048 -> 1024
        decoders_.push_back(std::make_unique<ZenithExpandBlock<T>>(2048));
        // Dec5: 1024 -> 512
        decoders_.push_back(std::make_unique<ZenithExpandBlock<T>>(1024));
        // Dec4: 512 -> 256
        decoders_.push_back(std::make_unique<ZenithExpandBlock<T>>(512));
        // Dec3: 256 -> 128
        decoders_.push_back(std::make_unique<ZenithExpandBlock<T>>(256));
        // Dec2: 128 -> 64
        decoders_.push_back(std::make_unique<ZenithExpandBlock<T>>(128));
        // Dec1: 64 -> 32
        decoders_.push_back(std::make_unique<ZenithExpandBlock<T>>(64));

        // 4. Head: 32 -> 3
        head_ = std::make_unique<layers::Conv2D<T>>(32, 3, 3, 1, 1);

        // 5. Ghost Projections
        // Map Decoder Outputs -> Encoder Outputs
        // Dec0 (Dec7) Out (2048) matches Enc5 (Enc6) Out (2048) -> Ghost 0
        ghosts_.push_back(std::make_unique<GhostProjection<T>>(2048, 2048));
        // Dec1 (Dec6) Out (1024) matches Enc4 (Enc5) Out (1024) -> Ghost 1
        ghosts_.push_back(std::make_unique<GhostProjection<T>>(1024, 1024));
        // Dec2 (Dec5) Out (512) matches Enc3 (Enc4) Out (512) -> Ghost 2
        ghosts_.push_back(std::make_unique<GhostProjection<T>>(512, 512));
        // Dec3 (Dec4) Out (256) matches Enc2 (Enc3) Out (256) -> Ghost 3
        ghosts_.push_back(std::make_unique<GhostProjection<T>>(256, 256));
        // Dec4 (Dec3) Out (128) matches Enc1 (Enc2) Out (128) -> Ghost 4
        ghosts_.push_back(std::make_unique<GhostProjection<T>>(128, 128));
        // Dec5 (Dec2) Out (64) matches Enc0 (Enc1) Out (64) -> Ghost 5
        ghosts_.push_back(std::make_unique<GhostProjection<T>>(64, 64));
        // Dec6 (Dec1) Out (32) matches Stem Out (32) -> Ghost 6
        ghosts_.push_back(std::make_unique<GhostProjection<T>>(32, 32));
    }

    // Standard Inference Forward
    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = input;
        x = stem_->forward(x);
        for (auto& enc : encoders_) x = enc->forward(x);
        for (auto& dec : decoders_) x = dec->forward(x);
        x = head_->forward(x);
        return x;
    }

    // Training Forward: Returns Reconstruction, Ghost Preds, Targets
    TrainOutput forward_train(const Tensor<T>& input) {
        TrainOutput out;

        // 1. Encoder Path
        Tensor<T> x_stem = stem_->forward(input);

        std::vector<Tensor<T>> x_encs;
        x_encs.push_back(x_stem); // Index 0 is Stem Out

        Tensor<T> curr = x_stem;
        for (size_t i = 0; i < encoders_.size(); ++i) {
            curr = encoders_[i]->forward(curr);
            // Store output of Enc0..Enc5 (Enc6 is bottleneck, we don't ghost it directly?
            // Wait, Ghost0 matches Dec0 Out(2048) to Enc5 Out(2048).
            // So we need Enc0..Enc5 outputs.
            // And Stem output for Ghost6.
            // Enc6 output is fed to Dec0.
            if (i < 6) {
                x_encs.push_back(curr);
            }
        }
        // x_encs content: [StemOut, Enc0Out, ..., Enc5Out]
        // Size: 1 + 6 = 7.
        // Reverse order of matching:
        // Ghost0 -> Enc5Out (last in x_encs)
        // Ghost6 -> StemOut (first in x_encs)

        // Pass through Enc6 (Bottleneck)
        // Note: curr is already Enc6 Output from loop if i==6

        // 2. Decoder Path
        Tensor<T> z = curr; // Start from Bottleneck output
        std::vector<Tensor<T>> ghost_preds;

        // Decoders loop
        // We have 7 decoders.
        // We have 7 ghosts.
        // Ghost i corresponds to Decoder i output.
        // Target is x_encs[6 - i].

        for (size_t i = 0; i < decoders_.size(); ++i) {
            z = decoders_[i]->forward(z);

            // Apply Ghost Projection
            Tensor<T> pred = ghosts_[i]->forward(z);
            ghost_preds.push_back(pred);

            // Collect Target
            // i=0 (Dec0) -> Target Enc5Out (Index 6 in x_encs)
            // i=6 (Dec6) -> Target StemOut (Index 0 in x_encs)
            out.encoder_targets.push_back(x_encs[x_encs.size() - 1 - i]);
        }
        out.ghost_preds = ghost_preds;

        // 3. Head
        out.reconstruction = head_->forward(z);
        return out;
    }

    // Standard Backward: Gradients for Reconstruction Only (ignores Ghosts)
    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> g = grad_output;
        g = head_->backward(g);
        for (auto it = decoders_.rbegin(); it != decoders_.rend(); ++it) {
            g = (*it)->backward(g);
        }
        for (auto it = encoders_.rbegin(); it != encoders_.rend(); ++it) {
            g = (*it)->backward(g);
        }
        g = stem_->backward(g);
        return g;
    }

    // Training Backward: Handles Recon Loss + Ghost Loss
    // grad_ghost_preds: Gradients of Loss w.r.t Ghost Predictions
    // Note: We assume targets are detached (no grad flow to encoder from ghost loss).
    void backward_train(const Tensor<T>& grad_recon, const std::vector<Tensor<T>>& grad_ghost_preds) {
        Tensor<T> g = grad_recon;
        g = head_->backward(g);

        // Iterate backwards through decoders
        // decoders_ has 0..6. rbegin gives 6..0.
        // grad_ghost_preds corresponds to ghosts_[0..6].
        // So when processing decoders_[i], we need grad_ghost_preds[i].

        int idx = static_cast<int>(decoders_.size()) - 1;
        for (auto it = decoders_.rbegin(); it != decoders_.rend(); ++it, --idx) {
            // 1. Calculate gradient from Ghost Projection
            // dL_ghost / dZ = Ghost.backward(dL / dPred)
            Tensor<T> g_ghost = ghosts_[idx]->backward(grad_ghost_preds[idx]);

            // 2. Add to incoming gradient from deeper layers/head
            // g += g_ghost
            T* g_ptr = g.data();
            const T* gg_ptr = g_ghost.data();
            #pragma omp parallel for
            for(size_t k=0; k<g.size(); ++k) g_ptr[k] += gg_ptr[k];

            // 3. Backprop through Decoder Block
            g = (*it)->backward(g);
        }

        // Backprop through Encoders and Stem (Standard)
        for (auto it = encoders_.rbegin(); it != encoders_.rend(); ++it) {
            g = (*it)->backward(g);
        }
        g = stem_->backward(g);
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        // Main Architecture Order (match ZenithOverhaulAE)
        auto p_stem = stem_->parameters(); params.insert(params.end(), p_stem.begin(), p_stem.end());
        for(auto& enc : encoders_) {
            auto p = enc->parameters(); params.insert(params.end(), p.begin(), p.end());
        }
        for(auto& dec : decoders_) {
            auto p = dec->parameters(); params.insert(params.end(), p.begin(), p.end());
        }
        auto p_head = head_->parameters(); params.insert(params.end(), p_head.begin(), p_head.end());

        // Ghost Parameters (Extra)
        for(auto& gh : ghosts_) {
            auto p = gh->parameters(); params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto g_stem = stem_->gradients(); grads.insert(grads.end(), g_stem.begin(), g_stem.end());
        for(auto& enc : encoders_) {
            auto g = enc->gradients(); grads.insert(grads.end(), g.begin(), g.end());
        }
        for(auto& dec : decoders_) {
            auto g = dec->gradients(); grads.insert(grads.end(), g.begin(), g.end());
        }
        auto g_head = head_->gradients(); grads.insert(grads.end(), g_head.begin(), g_head.end());

        for(auto& gh : ghosts_) {
            auto g = gh->gradients(); grads.insert(grads.end(), g.begin(), g.end());
        }
        return grads;
    }

    void set_training(bool training) override {
        stem_->set_training(training);
        for(auto& enc : encoders_) enc->set_training(training);
        for(auto& dec : decoders_) dec->set_training(training);
        head_->set_training(training);
        for(auto& gh : ghosts_) gh->set_training(training);
    }

    std::string name() const override { return "ZenithGhostAE"; }

private:
    std::unique_ptr<layers::Conv2D<T>> stem_;
    std::vector<std::unique_ptr<ZenithCompressBlock<T>>> encoders_;
    std::vector<std::unique_ptr<ZenithExpandBlock<T>>> decoders_;
    std::unique_ptr<layers::Conv2D<T>> head_;
    std::vector<std::unique_ptr<GhostProjection<T>>> ghosts_;
};

} // namespace models
} // namespace dreidel
