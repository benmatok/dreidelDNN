#pragma once

#include "../layers/Layer.hpp"
#include "../core/Tensor.hpp"
#include "ZenithEncoder.hpp"
#include "ZenithDecoder.hpp"
#include <vector>
#include <memory>
#include <cmath>

namespace dreidel {
namespace models {

class ZenithHydra : public layers::Layer<float> {
public:
    // Zenith-Hydra:
    // One Encoder
    // Two Decoders: Purist (Reconstruction), Dreamer (Inpainting)
    // Loss Weights: log_vars (3 params)

    ZenithHydra(size_t batch_size = 1) : batch_size_(batch_size) {
        encoder_ = std::make_unique<ZenithEncoder>(batch_size);
        decoder_purist_ = std::make_unique<ZenithDecoder>(batch_size);
        decoder_dreamer_ = std::make_unique<ZenithDecoder>(batch_size);

        // Learnable uncertainty parameters (log variance)
        // Initialize to 0.0 (variance = 1.0)
        log_vars_ = Tensor<float>({3}); // [Clean, Robust, Stable]
        log_vars_.fill(0.0f);

        grad_log_vars_ = Tensor<float>({3});
        grad_log_vars_.fill(0.0f);
    }

    // Forward method returns the "Clean" path output by default,
    // but Hydra is usually used by calling specific heads.
    Tensor<float> forward(const Tensor<float>& input) override {
        // Default: Clean Path (Purist)
        Tensor<float> latents = encoder_->forward(input);
        return decoder_purist_->forward(latents);
    }

    // Specialized forwards
    Tensor<float> forward_purist(const Tensor<float>& input) {
        Tensor<float> latents = encoder_->forward(input);
        return decoder_purist_->forward(latents);
    }

    Tensor<float> forward_dreamer(const Tensor<float>& input) {
        Tensor<float> latents = encoder_->forward(input);
        return decoder_dreamer_->forward(latents);
    }

    // Backward needs careful handling because we have multiple paths.
    // The training loop should manage backprop through specific heads.
    // However, if we call backward() on this class, it's ambiguous.
    // We'll leave the default backward as "not implemented" or just backward through default path.
    Tensor<float> backward(const Tensor<float>& grad_output) override {
        // Assume default path (Purist)
        Tensor<float> d_latents = decoder_purist_->backward(grad_output);
        return encoder_->backward(d_latents);
    }

    // Specific backward for Hydra training
    Tensor<float> backward_hydra(const Tensor<float>& grad_purist, const Tensor<float>& grad_dreamer) {
        // Backprop through both decoders
        Tensor<float> d_z_purist = decoder_purist_->backward(grad_purist);
        Tensor<float> d_z_dreamer = decoder_dreamer_->backward(grad_dreamer);

        // Sum gradients at latent space
        size_t size = d_z_purist.size();
        Tensor<float> d_z_total = d_z_purist; // copy shape/buffer
        float* tot_ptr = d_z_total.data();
        const float* p_ptr = d_z_purist.data();
        const float* d_ptr = d_z_dreamer.data();

        #pragma omp parallel for
        for(size_t i=0; i<size; ++i) {
            tot_ptr[i] = p_ptr[i] + d_ptr[i];
        }

        return encoder_->backward(d_z_total);
    }

    // Single path backward helpers
    Tensor<float> backward_purist(const Tensor<float>& grad_output) {
        Tensor<float> d_latents = decoder_purist_->backward(grad_output);
        return encoder_->backward(d_latents);
    }

    Tensor<float> backward_dreamer(const Tensor<float>& grad_output) {
        Tensor<float> d_latents = decoder_dreamer_->backward(grad_output);
        return encoder_->backward(d_latents);
    }

    std::vector<Tensor<float>*> parameters() override {
        std::vector<Tensor<float>*> p;
        auto add = [&](auto& l) {
            auto pp = l->parameters();
            p.insert(p.end(), pp.begin(), pp.end());
        };
        add(encoder_);
        add(decoder_purist_);
        add(decoder_dreamer_);
        p.push_back(&log_vars_);
        return p;
    }

    std::vector<Tensor<float>*> gradients() override {
        std::vector<Tensor<float>*> g;
        auto add = [&](auto& l) {
            auto gg = l->gradients();
            g.insert(g.end(), gg.begin(), gg.end());
        };
        add(encoder_);
        add(decoder_purist_);
        add(decoder_dreamer_);
        g.push_back(&grad_log_vars_);
        return g;
    }

    void set_training(bool training) override {
        Layer::set_training(training);
        encoder_->set_training(training);
        decoder_purist_->set_training(training);
        decoder_dreamer_->set_training(training);
    }

    std::string name() const override { return "ZenithHydra"; }

private:
    std::unique_ptr<ZenithEncoder> encoder_;
    std::unique_ptr<ZenithDecoder> decoder_purist_;
    std::unique_ptr<ZenithDecoder> decoder_dreamer_;

    Tensor<float> log_vars_;
    Tensor<float> grad_log_vars_;

    size_t batch_size_;
};

} // namespace models
} // namespace dreidel
