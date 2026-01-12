#pragma once

#include "../layers/Layer.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/PixelShuffle.hpp"
#include "../layers/GroupNorm.hpp"
#include "../layers/GELU.hpp"
#include <vector>
#include <memory>
#include <string>

namespace dreidel {
namespace models {

template <typename T>
class ConvGhostProjection : public layers::Layer<T> {
public:
    ConvGhostProjection(size_t in_channels, size_t out_channels, const std::string& init_scheme = "he") {
        // Conv 1x1 -> GELU -> Conv 1x1
        // Replaces ZenithBlock(1x1)

        block1_ = std::make_unique<layers::Conv2D<T>>(in_channels, out_channels, 1, 1, 0);
        // Note: ZenithBlock includes GroupNorm at the end. Should we?
        // ZenithGhostAE's GhostProjection uses ZenithBlock which has GN.
        // So we should add GroupNorm here too to be parallel.
        gn1_ = std::make_unique<layers::GroupNorm<T>>(32, out_channels);

        act_ = std::make_unique<layers::GELU<T>>();

        block2_ = std::make_unique<layers::Conv2D<T>>(out_channels, out_channels, 1, 1, 0);
        gn2_ = std::make_unique<layers::GroupNorm<T>>(32, out_channels);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = block1_->forward(input);
        x = gn1_->forward(x);
        x = act_->forward(x);
        x = block2_->forward(x);
        x = gn2_->forward(x);
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> g = gn2_->backward(grad_output);
        g = block2_->backward(g);
        g = act_->backward(g);
        g = gn1_->backward(g);
        g = block1_->backward(g);
        return g;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto p1 = block1_->parameters(); params.insert(params.end(), p1.begin(), p1.end());
        auto p_gn1 = gn1_->parameters(); params.insert(params.end(), p_gn1.begin(), p_gn1.end());
        auto p2 = block2_->parameters(); params.insert(params.end(), p2.begin(), p2.end());
        auto p_gn2 = gn2_->parameters(); params.insert(params.end(), p_gn2.begin(), p_gn2.end());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto g1 = block1_->gradients(); grads.insert(grads.end(), g1.begin(), g1.end());
        auto g_gn1 = gn1_->gradients(); grads.insert(grads.end(), g_gn1.begin(), g_gn1.end());
        auto g2 = block2_->gradients(); grads.insert(grads.end(), g2.begin(), g2.end());
        auto g_gn2 = gn2_->gradients(); grads.insert(grads.end(), g_gn2.begin(), g_gn2.end());
        return grads;
    }

    void set_training(bool training) override {
        // block1_->set_training(training); // Conv2D doesn't have set_training
        // gn1_->set_training(training); // GroupNorm doesn't typically need it unless tracking stats
    }

    std::string name() const override { return "ConvGhostProjection"; }

private:
    std::unique_ptr<layers::Conv2D<T>> block1_;
    std::unique_ptr<layers::GroupNorm<T>> gn1_;
    std::unique_ptr<layers::GELU<T>> act_;
    std::unique_ptr<layers::Conv2D<T>> block2_;
    std::unique_ptr<layers::GroupNorm<T>> gn2_;
};

template <typename T>
class ConvCompressBlock : public layers::Layer<T> {
public:
    ConvCompressBlock(size_t in_channels, const std::string& init_scheme = "he") {
        size_t expanded_dim = in_channels * 4;
        size_t out_channels = in_channels * 2;

        // 1. Physical Transmutation (Space -> Depth)
        unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(2);

        // 2. PixelMix Group Conv (Local Fidelity)
        pixel_mix_ = std::make_unique<layers::Conv2D<T>>(expanded_dim, expanded_dim, 3, 1, 1, in_channels);
        norm_ = std::make_unique<layers::GroupNorm<T>>(32, expanded_dim);
        act_ = std::make_unique<layers::GELU<T>>();

        // 3. Dense Mixer (Replaces Spectral Mixer)
        // Replaces ZenithBlock(1x1) which has GN at the end.
        mixer_ = std::make_unique<layers::Conv2D<T>>(expanded_dim, expanded_dim, 1, 1, 0);
        mixer_norm_ = std::make_unique<layers::GroupNorm<T>>(32, expanded_dim);

        // 4. Distillation (Compression)
        compressor_ = std::make_unique<layers::Conv2D<T>>(expanded_dim, out_channels, 1);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // 1. Physics
        Tensor<T> x = unshuffle_->forward(input);

        // 2. Local Fidelity (Residual)
        Tensor<T> res = x;
        x = pixel_mix_->forward(x);
        x = norm_->forward(x);
        x = act_->forward(x);

        float* x_ptr = x.data();
        const float* res_ptr = res.data();
        #pragma omp parallel for
        for(size_t i=0; i<x.size(); ++i) x_ptr[i] += res_ptr[i];

        // 3. Dense Context (Residual)
        res = x;
        x = mixer_->forward(x);
        x = mixer_norm_->forward(x);

        x_ptr = x.data();
        res_ptr = res.data();
        #pragma omp parallel for
        for(size_t i=0; i<x.size(); ++i) x_ptr[i] += res_ptr[i];

        // 4. Compress
        x = compressor_->forward(x);
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // 4. Compress
        Tensor<T> g = compressor_->backward(grad_output);

        // 3. Dense Context (Residual)
        Tensor<T> g_res = g;
        Tensor<T> g_mix = mixer_norm_->backward(g);
        g_mix = mixer_->backward(g_mix);

        float* g_ptr = g_mix.data();
        const float* gr_ptr = g_res.data();
        #pragma omp parallel for
        for(size_t i=0; i<g_mix.size(); ++i) g_ptr[i] += gr_ptr[i];
        g = g_mix;

        // 2. Local Fidelity (Residual)
        g_res = g;
        Tensor<T> g_act = act_->backward(g);
        Tensor<T> g_norm = norm_->backward(g_act);
        Tensor<T> g_pixel = pixel_mix_->backward(g_norm);

        g_ptr = g_pixel.data();
        gr_ptr = g_res.data();
        #pragma omp parallel for
        for(size_t i=0; i<g_pixel.size(); ++i) g_ptr[i] += gr_ptr[i];
        g = g_pixel;

        // 1. Physics
        g = unshuffle_->backward(g);
        return g;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto append = [&](layers::Layer<T>* l) {
            auto p = l->parameters();
            params.insert(params.end(), p.begin(), p.end());
        };
        append(pixel_mix_.get());
        append(norm_.get());
        append(mixer_.get());
        append(mixer_norm_.get());
        append(compressor_.get());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto append = [&](layers::Layer<T>* l) {
            auto g = l->gradients();
            grads.insert(grads.end(), g.begin(), g.end());
        };
        append(pixel_mix_.get());
        append(norm_.get());
        append(mixer_.get());
        append(mixer_norm_.get());
        append(compressor_.get());
        return grads;
    }

    void set_training(bool training) override {}

    std::string name() const override { return "ConvCompressBlock"; }

private:
    std::unique_ptr<layers::PixelUnshuffle<T>> unshuffle_;
    std::unique_ptr<layers::Conv2D<T>> pixel_mix_;
    std::unique_ptr<layers::GroupNorm<T>> norm_;
    std::unique_ptr<layers::GELU<T>> act_;
    std::unique_ptr<layers::Conv2D<T>> mixer_;
    std::unique_ptr<layers::GroupNorm<T>> mixer_norm_;
    std::unique_ptr<layers::Conv2D<T>> compressor_;
};

template <typename T>
class ConvExpandBlock : public layers::Layer<T> {
public:
    ConvExpandBlock(size_t in_channels, const std::string& init_scheme = "he") {
        size_t expanded_dim = in_channels * 2;
        size_t out_channels = in_channels / 2;

        // 1. Expansion
        expander_ = std::make_unique<layers::Conv2D<T>>(in_channels, expanded_dim, 1);

        // 2. Dense Context (Replaces Spectral Mixer)
        mixer_ = std::make_unique<layers::Conv2D<T>>(expanded_dim, expanded_dim, 1, 1, 0);
        mixer_norm_ = std::make_unique<layers::GroupNorm<T>>(32, expanded_dim);

        // 3. Local Fidelity
        size_t final_channels = expanded_dim / 4;
        pixel_mix_ = std::make_unique<layers::Conv2D<T>>(expanded_dim, expanded_dim, 3, 1, 1, final_channels);
        norm_ = std::make_unique<layers::GroupNorm<T>>(32, expanded_dim);
        act_ = std::make_unique<layers::GELU<T>>();

        // 4. Physics (Depth -> Space)
        shuffle_ = std::make_unique<layers::PixelShuffle<T>>(2);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // 1. Expand
        Tensor<T> x = expander_->forward(input);

        // 2. Global (Residual)
        Tensor<T> res = x;
        x = mixer_->forward(x);
        x = mixer_norm_->forward(x);

        float* x_ptr = x.data();
        const float* res_ptr = res.data();
        #pragma omp parallel for
        for(size_t i=0; i<x.size(); ++i) x_ptr[i] += res_ptr[i];

        // 3. Local (Residual)
        res = x;
        x = pixel_mix_->forward(x);
        x = norm_->forward(x);
        x = act_->forward(x);

        x_ptr = x.data();
        res_ptr = res.data();
        #pragma omp parallel for
        for(size_t i=0; i<x.size(); ++i) x_ptr[i] += res_ptr[i];

        // 4. Shuffle
        x = shuffle_->forward(x);
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // 4. Shuffle
        Tensor<T> g = shuffle_->backward(grad_output);

        // 3. Local
        Tensor<T> g_res = g;
        Tensor<T> g_act = act_->backward(g);
        Tensor<T> g_norm = norm_->backward(g_act);
        Tensor<T> g_pixel = pixel_mix_->backward(g_norm);

        float* g_ptr = g_pixel.data();
        const float* gr_ptr = g_res.data();
        #pragma omp parallel for
        for(size_t i=0; i<g_pixel.size(); ++i) g_ptr[i] += gr_ptr[i];
        g = g_pixel;

        // 2. Global
        g_res = g;
        Tensor<T> g_mix = mixer_norm_->backward(g);
        g_mix = mixer_->backward(g_mix);

        g_ptr = g_mix.data();
        gr_ptr = g_res.data();
        #pragma omp parallel for
        for(size_t i=0; i<g_mix.size(); ++i) g_ptr[i] += gr_ptr[i];
        g = g_mix;

        // 1. Expand
        g = expander_->backward(g);
        return g;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto append = [&](layers::Layer<T>* l) {
            auto p = l->parameters();
            params.insert(params.end(), p.begin(), p.end());
        };
        append(expander_.get());
        append(mixer_.get());
        append(mixer_norm_.get());
        append(pixel_mix_.get());
        append(norm_.get());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto append = [&](layers::Layer<T>* l) {
            auto g = l->gradients();
            grads.insert(grads.end(), g.begin(), g.end());
        };
        append(expander_.get());
        append(mixer_.get());
        append(mixer_norm_.get());
        append(pixel_mix_.get());
        append(norm_.get());
        return grads;
    }

    void set_training(bool training) override {}

    std::string name() const override { return "ConvExpandBlock"; }

private:
    std::unique_ptr<layers::Conv2D<T>> expander_;
    std::unique_ptr<layers::Conv2D<T>> mixer_;
    std::unique_ptr<layers::GroupNorm<T>> mixer_norm_;
    std::unique_ptr<layers::Conv2D<T>> pixel_mix_;
    std::unique_ptr<layers::GroupNorm<T>> norm_;
    std::unique_ptr<layers::GELU<T>> act_;
    std::unique_ptr<layers::PixelShuffle<T>> shuffle_;
};


template <typename T>
class ConvGhostAE : public layers::Layer<T> {
public:
    struct TrainOutput {
        Tensor<T> reconstruction;
        std::vector<Tensor<T>> ghost_preds;
        std::vector<Tensor<T>> encoder_targets;
    };

    ConvGhostAE(const std::string& init_scheme = "he") {
        // 1. Stem: 3 -> 32
        stem_ = std::make_unique<layers::Conv2D<T>>(3, 32, 3, 1, 1);

        // 2. Encoders (7 blocks)
        encoders_.push_back(std::make_unique<ConvCompressBlock<T>>(32, init_scheme));
        encoders_.push_back(std::make_unique<ConvCompressBlock<T>>(64, init_scheme));
        encoders_.push_back(std::make_unique<ConvCompressBlock<T>>(128, init_scheme));
        encoders_.push_back(std::make_unique<ConvCompressBlock<T>>(256, init_scheme));
        encoders_.push_back(std::make_unique<ConvCompressBlock<T>>(512, init_scheme));
        encoders_.push_back(std::make_unique<ConvCompressBlock<T>>(1024, init_scheme));
        encoders_.push_back(std::make_unique<ConvCompressBlock<T>>(2048, init_scheme));

        // 3. Decoders (7 blocks)
        decoders_.push_back(std::make_unique<ConvExpandBlock<T>>(4096, init_scheme));
        decoders_.push_back(std::make_unique<ConvExpandBlock<T>>(2048, init_scheme));
        decoders_.push_back(std::make_unique<ConvExpandBlock<T>>(1024, init_scheme));
        decoders_.push_back(std::make_unique<ConvExpandBlock<T>>(512, init_scheme));
        decoders_.push_back(std::make_unique<ConvExpandBlock<T>>(256, init_scheme));
        decoders_.push_back(std::make_unique<ConvExpandBlock<T>>(128, init_scheme));
        decoders_.push_back(std::make_unique<ConvExpandBlock<T>>(64, init_scheme));

        // 4. Head: 32 -> 3
        head_ = std::make_unique<layers::Conv2D<T>>(32, 3, 3, 1, 1);

        // 5. Ghost Projections
        ghosts_.push_back(std::make_unique<ConvGhostProjection<T>>(2048, 2048, init_scheme));
        ghosts_.push_back(std::make_unique<ConvGhostProjection<T>>(1024, 1024, init_scheme));
        ghosts_.push_back(std::make_unique<ConvGhostProjection<T>>(512, 512, init_scheme));
        ghosts_.push_back(std::make_unique<ConvGhostProjection<T>>(256, 256, init_scheme));
        ghosts_.push_back(std::make_unique<ConvGhostProjection<T>>(128, 128, init_scheme));
        ghosts_.push_back(std::make_unique<ConvGhostProjection<T>>(64, 64, init_scheme));
        ghosts_.push_back(std::make_unique<ConvGhostProjection<T>>(32, 32, init_scheme));
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = input;
        x = stem_->forward(x);
        for (auto& enc : encoders_) x = enc->forward(x);
        for (auto& dec : decoders_) x = dec->forward(x);
        x = head_->forward(x);
        return x;
    }

    TrainOutput forward_train(const Tensor<T>& input) {
        TrainOutput out;

        // 1. Encoder Path
        Tensor<T> x_stem = stem_->forward(input);

        std::vector<Tensor<T>> x_encs;
        x_encs.push_back(x_stem);

        Tensor<T> curr = x_stem;
        for (size_t i = 0; i < encoders_.size(); ++i) {
            curr = encoders_[i]->forward(curr);
            if (i < 6) {
                x_encs.push_back(curr);
            }
        }

        // 2. Decoder Path
        Tensor<T> z = curr;
        std::vector<Tensor<T>> ghost_preds;

        for (size_t i = 0; i < decoders_.size(); ++i) {
            z = decoders_[i]->forward(z);

            // Apply Ghost Projection
            Tensor<T> pred = ghosts_[i]->forward(z);
            ghost_preds.push_back(pred);

            out.encoder_targets.push_back(x_encs[x_encs.size() - 1 - i]);
        }
        out.ghost_preds = ghost_preds;

        // 3. Head
        out.reconstruction = head_->forward(z);
        return out;
    }

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

    void backward_train(const Tensor<T>& grad_recon, const std::vector<Tensor<T>>& grad_ghost_preds) {
        Tensor<T> g = grad_recon;
        g = head_->backward(g);

        int idx = static_cast<int>(decoders_.size()) - 1;
        for (auto it = decoders_.rbegin(); it != decoders_.rend(); ++it, --idx) {
            Tensor<T> g_ghost = ghosts_[idx]->backward(grad_ghost_preds[idx]);

            T* g_ptr = g.data();
            const T* gg_ptr = g_ghost.data();
            #pragma omp parallel for
            for(size_t k=0; k<g.size(); ++k) g_ptr[k] += gg_ptr[k];

            g = (*it)->backward(g);
        }

        for (auto it = encoders_.rbegin(); it != encoders_.rend(); ++it) {
            g = (*it)->backward(g);
        }
        g = stem_->backward(g);
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto p_stem = stem_->parameters(); params.insert(params.end(), p_stem.begin(), p_stem.end());
        for(auto& enc : encoders_) {
            auto p = enc->parameters(); params.insert(params.end(), p.begin(), p.end());
        }
        for(auto& dec : decoders_) {
            auto p = dec->parameters(); params.insert(params.end(), p.begin(), p.end());
        }
        auto p_head = head_->parameters(); params.insert(params.end(), p_head.begin(), p_head.end());

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

    std::string name() const override { return "ConvGhostAE"; }

private:
    std::unique_ptr<layers::Conv2D<T>> stem_;
    std::vector<std::unique_ptr<ConvCompressBlock<T>>> encoders_;
    std::vector<std::unique_ptr<ConvExpandBlock<T>>> decoders_;
    std::unique_ptr<layers::Conv2D<T>> head_;
    std::vector<std::unique_ptr<ConvGhostProjection<T>>> ghosts_;
};

} // namespace models
} // namespace dreidel
