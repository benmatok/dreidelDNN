#pragma once

#include "../layers/Layer.hpp"
#include "../layers/PixelShuffle.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/GroupNorm.hpp"
#include "../layers/GELU.hpp"
#include "../layers/ZenithBlock.hpp" // For Spectral Mixer part
#include <vector>
#include <memory>

namespace dreidel {
namespace models {

template <typename T>
class ZenithCompressBlock : public layers::Layer<T> {
public:
    ZenithCompressBlock(size_t in_channels, const std::string& init_scheme = "he") {
        size_t expanded_dim = in_channels * 4;
        size_t out_channels = in_channels * 2;

        // 1. Physical Transmutation (Space -> Depth)
        // [B, C, H, W] -> [B, 4C, H/2, W/2]
        unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(2);

        // 2. PixelMix Group Conv (Local Fidelity)
        // groups=in_channels ensures each kernel sees exactly 4 channels (expanded_dim / in_channels = 4)
        pixel_mix_ = std::make_unique<layers::Conv2D<T>>(expanded_dim, expanded_dim, 3, 1, 1, in_channels);
        norm_ = std::make_unique<layers::GroupNorm<T>>(32, expanded_dim);
        act_ = std::make_unique<layers::GELU<T>>();

        // 3. Spectral Mixer (Global Context)
        // Reuse ZenithBlock but focusing on mixing.
        // We use ZenithBlock(expanded_dim, 1, expanded_dim) -> 1x1 kernel size, so it's just mixing
        // args: in, out, k, spec_dim, ifwht, dilated, gating, stride, upscale, init, slm, seq, eps
        spectral_mixer_ = std::make_unique<layers::ZenithBlock<T>>(expanded_dim, expanded_dim, 1, expanded_dim, true, false, false, 1, 1, init_scheme, false, false, 1.0f);

        // 4. Distillation (Compression)
        // 1x1 Conv to reduce 4C -> 2C
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
        // Residual add
        // Need operator+ for Tensor
        float* x_ptr = x.data();
        const float* res_ptr = res.data();
        #pragma omp parallel for
        for(size_t i=0; i<x.size(); ++i) x_ptr[i] += res_ptr[i];

        // 3. Global Context (Residual)
        res = x;
        x = spectral_mixer_->forward(x);
        x_ptr = x.data();
        res_ptr = res.data();
        #pragma omp parallel for
        for(size_t i=0; i<x.size(); ++i) x_ptr[i] += res_ptr[i];

        // 4. Compress
        x = compressor_->forward(x);
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // 4. Compress Backward
        Tensor<T> g = compressor_->backward(grad_output);

        // 3. Global Context Backward (Residual)
        // y = x + f(x) -> dy = g.  dx = g + df_bw(g)
        Tensor<T> g_res = g; // Branch for residual connection
        Tensor<T> g_spectral = spectral_mixer_->backward(g);

        // Sum gradients: g_res + g_spectral
        float* g_ptr = g_spectral.data();
        const float* gr_ptr = g_res.data();
        #pragma omp parallel for
        for(size_t i=0; i<g_spectral.size(); ++i) g_ptr[i] += gr_ptr[i];

        g = g_spectral;

        // 2. Local Fidelity Backward (Residual)
        g_res = g;
        Tensor<T> g_act = act_->backward(g);
        Tensor<T> g_norm = norm_->backward(g_act);
        Tensor<T> g_pixel = pixel_mix_->backward(g_norm);

        g_ptr = g_pixel.data();
        gr_ptr = g_res.data();
        #pragma omp parallel for
        for(size_t i=0; i<g_pixel.size(); ++i) g_ptr[i] += gr_ptr[i];

        g = g_pixel;

        // 1. Physics Backward
        g = unshuffle_->backward(g);
        return g;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto p1 = pixel_mix_->parameters(); params.insert(params.end(), p1.begin(), p1.end());
        auto p2 = norm_->parameters(); params.insert(params.end(), p2.begin(), p2.end());
        auto p3 = spectral_mixer_->parameters(); params.insert(params.end(), p3.begin(), p3.end());
        auto p4 = compressor_->parameters(); params.insert(params.end(), p4.begin(), p4.end());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto g1 = pixel_mix_->gradients(); grads.insert(grads.end(), g1.begin(), g1.end());
        auto g2 = norm_->gradients(); grads.insert(grads.end(), g2.begin(), g2.end());
        auto g3 = spectral_mixer_->gradients(); grads.insert(grads.end(), g3.begin(), g3.end());
        auto g4 = compressor_->gradients(); grads.insert(grads.end(), g4.begin(), g4.end());
        return grads;
    }

    void set_training(bool training) override {
        spectral_mixer_->set_training(training);
    }

    std::string name() const override { return "ZenithCompressBlock"; }

private:
    std::unique_ptr<layers::PixelUnshuffle<T>> unshuffle_;
    std::unique_ptr<layers::Conv2D<T>> pixel_mix_;
    std::unique_ptr<layers::GroupNorm<T>> norm_;
    std::unique_ptr<layers::GELU<T>> act_;
    std::unique_ptr<layers::ZenithBlock<T>> spectral_mixer_;
    std::unique_ptr<layers::Conv2D<T>> compressor_;
};

template <typename T>
class ZenithExpandBlock : public layers::Layer<T> {
public:
    ZenithExpandBlock(size_t in_channels, const std::string& init_scheme = "he") {
        // Reverse of Compress
        // Input: 2C -> Expand to 4C -> Spectral -> PixelMix -> Shuffle to C
        size_t expanded_dim = in_channels * 2; // Input is "2C" relative to original block, output should be "C"
        // Wait, logic:
        // Compress: In(C) -> Unshuf(4C) -> ... -> Out(2C)
        // Expand: In(2C) -> Expand(4C) -> ... -> Shuf(C)

        size_t out_channels = in_channels / 2; // If in_channels is the "2C"
        // But let's define constructor by input channels.
        // If Compress output 64, Expand input 64.
        // Expand(64) -> 128 -> Shuffle -> 32.

        // 1. Expansion
        expander_ = std::make_unique<layers::Conv2D<T>>(in_channels, in_channels * 2, 1);
        expanded_dim = in_channels * 2;

        // 2. Global Context
        spectral_mixer_ = std::make_unique<layers::ZenithBlock<T>>(expanded_dim, expanded_dim, 1, expanded_dim, true, false, false, 1, 1, init_scheme, false, false, 1.0f);

        // 3. Local Fidelity
        // groups = out_channels (which is expanded_dim / 4).
        // Wait, shuffle goes 4C -> C.
        // So output of PixelMix is expanded_dim.
        // If expanded_dim = 128. Shuffle(2) -> 32 channels.
        // Groups should be 32. 128/32 = 4.
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

        // 2. Global
        Tensor<T> res = x;
        x = spectral_mixer_->forward(x);
        float* x_ptr = x.data();
        const float* res_ptr = res.data();
        #pragma omp parallel for
        for(size_t i=0; i<x.size(); ++i) x_ptr[i] += res_ptr[i];

        // 3. Local
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
        Tensor<T> g_spectral = spectral_mixer_->backward(g);
        g_ptr = g_spectral.data();
        gr_ptr = g_res.data();
        #pragma omp parallel for
        for(size_t i=0; i<g_spectral.size(); ++i) g_ptr[i] += gr_ptr[i];
        g = g_spectral;

        // 1. Expand
        g = expander_->backward(g);
        return g;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto p1 = expander_->parameters(); params.insert(params.end(), p1.begin(), p1.end());
        auto p2 = spectral_mixer_->parameters(); params.insert(params.end(), p2.begin(), p2.end());
        auto p3 = pixel_mix_->parameters(); params.insert(params.end(), p3.begin(), p3.end());
        auto p4 = norm_->parameters(); params.insert(params.end(), p4.begin(), p4.end());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto g1 = expander_->gradients(); grads.insert(grads.end(), g1.begin(), g1.end());
        auto g2 = spectral_mixer_->gradients(); grads.insert(grads.end(), g2.begin(), g2.end());
        auto g3 = pixel_mix_->gradients(); grads.insert(grads.end(), g3.begin(), g3.end());
        auto g4 = norm_->gradients(); grads.insert(grads.end(), g4.begin(), g4.end());
        return grads;
    }

    void set_training(bool training) override {
        spectral_mixer_->set_training(training);
    }

    std::string name() const override { return "ZenithExpandBlock"; }

private:
    std::unique_ptr<layers::Conv2D<T>> expander_;
    std::unique_ptr<layers::ZenithBlock<T>> spectral_mixer_;
    std::unique_ptr<layers::Conv2D<T>> pixel_mix_;
    std::unique_ptr<layers::GroupNorm<T>> norm_;
    std::unique_ptr<layers::GELU<T>> act_;
    std::unique_ptr<layers::PixelShuffle<T>> shuffle_;
};

template <typename T>
class ZenithOverhaulAE : public layers::Layer<T> {
public:
    ZenithOverhaulAE(const std::string& init_scheme = "he") {
        // Encoder
        // Stem: 3 -> 32
        layers_.push_back(std::make_unique<layers::Conv2D<T>>(3, 32, 3, 1, 1));

        // Enc1: 32 -> 64
        layers_.push_back(std::make_unique<ZenithCompressBlock<T>>(32, init_scheme));
        // Enc2: 64 -> 128
        layers_.push_back(std::make_unique<ZenithCompressBlock<T>>(64, init_scheme));
        // Enc3: 128 -> 256
        layers_.push_back(std::make_unique<ZenithCompressBlock<T>>(128, init_scheme));
        // Enc4: 256 -> 512
        layers_.push_back(std::make_unique<ZenithCompressBlock<T>>(256, init_scheme));
        // Enc5: 512 -> 1024
        layers_.push_back(std::make_unique<ZenithCompressBlock<T>>(512, init_scheme));
        // Enc6: 1024 -> 2048
        layers_.push_back(std::make_unique<ZenithCompressBlock<T>>(1024, init_scheme));
        // Enc7: 2048 -> 4096 (1x1)
        layers_.push_back(std::make_unique<ZenithCompressBlock<T>>(2048, init_scheme));

        // Decoder (Inverted)
        // Dec7: 4096 -> 2048
        layers_.push_back(std::make_unique<ZenithExpandBlock<T>>(4096, init_scheme));
        // Dec6: 2048 -> 1024
        layers_.push_back(std::make_unique<ZenithExpandBlock<T>>(2048, init_scheme));
        // Dec5: 1024 -> 512
        layers_.push_back(std::make_unique<ZenithExpandBlock<T>>(1024, init_scheme));
        // Dec4: 512 -> 256
        layers_.push_back(std::make_unique<ZenithExpandBlock<T>>(512, init_scheme));
        // Dec3: 256 -> 128
        layers_.push_back(std::make_unique<ZenithExpandBlock<T>>(256, init_scheme));
        // Dec2: 128 -> 64
        layers_.push_back(std::make_unique<ZenithExpandBlock<T>>(128, init_scheme));
        // Dec1: 64 -> 32
        layers_.push_back(std::make_unique<ZenithExpandBlock<T>>(64, init_scheme));

        // Head: 32 -> 3
        layers_.push_back(std::make_unique<layers::Conv2D<T>>(32, 3, 3, 1, 1));
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

    void set_training(bool training) override {
        for(auto& layer : layers_) {
            layer->set_training(training);
        }
    }

    std::string name() const override { return "ZenithOverhaulAE"; }

private:
    std::vector<std::unique_ptr<layers::Layer<T>>> layers_;
};

} // namespace models
} // namespace dreidel
