#ifndef DREIDEL_MODELS_ZENITHTAESD_HPP
#define DREIDEL_MODELS_ZENITHTAESD_HPP

#include "../layers/Layer.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/ZenithBlock.hpp"
#include "../layers/PixelShuffle.hpp"
#include "../layers/LearnedBlur.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

namespace dreidel {
namespace models {

template <typename T>
class ZenithUpsampleBlock : public layers::Layer<T> {
public:
    ZenithUpsampleBlock(size_t in_channels, size_t out_channels) {
        // Step 1: Zenith Channel Expansion (The "Mixer")
        // Expands channels to (out_channels * 4) to support PixelShuffle(scale=2).
        size_t expanded_channels = out_channels * 4;

        // Use Identity Init for stability in early training
        expansion_ = std::make_unique<layers::ZenithBlock<T>>(
            in_channels, expanded_channels, 3, expanded_channels,
            true, false, false, 1, 1, "identity", false, false
        );

        // Step 2: Pixel Shuffle (The "Mover")
        pixel_shuffle_ = std::make_unique<layers::PixelShuffle<T>>(2);

        // Step 3: Learned Blur (The "Polisher")
        blur_ = std::make_unique<layers::LearnedBlur<T>>(out_channels);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = expansion_->forward(input);
        x = pixel_shuffle_->forward(x);
        x = blur_->forward(x);
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> grad = blur_->backward(grad_output);
        grad = pixel_shuffle_->backward(grad);
        grad = expansion_->backward(grad);
        return grad;
    }

    std::vector<Tensor<T>*> parameters() override {
        auto p1 = expansion_->parameters();
        auto p2 = blur_->parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }

    std::vector<Tensor<T>*> gradients() override {
        auto g1 = expansion_->gradients();
        auto g2 = blur_->gradients();
        g1.insert(g1.end(), g2.begin(), g2.end());
        return g1;
    }

    std::string name() const override { return "ZenithUpsampleBlock"; }

    void set_training(bool training) override {
        expansion_->set_training(training);
        blur_->set_training(training);
    }

private:
    std::unique_ptr<layers::ZenithBlock<T>> expansion_;
    std::unique_ptr<layers::PixelShuffle<T>> pixel_shuffle_;
    std::unique_ptr<layers::LearnedBlur<T>> blur_;
};

template <typename T>
class ZenithTAESD : public layers::Layer<T> {
public:
    ZenithTAESD() {
        // 1. Initial Conv: Maps 4 channels (Latent) to 64 channels.
        conv_in_ = std::make_unique<layers::Conv2D<T>>(4, 64, 3, 1, 1);

        // 2. Zenith Upsample Blocks
        // Block 1: 64x64 -> 128x128
        block1_ = std::make_unique<ZenithUpsampleBlock<T>>(64, 64);
        // Block 2: 128x128 -> 256x256
        block2_ = std::make_unique<ZenithUpsampleBlock<T>>(64, 64);
        // Block 3: 256x256 -> 512x512
        block3_ = std::make_unique<ZenithUpsampleBlock<T>>(64, 64);

        // 3. Output Conv to RGB (3 channels)
        conv_out_ = std::make_unique<layers::Conv2D<T>>(64, 3, 3, 1, 1);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = conv_in_->forward(input);
        x = block1_->forward(x);
        x = block2_->forward(x);
        x = block3_->forward(x);
        x = conv_out_->forward(x);
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> grad = conv_out_->backward(grad_output);
        grad = block3_->backward(grad);
        grad = block2_->backward(grad);
        grad = block1_->backward(grad);
        grad = conv_in_->backward(grad);
        return grad;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto p_in = conv_in_->parameters(); params.insert(params.end(), p_in.begin(), p_in.end());
        auto p1 = block1_->parameters(); params.insert(params.end(), p1.begin(), p1.end());
        auto p2 = block2_->parameters(); params.insert(params.end(), p2.begin(), p2.end());
        auto p3 = block3_->parameters(); params.insert(params.end(), p3.begin(), p3.end());
        auto p_out = conv_out_->parameters(); params.insert(params.end(), p_out.begin(), p_out.end());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto g_in = conv_in_->gradients(); grads.insert(grads.end(), g_in.begin(), g_in.end());
        auto g1 = block1_->gradients(); grads.insert(grads.end(), g1.begin(), g1.end());
        auto g2 = block2_->gradients(); grads.insert(grads.end(), g2.begin(), g2.end());
        auto g3 = block3_->gradients(); grads.insert(grads.end(), g3.begin(), g3.end());
        auto g_out = conv_out_->gradients(); grads.insert(grads.end(), g_out.begin(), g_out.end());
        return grads;
    }

    std::string name() const override { return "ZenithTAESD"; }

    void set_training(bool training) override {
        conv_in_->set_training(training);
        block1_->set_training(training);
        block2_->set_training(training);
        block3_->set_training(training);
        conv_out_->set_training(training);
    }

private:
    std::unique_ptr<layers::Conv2D<T>> conv_in_;
    std::unique_ptr<ZenithUpsampleBlock<T>> block1_;
    std::unique_ptr<ZenithUpsampleBlock<T>> block2_;
    std::unique_ptr<ZenithUpsampleBlock<T>> block3_;
    std::unique_ptr<layers::Conv2D<T>> conv_out_;
};

} // namespace models
} // namespace dreidel

#endif
