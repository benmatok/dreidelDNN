#ifndef DREIDEL_MODELS_CONV_BASELINE_AE_HPP
#define DREIDEL_MODELS_CONV_BASELINE_AE_HPP

#include "../layers/Layer.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/ConvTranspose2D.hpp"
#include "../layers/ReLU.hpp"
#include "../core/Tensor.hpp"
#include <vector>
#include <memory>

namespace dreidel {
namespace models {

template <typename T>
class ConvBaselineAE : public layers::Layer<T> {
public:
    // Comparison Baseline
    // Stem: Conv2d (2x2, s=2) C=32
    // Stage 1: Conv2d (4x4, s=4) C=32 -> C=512 (to match PixelUnshuffle expansion 32*16)
    // Stage 2: Conv2d (4x4, s=4) C=512 -> C=8192
    // Decoder: ConvTranspose2d mirroring above.

    ConvBaselineAE(size_t input_channels = 3, size_t base_channels = 32) {
        stem_ = std::make_unique<layers::Conv2D<T>>(input_channels, base_channels, 2, 2);

        // Match channel widths exactly
        size_t s1_channels = base_channels * 16;
        stage1_conv_ = std::make_unique<layers::Conv2D<T>>(base_channels, s1_channels, 4, 4);

        size_t s2_channels = s1_channels * 16;
        stage2_conv_ = std::make_unique<layers::Conv2D<T>>(s1_channels, s2_channels, 4, 4);

        // Decoder
        stage2_deconv_ = std::make_unique<layers::ConvTranspose2D<T>>(s2_channels, s1_channels, 4, 4);
        stage1_deconv_ = std::make_unique<layers::ConvTranspose2D<T>>(s1_channels, base_channels, 4, 4);

        // Final upsample to match Stem
        final_deconv_ = std::make_unique<layers::ConvTranspose2D<T>>(base_channels, 3, 2, 2);

        // Standard activation
        relu_ = std::make_unique<layers::ReLU<T>>();
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = stem_->forward(input);
        x = relu_->forward(x);

        x = stage1_conv_->forward(x);
        x = relu_->forward(x);

        x = stage2_conv_->forward(x);
        x = relu_->forward(x);

        // Latent Space

        x = stage2_deconv_->forward(x);
        x = relu_->forward(x);

        x = stage1_deconv_->forward(x);
        x = relu_->forward(x);

        x = final_deconv_->forward(x);
        // No final activation usually for image reconstruction (or Sigmoid/Tanh depending on norm)

        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Not strictly needed for forward benchmark, but good for completeness
        Tensor<T> dx = final_deconv_->backward(grad_output);
        dx = relu_->backward(dx);
        dx = stage1_deconv_->backward(dx);
        dx = relu_->backward(dx);
        dx = stage2_deconv_->backward(dx);
        dx = relu_->backward(dx);

        dx = stage2_conv_->backward(dx);
        dx = relu_->backward(dx);
        dx = stage1_conv_->backward(dx);
        dx = relu_->backward(dx);
        dx = stem_->backward(dx);

        return dx;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto append = [&](layers::Layer<T>* l) {
             auto p = l->parameters();
             params.insert(params.end(), p.begin(), p.end());
        };
        append(stem_.get());
        append(stage1_conv_.get());
        append(stage2_conv_.get());
        append(stage2_deconv_.get());
        append(stage1_deconv_.get());
        append(final_deconv_.get());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto append = [&](layers::Layer<T>* l) {
             auto g = l->gradients();
             grads.insert(grads.end(), g.begin(), g.end());
        };
        append(stem_.get());
        append(stage1_conv_.get());
        append(stage2_conv_.get());
        append(stage2_deconv_.get());
        append(stage1_deconv_.get());
        append(final_deconv_.get());
        return grads;
    }

    std::string name() const override { return "ConvBaselineAE"; }

private:
    std::unique_ptr<layers::Conv2D<T>> stem_;
    std::unique_ptr<layers::Conv2D<T>> stage1_conv_;
    std::unique_ptr<layers::Conv2D<T>> stage2_conv_;

    std::unique_ptr<layers::ConvTranspose2D<T>> stage2_deconv_;
    std::unique_ptr<layers::ConvTranspose2D<T>> stage1_deconv_;
    std::unique_ptr<layers::ConvTranspose2D<T>> final_deconv_;

    std::unique_ptr<layers::ReLU<T>> relu_;
};

} // namespace models
} // namespace dreidel

#endif
