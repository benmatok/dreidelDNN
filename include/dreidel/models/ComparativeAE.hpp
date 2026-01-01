#ifndef DREIDEL_MODELS_COMPARATIVEAE_HPP
#define DREIDEL_MODELS_COMPARATIVEAE_HPP

#include "../layers/Layer.hpp"
#include "../layers/ZenithBlock.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/ConvTranspose2D.hpp"
#include "../layers/PixelShuffle.hpp" // For PixelUnshuffle
#include "../layers/ReLU.hpp"
#include <vector>
#include <memory>
#include <iostream>

namespace dreidel {
namespace models {

using namespace layers;

// Helper: Residual Block
template <typename T>
class ResidualBlock : public Layer<T> {
public:
    ResidualBlock(size_t channels) {
        // Standard ResNet Block: Conv3x3 -> ReLU -> Conv3x3 -> Add
        // Padding=1 for 3x3 to maintain size
        conv1_ = std::make_shared<Conv2D<T>>(channels, channels, 3, 1, 1);
        relu1_ = std::make_shared<ReLU<T>>();
        conv2_ = std::make_shared<Conv2D<T>>(channels, channels, 3, 1, 1);
        relu2_ = std::make_shared<ReLU<T>>();
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = conv1_->forward(input);
        x = relu1_->forward(x);
        x = conv2_->forward(x);
        // Add residual
        // Tensor + Tensor handles element-wise add
        Tensor<T> out = input + x;
        return relu2_->forward(out);
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // y = relu2(x + F(x))
        // dy = grad_output
        // d(x + F(x)) = dy * relu2'(y)
        // Let's assume ReLU backward modifies in place or we call backward on relu layer

        // This simple implementation might be tricky if layers don't cache inputs perfectly or handle branches.
        // Conv2D etc cache input.
        // But for the add: d(input) = d(out_before_relu) + d(F(x))/dx

        // 1. Backward through last ReLU
        Tensor<T> d_out_sum = relu2_->backward(grad_output);

        // 2. Branch: One path to F(x), one to input directly.
        // Backward through conv2
        Tensor<T> d_conv2 = conv2_->backward(d_out_sum);
        // Backward through relu1
        Tensor<T> d_relu1 = relu1_->backward(d_conv2);
        // Backward through conv1
        Tensor<T> d_conv1 = conv1_->backward(d_relu1);

        // Total gradient = d_out_sum (identity path) + d_conv1 (residual path)
        return d_out_sum + d_conv1;
    }

    std::vector<Tensor<T>*> parameters() override {
        auto p1 = conv1_->parameters();
        auto p2 = conv2_->parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }

    std::vector<Tensor<T>*> gradients() override {
        auto g1 = conv1_->gradients();
        auto g2 = conv2_->gradients();
        g1.insert(g1.end(), g2.begin(), g2.end());
        return g1;
    }

    std::string name() const override { return "ResidualBlock"; }

private:
    std::shared_ptr<Conv2D<T>> conv1_;
    std::shared_ptr<ReLU<T>> relu1_;
    std::shared_ptr<Conv2D<T>> conv2_;
    std::shared_ptr<ReLU<T>> relu2_;
};

// I. Zenith Hierarchical Autoencoder
template <typename T>
class ZenithHierarchicalAE : public Layer<T> {
public:
    ZenithHierarchicalAE(size_t base_filters = 16) {
        // 1. Stem: Conv2d(3, C, k=2, s=2) -> H/2, W/2
        stem_ = std::make_unique<Conv2D<T>>(3, base_filters, 2, 2, 0);

        // 2. Stage 1: PixelUnshuffle(4) -> C*16, H/8
        // Input C, H/2. Output C*16, H/8.
        stage1_unshuffle_ = std::make_unique<PixelUnshuffle<T>>(4);
        // Body: 1x ZenithBlock(C*16)
        // Kernel size? Let's assume 3. Spectral dim? C*16 or something smaller?
        // Let's use spectral_dim = channels for full mixing.
        size_t s1_ch = base_filters * 16;
        stage1_block_ = std::make_unique<ZenithBlock<T>>(s1_ch, 3, s1_ch);

        // 3. Stage 2: PixelUnshuffle(4) -> C*256, H/32
        // Input C*16, H/8. Output (C*16)*16 = C*256, H/32.
        stage2_unshuffle_ = std::make_unique<PixelUnshuffle<T>>(4);
        size_t s2_ch = base_filters * 256;
        // Body: 2x ZenithBlock(C*256)
        stage2_block1_ = std::make_unique<ZenithBlock<T>>(s2_ch, 3, s2_ch);
        stage2_block2_ = std::make_unique<ZenithBlock<T>>(s2_ch, 3, s2_ch);

        // 4. Head: ConvTranspose2d(in=C*256, out=3, k=32, s=32)
        // Maps H/32 -> H.
        head_ = std::make_unique<ConvTranspose2D<T>>(s2_ch, 3, 32, 32, 0);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = stem_->forward(input);

        x = stage1_unshuffle_->forward(x);
        x = stage1_block_->forward(x);

        x = stage2_unshuffle_->forward(x);
        x = stage2_block1_->forward(x);
        x = stage2_block2_->forward(x);

        x = head_->forward(x);
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Not needed for inference benchmark, but required by interface
        return Tensor<T>();
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto append = [&](auto& layer) {
            auto p = layer->parameters();
            params.insert(params.end(), p.begin(), p.end());
        };
        append(stem_);
        append(stage1_block_);
        append(stage2_block1_);
        append(stage2_block2_);
        append(head_);
        return params;
    }

    std::vector<Tensor<T>*> gradients() override { return {}; }
    std::string name() const override { return "ZenithHierarchicalAE"; }

private:
    std::unique_ptr<Conv2D<T>> stem_;
    std::unique_ptr<PixelUnshuffle<T>> stage1_unshuffle_;
    std::unique_ptr<ZenithBlock<T>> stage1_block_;
    std::unique_ptr<PixelUnshuffle<T>> stage2_unshuffle_;
    std::unique_ptr<ZenithBlock<T>> stage2_block1_;
    std::unique_ptr<ZenithBlock<T>> stage2_block2_;
    std::unique_ptr<ConvTranspose2D<T>> head_;
};


// II. Conv Baseline Autoencoder
template <typename T>
class ConvBaselineAE : public Layer<T> {
public:
    ConvBaselineAE(size_t base_filters = 16) {
        // 1. Stem: Conv2d(3, C, k=2, s=2)
        stem_ = std::make_unique<Conv2D<T>>(3, base_filters, 2, 2, 0);

        // 2. Stage 1 Equiv: Conv2d(C, C*16, k=4, s=4) -> H/8
        // Note: PixelUnshuffle(4) reduces H by 4.
        stage1_conv_ = std::make_unique<Conv2D<T>>(base_filters, base_filters * 16, 4, 4, 0);
        stage1_relu_ = std::make_unique<ReLU<T>>();
        stage1_res_ = std::make_unique<ResidualBlock<T>>(base_filters * 16);

        // 3. Stage 2 Equiv: Conv2d(C*16, C*256, k=4, s=4) -> H/32
        stage2_conv_ = std::make_unique<Conv2D<T>>(base_filters * 16, base_filters * 256, 4, 4, 0);
        stage2_relu_ = std::make_unique<ReLU<T>>();
        stage2_res1_ = std::make_unique<ResidualBlock<T>>(base_filters * 256);
        stage2_res2_ = std::make_unique<ResidualBlock<T>>(base_filters * 256);

        // 4. Head
        head_ = std::make_unique<ConvTranspose2D<T>>(base_filters * 256, 3, 32, 32, 0);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> x = stem_->forward(input);

        x = stage1_conv_->forward(x);
        x = stage1_relu_->forward(x);
        x = stage1_res_->forward(x);

        x = stage2_conv_->forward(x);
        x = stage2_relu_->forward(x);
        x = stage2_res1_->forward(x);
        x = stage2_res2_->forward(x);

        x = head_->forward(x);
        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override { return Tensor<T>(); }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto append = [&](auto& layer) {
            auto p = layer->parameters();
            params.insert(params.end(), p.begin(), p.end());
        };
        append(stem_);
        append(stage1_conv_);
        append(stage1_res_);
        append(stage2_conv_);
        append(stage2_res1_);
        append(stage2_res2_);
        append(head_);
        return params;
    }

    std::vector<Tensor<T>*> gradients() override { return {}; }
    std::string name() const override { return "ConvBaselineAE"; }

private:
    std::unique_ptr<Conv2D<T>> stem_;
    std::unique_ptr<Conv2D<T>> stage1_conv_;
    std::unique_ptr<ReLU<T>> stage1_relu_;
    std::unique_ptr<ResidualBlock<T>> stage1_res_;
    std::unique_ptr<Conv2D<T>> stage2_conv_;
    std::unique_ptr<ReLU<T>> stage2_relu_;
    std::unique_ptr<ResidualBlock<T>> stage2_res1_;
    std::unique_ptr<ResidualBlock<T>> stage2_res2_;
    std::unique_ptr<ConvTranspose2D<T>> head_;
};

} // namespace models
} // namespace dreidel

#endif
