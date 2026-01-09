#ifndef DREIDEL_MODELS_ZENITH_HIERARCHICAL_AE_HPP
#define DREIDEL_MODELS_ZENITH_HIERARCHICAL_AE_HPP

#include "../layers/Layer.hpp"
#include "../layers/ZenithBlock.hpp"
#include "../layers/ZenithAVXGate.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/PixelShuffle.hpp"
#include "../layers/GroupNorm.hpp"
#include "../core/Tensor.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>
#include <mutex>

namespace dreidel {
namespace models {

template <typename T>
class ZenithHierarchicalAE : public layers::Layer<T> {
public:
    // Stem: Conv2d (2x2, stride 2, C=32)
    // Stage 1: PixelUnshuffle (factor=4), 1x ZenithBlock
    // Stage 2: PixelUnshuffle (factor=4), 2x ZenithBlock, with ZenithAVXGate inputs
    // Head: Mirrors encoder

    ZenithHierarchicalAE(size_t input_channels = 3, size_t base_channels = 32, const std::string& init_scheme = "he", bool use_slm = false)
    {
        // Stem
        stem_ = std::make_unique<layers::Conv2D<T>>(input_channels, base_channels, 2, 2); // k=2, s=2

        // Stage 1
        // Input: H/2, W/2, C=32
        // PixelUnshuffle factor 4 -> H/8, W/8, C = 32 * 16 = 512
        stage1_unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(4);
        stage1_channels_ = base_channels * 16;

        // 1x ZenithBlock
        stage1_block_ = std::make_unique<layers::ZenithBlock<T>>(
            stage1_channels_, stage1_channels_, 3, stage1_channels_, true, false, false, 1, 1, init_scheme, false
        );

        // Stage 2 (Bottleneck)
        // Input: H/8, W/8, C=512
        // PixelUnshuffle factor 4 -> H/32, W/32, C = 512 * 16 = 8192
        stage2_unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(4);
        stage2_channels_ = stage1_channels_ * 16;

        // 2x ZenithBlock with ZenithAVXGate before each mixer (conceptually before the block here)
        // Enable SLM here if requested (Stage 2 Only)

        // Gate for block 1
        stage2_gate1_ = std::make_unique<layers::ZenithAVXGate<T>>(stage2_channels_);
        stage2_block1_ = std::make_unique<layers::ZenithBlock<T>>(
            stage2_channels_, stage2_channels_, 3, stage2_channels_, true, false, false, 1, 1, init_scheme, use_slm
        );

        // Gate for block 2
        stage2_gate2_ = std::make_unique<layers::ZenithAVXGate<T>>(stage2_channels_);
        stage2_block2_ = std::make_unique<layers::ZenithBlock<T>>(
            stage2_channels_, stage2_channels_, 3, stage2_channels_, true, false, false, 1, 1, init_scheme, use_slm
        );

        // Decoder Head
        // Un-Stage 2
        // Input: H/32, W/32, C=8192
        // PixelShuffle factor 4 -> H/8, W/8, C=512
        head_shuffle2_ = std::make_unique<layers::PixelShuffle<T>>(4);

        // Mirror Stage 1
        head_shuffle1_ = std::make_unique<layers::PixelShuffle<T>>(4); // Mirrors Stage 1

        // Mirror Stem
        head_shuffle_stem_ = std::make_unique<layers::PixelShuffle<T>>(2); // Mirrors Stem Stride 2

        // Final Conv: base_channels/4 -> 3. (e.g. 8->2->3 or 32->8->3).
        size_t final_in_channels = base_channels / 4;
        if (final_in_channels == 0) final_in_channels = 1; // Safety

        final_conv_ = std::make_unique<layers::Conv2D<T>>(final_in_channels, 3, 3, 1, 1);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Stem
        Tensor<T> x = stem_->forward(input); // H/2, W/2, C=32

        // Stage 1
        x = stage1_unshuffle_->forward(x); // H/8, W/8, C=512
        x = stage1_block_->forward(x);

        // Stage 2
        x = stage2_unshuffle_->forward(x); // H/32, W/32, C=8192

        // Apply Gate 1 -> Block 1
        x = stage2_gate1_->forward(x);
        stage2_block1_->set_pruning_mask(stage2_gate1_->get_last_mask());
        x = stage2_block1_->forward(x);

        // Apply Gate 2 -> Block 2
        x = stage2_gate2_->forward(x);
        stage2_block2_->set_pruning_mask(stage2_gate2_->get_last_mask());
        x = stage2_block2_->forward(x);

        // Head (Decoder)
        // Mirror Stage 2
        x = head_shuffle2_->forward(x); // H/8, W/8, C=512

        // Mirror Stage 1
        x = head_shuffle1_->forward(x); // H/2, W/2, C=32

        // Mirror Stem
        x = head_shuffle_stem_->forward(x); // H, W, C=8

        x = final_conv_->forward(x); // H, W, C=3

        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Reverse order
        Tensor<T> dx = final_conv_->backward(grad_output);
        dx = head_shuffle_stem_->backward(dx);
        dx = head_shuffle1_->backward(dx);
        dx = head_shuffle2_->backward(dx);

        // Block 2 -> Gate 2
        dx = stage2_block2_->backward(dx);
        dx = stage2_gate2_->backward(dx);

        // Block 1 -> Gate 1
        dx = stage2_block1_->backward(dx);
        dx = stage2_gate1_->backward(dx);

        dx = stage2_unshuffle_->backward(dx);
        dx = stage1_block_->backward(dx);
        dx = stage1_unshuffle_->backward(dx);
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
        append(stage1_block_.get());

        append(stage2_gate1_.get());
        append(stage2_block1_.get());
        append(stage2_gate2_.get());
        append(stage2_block2_.get());

        append(final_conv_.get());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads;
        auto append = [&](layers::Layer<T>* l) {
            auto g = l->gradients();
            grads.insert(grads.end(), g.begin(), g.end());
        };
        append(stem_.get());
        append(stage1_block_.get());

        append(stage2_gate1_.get());
        append(stage2_block1_.get());
        append(stage2_gate2_.get());
        append(stage2_block2_.get());

        append(final_conv_.get());
        return grads;
    }

    void set_gate_training(bool training, float temp = 1.0f) {
        stage2_gate1_->set_training(training);
        stage2_gate1_->set_temperature(temp);
        stage2_gate2_->set_training(training);
        stage2_gate2_->set_temperature(temp);
    }

    // Expose sparsity stats
    T get_sparsity_loss() const {
        return stage2_gate1_->get_sparsity_loss() + stage2_gate2_->get_sparsity_loss();
    }

    std::string name() const override { return "ZenithHierarchicalAE"; }

private:
    std::unique_ptr<layers::Conv2D<T>> stem_;

    std::unique_ptr<layers::PixelUnshuffle<T>> stage1_unshuffle_;
    std::unique_ptr<layers::ZenithBlock<T>> stage1_block_;
    size_t stage1_channels_;

    std::unique_ptr<layers::PixelUnshuffle<T>> stage2_unshuffle_;

    std::unique_ptr<layers::ZenithAVXGate<T>> stage2_gate1_;
    std::unique_ptr<layers::ZenithBlock<T>> stage2_block1_;

    std::unique_ptr<layers::ZenithAVXGate<T>> stage2_gate2_;
    std::unique_ptr<layers::ZenithBlock<T>> stage2_block2_;

    size_t stage2_channels_;

    std::unique_ptr<layers::PixelShuffle<T>> head_shuffle2_;
    std::unique_ptr<layers::PixelShuffle<T>> head_shuffle1_;
    std::unique_ptr<layers::PixelShuffle<T>> head_shuffle_stem_;
    std::unique_ptr<layers::Conv2D<T>> final_conv_;
};

} // namespace models
} // namespace dreidel

#endif
