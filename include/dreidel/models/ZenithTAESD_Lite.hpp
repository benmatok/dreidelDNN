#pragma once

#include "../layers/Layer.hpp"
#include "../core/Tensor.hpp"
#include "../layers/ZenithLiteBlock.hpp"
#include "../layers/OptimizedConv2D.hpp"
#include "../layers/PixelShuffle.hpp"
#include <vector>
#include <memory>
#include <string>

namespace dreidel {
namespace models {

template <typename T>
class ZenithTAESD_Lite : public layers::Layer<T> {
public:
    // Lite Architecture: Replaces 3x3 Convs with 1x1 Convs and PixelShuffle/Unshuffle
    // Relying on ZenithLiteBlock for spatial mixing.

    ZenithTAESD_Lite(size_t in_channels = 3, size_t latent_channels = 4, size_t base_channels = 64, size_t height = 512, size_t width = 512)
        : H_(height), W_(width)
    {
        // --- Encoder ---
        // Stem: 3 -> 64. 1x1 Conv.
        enc_stem_ = std::make_unique<layers::OptimizedConv2D<T>>(in_channels, base_channels, 1, 1, 0);

        // Stage 1: Res 512.
        enc_block1_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels, H_, W_);

        // Down 1: 512 -> 256. PixelUnshuffle(2).
        // Input: 64ch, 512x512.
        // Unshuffle: 256ch, 256x256.
        // Conv1x1: 256 -> 128 (base*2).
        enc_down1_unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(2);
        enc_down1_conv_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*4, base_channels*2, 1, 1, 0);

        // Stage 2: Res 256.
        enc_block2_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*2, H_/2, W_/2);

        // Down 2: 256 -> 128. PixelUnshuffle(2).
        // Input: 128ch. Unshuffle -> 512ch.
        // Conv1x1: 512 -> 256 (base*4).
        enc_down2_unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(2);
        enc_down2_conv_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*8, base_channels*4, 1, 1, 0);

        // Stage 3: Res 128.
        enc_block3_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*4, H_/4, W_/4);

        // Final Proj: 256 -> Latent (4).
        // Downsample to 1/8th? (64x64).
        // We are at 128x128 (1/4). Need one more down.
        // Unshuffle: 128ch -> 512ch. 64x64.
        // Conv1x1: 512ch -> 4ch.
        // But 256 (base*4) -> 4 * 256 = 1024ch after unshuffle.
        // Input to stage 3 is 256ch.
        // Unshuffle -> 1024ch.
        // Conv1x1: 1024 -> 4.
        enc_out_unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(2);
        enc_out_conv_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*16, latent_channels, 1, 1, 0);

        // --- Decoder ---
        // Input: Latent (4ch). 64x64.
        // Project to Base*4 (256ch)? No, we need to match Encoder reverse.
        // Encoder out: 4ch.
        // Decoder in: 4ch -> 1024ch (for Shuffle)? Or Conv first?
        // Usually Conv -> Shuffle.
        // We want to reach 128x128, 256ch.
        // Target: 256ch.
        // Shuffle(2) takes C*4 -> C.
        // So we need 256*4 = 1024ch input to shuffle.
        // Input 4ch -> 1024ch via 1x1.
        dec_in_conv_ = std::make_unique<layers::OptimizedConv2D<T>>(latent_channels, base_channels*16, 1, 1, 0);
        dec_in_shuffle_ = std::make_unique<layers::PixelShuffle<T>>(2);

        // Stage 1: Res 128.
        dec_block1_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*4, H_/4, W_/4);

        // Up 1: 128 -> 256.
        // Input 256ch. Target 128ch.
        // Shuffle input needed: 128*4 = 512ch.
        // Conv: 256 -> 512.
        dec_up1_conv_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*4, base_channels*8, 1, 1, 0);
        dec_up1_shuffle_ = std::make_unique<layers::PixelShuffle<T>>(2);

        // Stage 2: Res 256.
        dec_block2_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*2, H_/2, W_/2);

        // Up 2: 256 -> 512.
        // Input 128ch. Target 64ch.
        // Shuffle input: 64*4 = 256ch.
        // Conv: 128 -> 256.
        dec_up2_conv_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*2, base_channels*4, 1, 1, 0);
        dec_up2_shuffle_ = std::make_unique<layers::PixelShuffle<T>>(2);

        // Stage 3: Res 512.
        dec_block3_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels, H_, W_);

        // Output: 64ch -> 3ch.
        // 1x1 Conv.
        dec_out_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels, in_channels, 1, 1, 0);

    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Encoder
        Tensor<T> x = enc_stem_->forward(input);
        x = enc_block1_->forward(x);
        x = enc_down1_unshuffle_->forward(x);
        x = enc_down1_conv_->forward(x);
        x = enc_block2_->forward(x);
        x = enc_down2_unshuffle_->forward(x);
        x = enc_down2_conv_->forward(x);
        x = enc_block3_->forward(x);
        x = enc_out_unshuffle_->forward(x);
        x = enc_out_conv_->forward(x);

        // Decoder
        x = dec_in_conv_->forward(x);
        x = dec_in_shuffle_->forward(x);
        x = dec_block1_->forward(x);
        x = dec_up1_conv_->forward(x);
        x = dec_up1_shuffle_->forward(x);
        x = dec_block2_->forward(x);
        x = dec_up2_conv_->forward(x);
        x = dec_up2_shuffle_->forward(x);
        x = dec_block3_->forward(x);
        x = dec_out_->forward(x);

        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        return grad_output;
    }

    std::vector<Tensor<T>*> parameters() override {
        // Collect params...
        // Simplified for now
        return {};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {};
    }

    std::string name() const override { return "ZenithTAESD_Lite"; }

private:
    size_t H_, W_;

    // Encoder
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_stem_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> enc_block1_;
    std::unique_ptr<layers::PixelUnshuffle<T>> enc_down1_unshuffle_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_down1_conv_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> enc_block2_;
    std::unique_ptr<layers::PixelUnshuffle<T>> enc_down2_unshuffle_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_down2_conv_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> enc_block3_;
    std::unique_ptr<layers::PixelUnshuffle<T>> enc_out_unshuffle_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_out_conv_;

    // Decoder
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_in_conv_;
    std::unique_ptr<layers::PixelShuffle<T>> dec_in_shuffle_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> dec_block1_;
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_up1_conv_;
    std::unique_ptr<layers::PixelShuffle<T>> dec_up1_shuffle_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> dec_block2_;
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_up2_conv_;
    std::unique_ptr<layers::PixelShuffle<T>> dec_up2_shuffle_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> dec_block3_;
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_out_;
};

} // namespace models
} // namespace dreidel
