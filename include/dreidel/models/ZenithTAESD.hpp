#pragma once

#include "../layers/Layer.hpp"
#include "../core/Tensor.hpp"
#include "../layers/ZenithLiteBlock.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/OptimizedConv2D.hpp"
#include "../layers/Upscale2D.hpp"
#include <vector>
#include <memory>
#include <string>

namespace dreidel {
namespace models {

template <typename T>
class ZenithTAESD : public layers::Layer<T> {
public:
    // Spec:
    // Encoder: Conv3x3 (Stem) -> ZenithLite -> Down -> ZenithLite -> Down -> ZenithLite -> Conv (Out)
    // Decoder: Conv (In) -> ZenithLite -> Up -> ZenithLite -> Up -> ZenithLite -> Conv (Out)

    ZenithTAESD(size_t in_channels = 3, size_t latent_channels = 4, size_t base_channels = 64, size_t height = 512, size_t width = 512)
        : in_channels_(in_channels), latent_channels_(latent_channels), base_channels_(base_channels), H_(height), W_(width)
    {
        // --- Encoder ---
        // Stem: 3 -> 64. Kernel 3. Stride 1.
        enc_stem_ = std::make_unique<layers::OptimizedConv2D<T>>(in_channels, base_channels, 3, 1, 1);

        // Stage 1: Res 512.
        enc_block1_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels, H_, W_);

        // Down 1: 512 -> 256. Stride 2.
        enc_down1_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels, base_channels*2, 3, 2, 1);

        // Stage 2: Res 256.
        enc_block2_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*2, H_/2, W_/2);

        // Down 2: 256 -> 128. Stride 2.
        enc_down2_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*2, base_channels*4, 3, 2, 1);

        // Stage 3: Res 128.
        enc_block3_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*4, H_/4, W_/4);

        // Final Proj: 256 -> 4.
        enc_out_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*4, latent_channels, 3, 2, 1);

        // --- Decoder ---
        // Input: Latent (4ch). H/8, W/8.
        dec_in_ = std::make_unique<layers::OptimizedConv2D<T>>(latent_channels, base_channels*4, 1, 1, 0); // 1x1 proj

        // Stage 1: Res 1/8.
        dec_block1_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*4, H_/8, W_/8);

        // Up 1: 1/8 -> 1/4.
        dec_up1_ = std::make_unique<layers::Upscale2D<T>>(2);
        dec_conv_up1_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*4, base_channels*2, 3, 1, 1);

        // Stage 2: Res 1/4.
        dec_block2_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*2, H_/4, W_/4);

        // Up 2: 1/4 -> 1/2.
        dec_up2_ = std::make_unique<layers::Upscale2D<T>>(2);
        dec_conv_up2_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels*2, base_channels, 3, 1, 1);

        // Stage 3: Res 1/2.
        dec_block3_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels, H_/2, W_/2);

        // We need to get to H, W (1x).
        // Spec: "ZenithLite -> Up -> ZenithLite -> Up -> ZenithLite -> Conv3x3 (Output)".
        // After block3 (at 1/2), we need another Up.
        dec_up3_ = std::make_unique<layers::Upscale2D<T>>(2); // 1/2 -> 1.
        dec_out_ = std::make_unique<layers::OptimizedConv2D<T>>(base_channels, in_channels, 3, 1, 1);

    }

    Tensor<T> encode(const Tensor<T>& input) {
        Tensor<T> x = enc_stem_->forward(input);
        x = enc_block1_->forward(x);
        x = enc_down1_->forward(x);
        x = enc_block2_->forward(x);
        x = enc_down2_->forward(x);
        x = enc_block3_->forward(x);
        x = enc_out_->forward(x); // 1/8
        return x;
    }

    Tensor<T> decode(const Tensor<T>& latent) {
        Tensor<T> x = dec_in_->forward(latent);
        x = dec_block1_->forward(x);
        x = dec_up1_->forward(x);
        x = dec_conv_up1_->forward(x);
        x = dec_block2_->forward(x);
        x = dec_up2_->forward(x);
        x = dec_conv_up2_->forward(x);
        x = dec_block3_->forward(x);
        x = dec_up3_->forward(x);
        x = dec_out_->forward(x);
        return x;
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        return decode(encode(input));
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Not implemented
        return grad_output;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params;
        auto append = [&](auto& layer) {
            auto p = layer->parameters();
            params.insert(params.end(), p.begin(), p.end());
        };
        append(enc_stem_); append(enc_block1_); append(enc_down1_);
        append(enc_block2_); append(enc_down2_); append(enc_block3_); append(enc_out_);

        append(dec_in_); append(dec_block1_); append(dec_conv_up1_);
        append(dec_block2_); append(dec_conv_up2_); append(dec_block3_); append(dec_out_);
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        return {};
    }

    std::string name() const override { return "ZenithTAESD"; }

    // Weight loading helpers could be added here

    // Accessors for layers (for loading)
    layers::ZenithLiteBlock<T>* get_enc_block(int i) {
        if (i==0) return enc_block1_.get();
        if (i==1) return enc_block2_.get();
        if (i==2) return enc_block3_.get();
        return nullptr;
    }
    layers::ZenithLiteBlock<T>* get_dec_block(int i) {
        if (i==0) return dec_block1_.get();
        if (i==1) return dec_block2_.get();
        if (i==2) return dec_block3_.get();
        return nullptr;
    }
    // ... similarly for convs if needed.

private:
    size_t in_channels_;
    size_t latent_channels_;
    size_t base_channels_;
    size_t H_, W_;

    // Encoder
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_stem_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> enc_block1_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_down1_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> enc_block2_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_down2_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> enc_block3_;
    std::unique_ptr<layers::OptimizedConv2D<T>> enc_out_;

    // Decoder
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_in_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> dec_block1_;
    std::unique_ptr<layers::Upscale2D<T>> dec_up1_;
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_conv_up1_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> dec_block2_;
    std::unique_ptr<layers::Upscale2D<T>> dec_up2_;
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_conv_up2_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> dec_block3_;
    std::unique_ptr<layers::Upscale2D<T>> dec_up3_;
    std::unique_ptr<layers::OptimizedConv2D<T>> dec_out_;
};

} // namespace models
} // namespace dreidel
