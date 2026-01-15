#pragma once

#include "../layers/Layer.hpp"
#include "../core/Tensor.hpp"
#include "../layers/ZenithLiteBlock.hpp"
#include "../layers/DeepSpectralLinear.hpp"
#include "../layers/Bias.hpp"
#include "../layers/PixelShuffle.hpp"
#include <vector>
#include <memory>
#include <string>

namespace dreidel {
namespace models {

template <typename T>
class ZenithTAESD_Spectral : public layers::Layer<T> {
public:
    // Spectral Architecture: Replaces 1x1 Convs (O(C^2)) with DeepSpectralLinear (O(C log C)).
    // Uses PixelShuffle/Unshuffle for spatial changes.
    // Pure Spectral Pipeline.

    ZenithTAESD_Spectral(size_t in_channels = 3, size_t latent_channels = 4, size_t base_channels = 64, size_t height = 512, size_t width = 512)
        : H_(height), W_(width)
    {
        // Helper to creating Spectral 1x1 block (Linear + Bias)
        // Note: DeepSpectralLinear requires power of 2 dimension.
        // Base channels = 64. Fine.
        // Latent = 4. Fine.
        // Input = 3. Not power of 2.
        // So Stem and Final Out must remain standard Conv or padded?
        // DeepSpectralLinear handles padding automatically (cached_input_padded_).
        // But for output size 64?
        // DeepSpectralLinear maps Dim -> Dim. It preserves dimension.
        // It does NOT change channel count.
        // Standard 1x1 Conv changes channels (e.g. 64 -> 128).
        // DeepSpectralLinear is square.
        // How to change channels with Spectral?
        // 1. Pad to max(Cin, Cout). Run Spectral. Slice to Cout.
        // 2. Use multiple heads?
        // Standard "Spectral 1x1" usually implies Square mixing.
        // If we need to change width, we might need a projection.
        // Can we do "Pad -> Spectral -> Slice"?
        // Yes.
        // But if Cout >> Cin (e.g. 3 -> 64), we pad 3 to 64. Run 64x64 spectral. Output 64.
        // If Cin >> Cout (e.g. 64 -> 4), we run 64x64 spectral. Slice to 4.
        // Efficient?
        // 64->128. Pad to 128 (64 zeros). Run 128x128. Output 128.
        // Ops: 128 log 128 * Depth.
        // Dense 1x1: 64*128 = 8192 ops.
        // Spectral (Depth 4): 128 * 7 * 4 = 3584 ops.
        // Speedup ~2.3x.

        // We need a wrapper `SpectralProjection` that handles this resizing logic.
        // Or implement it here.

        auto make_spectral = [&](size_t cin, size_t cout) -> std::pair<std::shared_ptr<layers::Layer<T>>, std::shared_ptr<layers::Bias<T>>> {
            size_t max_dim = std::max(cin, cout);
            // Next power of 2
            size_t p2 = 1;
            while(p2 < max_dim) p2 <<= 1;

            // We use DeepSpectralLinear(p2).
            // But we need to handle input padding/slicing wrapper?
            // DeepSpectralLinear pads input if smaller than dim_.
            // But if input is larger? No, we set dim_ = p2 >= cin.
            // So input <= dim_. Padding works.
            // Output is dim_. We need to slice to cout.
            // But `DeepSpectralLinear` returns `x` (padded size).
            // We need a slicing layer? Or just valid indexing.
            // Let's assume we implement a helper `forward_spectral` that does the slicing.
            // But we need `Layer` objects for `parameters()`.
            // We'll store `DeepSpectralLinear` and `Bias`.
            return {std::make_shared<layers::DeepSpectralLinear<T>>(p2, 4), std::make_shared<layers::Bias<T>>(cout)};
        };

        // --- Encoder ---
        // Stem: 3 -> 64.
        auto [s_stem, b_stem] = make_spectral(in_channels, base_channels);
        enc_stem_ = s_stem; enc_stem_bias_ = b_stem;

        // Block 1: 64 -> 64. ZenithLite.
        enc_block1_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels, H_, W_);

        // Down 1: 64 -> 256 (Unshuffle) -> 128.
        enc_down1_unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(2);
        auto [s_d1, b_d1] = make_spectral(base_channels*4, base_channels*2);
        enc_down1_proj_ = s_d1; enc_down1_bias_ = b_d1;

        // Block 2: 128 -> 128.
        enc_block2_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*2, H_/2, W_/2);

        // Down 2: 128 -> 512 (Unshuffle) -> 256.
        enc_down2_unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(2);
        auto [s_d2, b_d2] = make_spectral(base_channels*8, base_channels*4);
        enc_down2_proj_ = s_d2; enc_down2_bias_ = b_d2;

        // Block 3: 256 -> 256.
        enc_block3_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*4, H_/4, W_/4);

        // Out: 256 -> 1024 (Unshuffle) -> 4.
        enc_out_unshuffle_ = std::make_unique<layers::PixelUnshuffle<T>>(2);
        auto [s_out, b_out] = make_spectral(base_channels*16, latent_channels);
        enc_out_proj_ = s_out; enc_out_bias_ = b_out;

        // --- Decoder ---
        // In: 4 -> 1024.
        auto [s_din, b_din] = make_spectral(latent_channels, base_channels*16);
        dec_in_proj_ = s_din; dec_in_bias_ = b_din;
        dec_in_shuffle_ = std::make_unique<layers::PixelShuffle<T>>(2);

        // Block 1: 256.
        dec_block1_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*4, H_/4, W_/4);

        // Up 1: 256 -> 512 (Proj) -> 128 (Shuffle).
        auto [s_u1, b_u1] = make_spectral(base_channels*4, base_channels*8);
        dec_up1_proj_ = s_u1; dec_up1_bias_ = b_u1;
        dec_up1_shuffle_ = std::make_unique<layers::PixelShuffle<T>>(2);

        // Block 2: 128.
        dec_block2_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels*2, H_/2, W_/2);

        // Up 2: 128 -> 256 (Proj) -> 64 (Shuffle).
        auto [s_u2, b_u2] = make_spectral(base_channels*2, base_channels*4);
        dec_up2_proj_ = s_u2; dec_up2_bias_ = b_u2;
        dec_up2_shuffle_ = std::make_unique<layers::PixelShuffle<T>>(2);

        // Block 3: 64.
        dec_block3_ = std::make_unique<layers::ZenithLiteBlock<T>>(base_channels, H_, W_);

        // Out: 64 -> 3.
        auto [s_dout, b_dout] = make_spectral(base_channels, in_channels);
        dec_out_proj_ = s_dout; dec_out_bias_ = b_dout;
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Helper to run spectral proj
        auto run_spectral = [&](const Tensor<T>& x, auto& proj, auto& bias, size_t cout) {
            Tensor<T> y = proj->forward(x); // Padded output
            // Slice to cout
            if (y.shape().back() > cout) {
                y = y.slice_last_dim(cout);
            }
            // Add bias
            y = bias->forward(y);
            return y;
        };

        // Encoder
        Tensor<T> x = run_spectral(input, enc_stem_, enc_stem_bias_, 64);
        x = enc_block1_->forward(x);

        x = enc_down1_unshuffle_->forward(x);
        x = run_spectral(x, enc_down1_proj_, enc_down1_bias_, 128);

        x = enc_block2_->forward(x);

        x = enc_down2_unshuffle_->forward(x);
        x = run_spectral(x, enc_down2_proj_, enc_down2_bias_, 256);

        x = enc_block3_->forward(x);

        x = enc_out_unshuffle_->forward(x);
        x = run_spectral(x, enc_out_proj_, enc_out_bias_, 4); // Latent

        // Decoder
        x = run_spectral(x, dec_in_proj_, dec_in_bias_, 1024);
        x = dec_in_shuffle_->forward(x);

        x = dec_block1_->forward(x);

        x = run_spectral(x, dec_up1_proj_, dec_up1_bias_, 512);
        x = dec_up1_shuffle_->forward(x);

        x = dec_block2_->forward(x);

        x = run_spectral(x, dec_up2_proj_, dec_up2_bias_, 256);
        x = dec_up2_shuffle_->forward(x);

        x = dec_block3_->forward(x);

        x = run_spectral(x, dec_out_proj_, dec_out_bias_, 3);

        return x;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        return grad_output;
    }

    std::vector<Tensor<T>*> parameters() override {
        // Simplified
        return {};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {};
    }

    std::string name() const override { return "ZenithTAESD_Spectral"; }

private:
    size_t H_, W_;

    // Encoder
    std::shared_ptr<layers::Layer<T>> enc_stem_; std::shared_ptr<layers::Bias<T>> enc_stem_bias_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> enc_block1_;
    std::unique_ptr<layers::PixelUnshuffle<T>> enc_down1_unshuffle_;
    std::shared_ptr<layers::Layer<T>> enc_down1_proj_; std::shared_ptr<layers::Bias<T>> enc_down1_bias_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> enc_block2_;
    std::unique_ptr<layers::PixelUnshuffle<T>> enc_down2_unshuffle_;
    std::shared_ptr<layers::Layer<T>> enc_down2_proj_; std::shared_ptr<layers::Bias<T>> enc_down2_bias_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> enc_block3_;
    std::unique_ptr<layers::PixelUnshuffle<T>> enc_out_unshuffle_;
    std::shared_ptr<layers::Layer<T>> enc_out_proj_; std::shared_ptr<layers::Bias<T>> enc_out_bias_;

    // Decoder
    std::shared_ptr<layers::Layer<T>> dec_in_proj_; std::shared_ptr<layers::Bias<T>> dec_in_bias_;
    std::unique_ptr<layers::PixelShuffle<T>> dec_in_shuffle_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> dec_block1_;
    std::shared_ptr<layers::Layer<T>> dec_up1_proj_; std::shared_ptr<layers::Bias<T>> dec_up1_bias_;
    std::unique_ptr<layers::PixelShuffle<T>> dec_up1_shuffle_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> dec_block2_;
    std::shared_ptr<layers::Layer<T>> dec_up2_proj_; std::shared_ptr<layers::Bias<T>> dec_up2_bias_;
    std::unique_ptr<layers::PixelShuffle<T>> dec_up2_shuffle_;
    std::unique_ptr<layers::ZenithLiteBlock<T>> dec_block3_;
    std::shared_ptr<layers::Layer<T>> dec_out_proj_; std::shared_ptr<layers::Bias<T>> dec_out_bias_;
};

} // namespace models
} // namespace dreidel
