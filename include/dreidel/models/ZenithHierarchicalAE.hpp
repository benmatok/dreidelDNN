#ifndef DREIDEL_MODELS_ZENITH_HIERARCHICAL_AE_HPP
#define DREIDEL_MODELS_ZENITH_HIERARCHICAL_AE_HPP

#include "../layers/Layer.hpp"
#include "../layers/ZenithBlock.hpp"
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
    // Stage 2: PixelUnshuffle (factor=4), Fixed 2D Sinusoidal Embeddings, 2x ZenithBlock
    // Head: Mirrors encoder

    ZenithHierarchicalAE(size_t input_channels = 3, size_t base_channels = 32, bool use_pe = true, const std::string& init_scheme = "he", bool use_slm = false)
        : use_pe_(use_pe)
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

        // 2x ZenithBlock
        // Enable SLM here if requested (Stage 2 Only)
        stage2_block1_ = std::make_unique<layers::ZenithBlock<T>>(
            stage2_channels_, stage2_channels_, 3, stage2_channels_, true, false, false, 1, 1, init_scheme, use_slm
        );
        stage2_block2_ = std::make_unique<layers::ZenithBlock<T>>(
            stage2_channels_, stage2_channels_, 3, stage2_channels_, true, false, false, 1, 1, init_scheme, use_slm
        );

        // Positional Embedding Params (Fixed)
        // We do NOT store them as a Layer parameter because they are non-learnable.
        // We will generate them on the fly or cache them.

        // Decoder Head

        // Un-Stage 2
        // Input: H/32, W/32, C=8192
        // PixelShuffle factor 4 -> H/8, W/8, C=512
        head_shuffle2_ = std::make_unique<layers::PixelShuffle<T>>(4);

        // Mirror Stage 2 blocks? The spec says "Head (Reconstruction): PixelShuffle mirrors of the encoder. Final: Conv2d"
        // Usually Decoder has processing blocks too. The spec says "Body: 2x ZenithBlock" in Stage 2.
        // It doesn't explicitly mention blocks in the decoder, just "PixelShuffle mirrors".
        // BUT usually AE is symmetric.
        // Assuming "Head" means just the upsampling and final conv.
        // Let's re-read: "Head (Reconstruction): Op: PixelShuffle (Depth-to-Space) mirrors of the encoder. Final: Conv2d to 3 channels."
        // This suggests NO ZenithBlocks in decoder? That would be very asymmetric (heavy encoder, light decoder).
        // OR "mirrors of the encoder" implies the whole structure mirrored.
        // Given "Body: 2x ZenithBlock" is listed under "Stage 2", and "Stage 1" has "Body: 1x ZenithBlock".
        // "Head" lists "Op: PixelShuffle mirrors".
        // I will implement a minimal decoder: Shuffle -> Shuffle -> Conv.
        // Wait, if I shuffle 8192 -> 512, then shuffle 512 -> 32.
        // Then Conv2d (what kernel/stride?)
        // The stem was Conv2d k=2 s=2 (downsample).
        // So final Conv should ideally upsample or just map 32 -> 3.
        // If Stem reduced H -> H/2. Then PixelUnshuffles reduced H/2 -> H/8 -> H/32.
        // Decoder Shuffles: H/32 -> H/8 -> H/2.
        // Final Conv needs to map H/2 -> H?
        // Standard "Conv2d to 3 channels" usually implies 1x1 or 3x3 preserving resolution, not upsampling.
        // If the output must match input size (H, W), we need another 2x upsample.
        // The Stem used Stride 2.
        // So we likely need a PixelShuffle(2) or ConvTranspose(2) at the end.
        // "Head... Final: Conv2d to 3 channels."
        // Maybe the last PixelShuffle mirrors the Stem?
        // Stem: H->H/2 (x2 down). Stage1: x4. Stage2: x4. Total x32.
        // Decoder: x4 (Shuffle), x4 (Shuffle). We are at H/2.
        // To get back to H, we need x2.
        // If "Mirrors of the encoder" includes Stem mirror, then we need a x2 upsample.
        // The spec says "Final: Conv2d".
        // I will add a final PixelShuffle(2) before the Conv2d, or use ConvTranspose.
        // "Op: PixelShuffle (Depth-to-Space) mirrors of the encoder" suggests we mirror the *unshuffles*.
        // Unshuffles were factor 4 and 4.
        // Stem was Conv.
        // So we have:
        // Latent -> Shuffle(4) -> Shuffle(4) -> (Now at H/2) -> ConvTranspose(2) or Upsample -> Conv.
        // I'll assume a final PixelShuffle(2) to mirror the Stem's stride 2.
        // Then a final Conv2d 32->3.

        head_shuffle2_ = std::make_unique<layers::PixelShuffle<T>>(4); // Mirrors Stage 2
        head_shuffle1_ = std::make_unique<layers::PixelShuffle<T>>(4); // Mirrors Stage 1
        head_shuffle_stem_ = std::make_unique<layers::PixelShuffle<T>>(2); // Mirrors Stem Stride 2

        final_conv_ = std::make_unique<layers::Conv2D<T>>(base_channels/4, 3, 3, 1); // 32 channels / 4 (since pixelshuffle 2 divides channels by 4? No.)

        // Wait, PixelShuffle reduces channels by r^2.
        // Stage 2 output: 8192. Shuffle(4) -> 8192/16 = 512. Matches Stage 1 channels.
        // Stage 1 output (from decoder perspective): 512. Shuffle(4) -> 512/16 = 32. Matches Stem channels.
        // Stem output: 32.
        // If we use PixelShuffle(2) to mirror Stem:
        // Input 32. Shuffle(2) -> 32/4 = 8 channels. Output H, W.
        // Then Final Conv: 8 -> 3.

        head_shuffle_stem_ = std::make_unique<layers::PixelShuffle<T>>(2);
        // Input C=32. Output C=8. H_out = H_in * 2.

        // Final Conv: 8 -> 3. Use Padding=1 to preserve 3x3 convolution size (H_out = H_in).
        final_conv_ = std::make_unique<layers::Conv2D<T>>(8, 3, 3, 1, 1);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Stem
        Tensor<T> x = stem_->forward(input); // H/2, W/2, C=32

        // Stage 1
        x = stage1_unshuffle_->forward(x); // H/8, W/8, C=512
        x = stage1_block_->forward(x);

        // Stage 2
        x = stage2_unshuffle_->forward(x); // H/32, W/32, C=8192

        // Injection: Fixed 2D Sinusoidal Positional Embeddings
        if (use_pe_) {
            add_positional_embeddings(x);
        }

        x = stage2_block1_->forward(x);
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

        dx = stage2_block2_->backward(dx);
        dx = stage2_block1_->backward(dx);

        // Positional embeddings are fixed (non-learnable), so gradients just pass through
        // No backward op needed for additive constant.

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
        append(stage2_block1_.get());
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
        append(stage2_block1_.get());
        append(stage2_block2_.get());
        append(final_conv_.get());
        return grads;
    }

    std::string name() const override { return "ZenithHierarchicalAE"; }

private:
    std::unique_ptr<layers::Conv2D<T>> stem_;

    std::unique_ptr<layers::PixelUnshuffle<T>> stage1_unshuffle_;
    std::unique_ptr<layers::ZenithBlock<T>> stage1_block_;
    size_t stage1_channels_;

    std::unique_ptr<layers::PixelUnshuffle<T>> stage2_unshuffle_;
    std::unique_ptr<layers::ZenithBlock<T>> stage2_block1_;
    std::unique_ptr<layers::ZenithBlock<T>> stage2_block2_;
    size_t stage2_channels_;

    std::unique_ptr<layers::PixelShuffle<T>> head_shuffle2_;
    std::unique_ptr<layers::PixelShuffle<T>> head_shuffle1_;
    std::unique_ptr<layers::PixelShuffle<T>> head_shuffle_stem_;
    std::unique_ptr<layers::Conv2D<T>> final_conv_;

    // Positional Embedding Helper
    void add_positional_embeddings(Tensor<T>& x) {
        // x shape: N, H, W, C
        auto shape = x.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3]; // Should be 8192

        // Thread-safe lazy initialization
        if (pe_cache_.shape().empty() || pe_cache_.shape()[1] != H || pe_cache_.shape()[2] != W || pe_cache_.shape()[3] != C) {
            std::lock_guard<std::mutex> lock(pe_mutex_);
            // Double-check locking pattern
            if (pe_cache_.shape().empty() || pe_cache_.shape()[1] != H || pe_cache_.shape()[2] != W || pe_cache_.shape()[3] != C) {
                // Recompute cache if dimensions change
                pe_cache_ = Tensor<T>({1, H, W, C});
                T* p = pe_cache_.data();

                size_t C_half = C / 2;

                for(size_t h=0; h<H; ++h) {
                    for(size_t w=0; w<W; ++w) {
                        T* pixel = p + (h*W + w)*C;

                        // Y embedding (Channels 0 to C_half - 1)
                        for(size_t c=0; c<C_half; ++c) {
                            size_t k = c / 2;
                            size_t max_k = C_half / 2;

                            double normalized_idx = (double)k / (double)max_k;
                            double omega = 1.0 / std::pow(10000.0, 1.0 - normalized_idx);

                            if (c % 2 == 0) {
                                pixel[c] = std::sin(h * omega);
                            } else {
                                pixel[c] = std::cos(h * omega);
                            }
                        }

                        // X embedding (Channels C_half to C - 1)
                        for(size_t c=0; c<C_half; ++c) {
                            size_t k = c / 2;
                            size_t max_k = C_half / 2;

                            double normalized_idx = (double)k / (double)max_k;
                            double omega = 1.0 / std::pow(10000.0, 1.0 - normalized_idx);

                            if (c % 2 == 0) {
                                pixel[C_half + c] = std::sin(w * omega);
                            } else {
                                pixel[C_half + c] = std::cos(w * omega);
                            }
                        }
                    }
                }
            }
        }

        // Add to x
        T* x_ptr = x.data();
        const T* p_ptr = pe_cache_.data();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    size_t idx = ((n*H + h)*W + w)*C;
                    size_t p_idx = (h*W + w)*C;
                    for(size_t c=0; c<C; ++c) {
                        x_ptr[idx + c] += p_ptr[p_idx + c];
                    }
                }
            }
        }
    }

    Tensor<T> pe_cache_;
    bool use_pe_;
    std::mutex pe_mutex_;
};

} // namespace models
} // namespace dreidel

#endif
