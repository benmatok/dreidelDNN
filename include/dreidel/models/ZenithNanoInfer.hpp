#pragma once

#include "../layers/Layer.hpp"
#include "../core/Tensor.hpp"
#include "../layers/ZenithNanoBlock_F16.hpp"
#include "../layers/OptimizedConv2D_F16.hpp"
#include "../kernels/AvxFwht.hpp"
#include "../hal/ops_f16.hpp"
#include <vector>
#include <memory>
#include <iostream>

namespace dreidel {
namespace models {

class ZenithNanoInfer : public layers::Layer<float> {
public:
    // Inference Only Variant of ZenithNano
    // Uses F16 storage for internal buffers to save bandwidth.
    // IO is F32 (compatible with standard pipeline).

    ZenithNanoInfer() {
        // Allocate Arena (F16)
        // Sizes in elements (uint16_t)
        size_t sz_s1 = 64 * 64 * 192; // 768 KB
        size_t sz_s2 = 64 * 64 * 64;  // 256 KB
        size_t sz_b = 64 * 64 * 64;   // 256 KB
        size_t sz_s4 = 64 * 64 * 192; // 768 KB

        size_t pad = 64; // Padding

        size_t off_s1 = 0;
        size_t off_s2 = off_s1 + sz_s1 + pad;
        size_t off_b1 = off_s2 + sz_s2 + pad;
        size_t off_b2 = off_b1 + sz_b + pad;
        size_t off_b3 = off_b2 + sz_b + pad;
        size_t off_s4 = off_b3 + sz_b + pad;

        size_t total = off_s4 + sz_s4 + pad;

        arena_.resize(total);
        uint16_t* base = arena_.data();

        s1_out_ = Tensor<uint16_t>({1, 64, 64, 192}, base + off_s1);
        s2_out_ = Tensor<uint16_t>({1, 64, 64, 64}, base + off_s2);
        b1_out_ = Tensor<uint16_t>({1, 64, 64, 64}, base + off_b1);
        b2_out_ = Tensor<uint16_t>({1, 64, 64, 64}, base + off_b2);
        b3_out_ = Tensor<uint16_t>({1, 64, 64, 64}, base + off_b3);
        s4_out_ = Tensor<uint16_t>({1, 64, 64, 192}, base + off_s4);

        compress_ = std::make_unique<layers::OptimizedConv2D_F16>(192, 64, 1, 1, 0);

        block1_ = std::make_unique<layers::ZenithNanoBlock_F16>(64, 64, 64);
        block2_ = std::make_unique<layers::ZenithNanoBlock_F16>(64, 64, 64);
        block3_ = std::make_unique<layers::ZenithNanoBlock_F16>(64, 64, 64);

        expand_ = std::make_unique<layers::OptimizedConv2D_F16>(64, 192, 1, 1, 0);
    }

    // Load Weights from Float Tensors (e.g. from loaded ZenithNano)
    void load_weights(
        const Tensor<float>& c_w, const Tensor<float>* c_b,
        const Tensor<float>& e_w, const Tensor<float>* e_b,
        const std::vector<Tensor<float>>& b1_p,
        const std::vector<Tensor<float>>& b2_p,
        const std::vector<Tensor<float>>& b3_p
    ) {
        compress_->set_weights(c_w, c_b);
        expand_->set_weights(e_w, e_b);

        // Block Params: GateH, GateW, ConvW, ConvB (ConvB optional, likely not used in ZenithNanoBlock proj?)
        // ZenithNanoBlock parameters(): {gate_h, gate_w, proj_conv_params...}
        // OptimizedConv2D parameters(): {weights, bias(optional)}

        auto load_block = [&](auto& blk, const std::vector<Tensor<float>>& p) {
            // p[0] = GateH, p[1] = GateW
            blk->set_gates(p[0], p[1]);
            // p[2] = Conv Weights, p[3] = Conv Bias (if any)
            const Tensor<float>* bias = (p.size() > 3) ? &p[3] : nullptr;
            blk->get_conv().set_weights(p[2], bias);
        };

        load_block(block1_, b1_p);
        load_block(block2_, b2_p);
        load_block(block3_, b3_p);
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        // S1: SpaceToDepth (Float)
        // [1, 512, 512, 3] -> [1, 64, 64, 192]
        int H = 512;
        int W = 512;
        int C = 3;
        int block = 8;

        // We need a temporary float buffer for S1 output OR we convert on the fly.
        // SpaceToDepth_Shuffle is just a copy.
        // We can create a specialized SpaceToDepth_F32_to_F16?
        // Or just do SpaceToDepth to a float buffer, then convert?
        // Allocating 64*64*192*4 = 3MB. Not too bad.
        // Or better: integrate conversion into SpaceToDepth.

        // Let's implement SpaceToDepth_Shuffle_F16 locally for now.
        space_to_depth_f16(input.data(), s1_out_.data(), H, W, C, block);

        // S2: Compress (F16 -> F16)
        compress_->forward(s1_out_, s2_out_);

        // S3: Zenith Blocks (F16 -> F16)
        block1_->forward(s2_out_, b1_out_);
        block2_->forward(b1_out_, b2_out_);
        block3_->forward(b2_out_, b3_out_);

        // S4: Expand (F16 -> F16)
        expand_->forward(b3_out_, s4_out_);

        // S5: DepthToSpace (F16 -> F32)
        Tensor<float> output({1, (size_t)H, (size_t)W, (size_t)C});
        depth_to_space_f16(s4_out_.data(), output.data(), H/block, W/block, C, block);

        return output;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override { return grad_output; }
    std::vector<Tensor<float>*> parameters() override { return {}; }
    std::string name() const override { return "ZenithNanoInfer"; }

private:
    std::unique_ptr<layers::OptimizedConv2D_F16> compress_;
    std::unique_ptr<layers::ZenithNanoBlock_F16> block1_;
    std::unique_ptr<layers::ZenithNanoBlock_F16> block2_;
    std::unique_ptr<layers::ZenithNanoBlock_F16> block3_;
    std::unique_ptr<layers::OptimizedConv2D_F16> expand_;

    // Persistent buffers (F16)
    Tensor<uint16_t> s1_out_;
    Tensor<uint16_t> s2_out_;
    Tensor<uint16_t> b1_out_;
    Tensor<uint16_t> b2_out_;
    Tensor<uint16_t> b3_out_;
    Tensor<uint16_t> s4_out_;

    std::vector<uint16_t, core::AlignedAllocator<uint16_t>> arena_;

    // Helpers
    void space_to_depth_f16(const float* src, uint16_t* dst, int H, int W, int C, int block) {
        // Naive implementation with F16 conversion
        int out_h = H / block;
        int out_w = W / block;
        // int out_c = C * block * block;

        #pragma omp parallel for collapse(2)
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                uint16_t* p_out = dst + (h * out_w + w) * (C * block * block);
                for (int r = 0; r < block; ++r) {
                    for (int c = 0; c < block; ++c) {
                        const float* p_in = src + ((h * block + r) * W + (w * block + c)) * C;
                        for (int k = 0; k < C; ++k) {
                            // Convert and Store
                            __m128i h_val = _mm_cvtps_ph(_mm_set_ss(p_in[k]), 0);
                            *p_out++ = (uint16_t)_mm_cvtsi128_si32(h_val);
                        }
                    }
                }
            }
        }
    }

    void depth_to_space_f16(const uint16_t* src, float* dst, int H, int W, int C, int block) {
        // H, W are dimensions of src (small)
        // dst is H*block, W*block

        #pragma omp parallel for collapse(2)
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const uint16_t* p_in = src + (h * W + w) * (C * block * block);
                for (int r = 0; r < block; ++r) {
                    for (int c = 0; c < block; ++c) {
                        float* p_out = dst + ((h * block + r) * (W * block) + (w * block + c)) * C;
                        for (int k = 0; k < C; ++k) {
                            uint16_t val_u = *p_in++;
                            __m128i val_h = _mm_set1_epi16((short)val_u);
                            __m256 val_v = _mm256_cvtph_ps(val_h);
                            float val_f = _mm_cvtss_f32(_mm256_castps256_ps128(val_v));
                            p_out[k] = val_f;
                        }
                    }
                }
            }
        }
    }
};

} // namespace models
} // namespace dreidel
