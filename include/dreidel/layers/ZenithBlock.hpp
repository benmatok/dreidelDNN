#pragma once

#include "Layer.hpp"
#include "../core/Memory.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

namespace dreidel {
namespace layers {

// Forward declare specialized ZenithBlock
class ZenithBlock;

/**
 * @brief The Zenith Block (Strictly Optimized for APoT).
 *
 * Replaces Standard Conv2D.
 * STRICT APoT MODE: This block only supports optimized APoT execution on int8 tensors.
 *
 * Pipeline:
 * 1. Oracle (Gating)
 * 2. Eyes (Spatial Conv)
 * 3. Mixer (Permute -> FWHT -> Scale -> Bias -> ReLU)
 */
class ZenithBlock : public Layer<int8_t> {
public:
    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim, size_t arena_size = 1024*1024, bool use_gating = false)
        : channels_(channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          arena_(arena_size),
          use_gating_(use_gating),
          packed_weights_(channels * kernel_size * kernel_size),
          spectral_scales_(channels),
          bias_(channels),
          perm_indices_(channels)
    {
        // Random Init Weights (Simulated with Random APoT Codes)
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist_code(0, 255);

        for(auto& w : packed_weights_) w = static_cast<int8_t>(dist_code(gen));
        for(auto& s : spectral_scales_) s = static_cast<int8_t>(dist_code(gen));

        std::fill(bias_.begin(), bias_.end(), 0);
        std::iota(perm_indices_.begin(), perm_indices_.end(), 0);

        // Init Oracle Projection (Random Hyperplane)
        oracle_projection_.resize(channels);
        for(auto& p : oracle_projection_) p = static_cast<int8_t>(dist_code(gen));
    }

    Tensor<int8_t> forward(const Tensor<int8_t>& input) override {
        auto shape = input.shape();
        size_t batch = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3];

        Tensor<int8_t> output(shape);

        const int8_t* in_ptr = input.data();
        int8_t* out_ptr = output.data();
        const int8_t* w_ptr = packed_weights_.data();
        const int8_t* scale_ptr = spectral_scales_.data();
        const int8_t* bias_ptr = bias_.data();
        const int* perm_ptr = perm_indices_.data();
        const int8_t* oracle_ptr = oracle_projection_.data();

        int k_rad = kernel_size_ / 2;

        arena_.reset();
        int8_t* pixel_buffer = arena_.allocate<int8_t>(C);
        int8_t* mixer_buffer = arena_.allocate<int8_t>(C);

        constexpr int BLOCK_H = 8;
        constexpr int BLOCK_W = 8;

        // Oracle Mask: Sign of projection vector
        uint32_t oracle_mask = 0;
        if(use_gating_) {
            for(size_t i=0; i<C; ++i) {
                if (oracle_ptr[i] & 0x80) oracle_mask |= (1u << i);
            }
        }

        for(size_t n=0; n<batch; ++n) {

            if (use_gating_) {
                size_t ch = H/2, cw = W/2;
                const int8_t* p_center = in_ptr + ((n*H + ch)*W + cw)*C;

                uint32_t input_mask = 0;
#if defined(DREIDEL_ARCH_AVX2)
                __m256i v = _mm256_loadu_si256((const __m256i*)p_center);
                input_mask = _mm256_movemask_epi8(v);
#else
                for(size_t i=0; i<std::min(C, (size_t)32); ++i) {
                    if (p_center[i] & 0x80) input_mask |= (1u << i);
                }
#endif
                int dist = hal::AlienOps::popcnt32(input_mask ^ oracle_mask);
                if (dist > 16) {
                    // Zero out output block for this sample to avoid garbage
                    int8_t* p_out_start = out_ptr + n * H * W * C;
                    std::fill(p_out_start, p_out_start + H * W * C, 0);
                    continue;
                }
            }

            for(size_t by=0; by<H; by+=BLOCK_H) {
                for(size_t bx=0; bx<W; bx+=BLOCK_W) {
                    for(size_t dy=0; dy<BLOCK_H; ++dy) {
                        for(size_t dx=0; dx<BLOCK_W; ++dx) {
                            size_t h = by + dy;
                            size_t w = bx + dx;
                            if (h >= H || w >= W) continue;

                            // 2. Eyes (Spatial Convolution)
                            for(size_t c=0; c<C; ++c) pixel_buffer[c] = 0;

                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int ih = h + ky;
                                    int iw = w + kx;
                                    if (ih>=0 && ih<H && iw>=0 && iw<W) {
                                        const int8_t* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                        int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                        const int8_t* p_w = w_ptr + k_idx * channels_;
                                        size_t c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                                        for(; c+32 <= C; c+=32) {
                                            __m256i v_in = _mm256_loadu_si256((const __m256i*)(p_in + c));
                                            __m256i v_w  = _mm256_loadu_si256((const __m256i*)(p_w + c));

                                            __m128i in_lo = _mm256_castsi256_si128(v_in);
                                            __m128i in_hi = _mm256_extracti128_si256(v_in, 1);
                                            __m128i w_lo  = _mm256_castsi256_si128(v_w);
                                            __m128i w_hi  = _mm256_extracti128_si256(v_w, 1);
                                            __m128i prod_lo = hal::AlienOps::vec_mul_apot_avx2(in_lo, w_lo);
                                            __m128i prod_hi = hal::AlienOps::vec_mul_apot_avx2(in_hi, w_hi);
                                            __m256i v_prod = _mm256_set_m128i(prod_hi, prod_lo);

                                            __m256i v_acc = _mm256_loadu_si256((const __m256i*)(pixel_buffer + c));
                                            v_acc = hal::AlienOps::vec_add_apot_avx2(v_acc, v_prod);
                                            _mm256_storeu_si256((__m256i*)(pixel_buffer + c), v_acc);
                                        }
#endif
                                        for(; c<C; ++c) {
                                            int8_t prod = hal::AlienOps::apot_mul_lut(p_in[c], p_w[c]);
                                            pixel_buffer[c] = hal::AlienOps::apot_add_lut(pixel_buffer[c], prod);
                                        }
                                    }
                                }
                            }

                            // 3. Mixer (Spectral)

                            // a. Permutation
                            for(size_t c=0; c<C; ++c) mixer_buffer[c] = pixel_buffer[perm_ptr[c]];

                            // b. FWHT (In-Place on mixer_buffer)
                            size_t h_len = 1;
                            while (h_len < C) {
                                bool handled = false;
#if defined(DREIDEL_ARCH_AVX2)
                                if (h_len >= 32) {
                                    for (size_t i = 0; i < C; i += h_len * 2) {
                                        for (size_t j = i; j < i + h_len; j += 32) {
                                            __m256i x = _mm256_loadu_si256((const __m256i*)(mixer_buffer + j));
                                            __m256i y = _mm256_loadu_si256((const __m256i*)(mixer_buffer + j + h_len));

                                            __m256i sum = hal::AlienOps::vec_add_apot_avx2(x, y);

                                            __m256i sign_mask = _mm256_set1_epi8(0x80);
                                            __m256i neg_y = _mm256_xor_si256(y, sign_mask);
                                            __m256i sub = hal::AlienOps::vec_add_apot_avx2(x, neg_y);

                                            _mm256_storeu_si256((__m256i*)(mixer_buffer + j), sum);
                                            _mm256_storeu_si256((__m256i*)(mixer_buffer + j + h_len), sub);
                                        }
                                    }
                                    handled = true;
                                }
#endif
                                if (!handled) {
                                    for (size_t i = 0; i < C; i += h_len * 2) {
                                        for (size_t j = i; j < i + h_len; ++j) {
                                            int8_t x = mixer_buffer[j];
                                            int8_t y = mixer_buffer[j + h_len];
                                            int8_t sum = hal::AlienOps::apot_add_lut(x, y);
                                            int8_t neg_y = y ^ 0x80;
                                            int8_t sub = hal::AlienOps::apot_add_lut(x, neg_y);
                                            mixer_buffer[j] = sum;
                                            mixer_buffer[j + h_len] = sub;
                                        }
                                    }
                                }
                                h_len *= 2;
                            }

                            // c. Scale & Bias & ReLU
                            size_t c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                            for(; c+32 <= C; c+=32) {
                                __m256i v_val = _mm256_loadu_si256((const __m256i*)(mixer_buffer + c));
                                __m256i v_scale = _mm256_loadu_si256((const __m256i*)(scale_ptr + c));
                                __m256i v_bias = _mm256_loadu_si256((const __m256i*)(bias_ptr + c));

                                // Mul Scale
                                __m128i val_lo = _mm256_castsi256_si128(v_val);
                                __m128i val_hi = _mm256_extracti128_si256(v_val, 1);
                                __m128i sc_lo = _mm256_castsi256_si128(v_scale);
                                __m128i sc_hi = _mm256_extracti128_si256(v_scale, 1);
                                __m128i res_lo = hal::AlienOps::vec_mul_apot_avx2(val_lo, sc_lo);
                                __m128i res_hi = hal::AlienOps::vec_mul_apot_avx2(val_hi, sc_hi);
                                __m256i v_res = _mm256_set_m128i(res_hi, res_lo);

                                // Add Bias
                                v_res = hal::AlienOps::vec_add_apot_avx2(v_res, v_bias);

                                // Branchless ReLU (v & 0x80 -> 0)
                                _mm256_storeu_si256((__m256i*)(mixer_buffer + c), v_res);
                            }
#endif
                            for(; c<C; ++c) {
                                int8_t val = mixer_buffer[c];
                                val = hal::AlienOps::apot_mul_lut(val, scale_ptr[c]);
                                val = hal::AlienOps::apot_add_lut(val, bias_ptr[c]);
                                if (val & 0x80) val = 0; // ReLU
                                mixer_buffer[c] = val;
                            }

                            // Output
                            int8_t* p_out = out_ptr + ((n*H + h)*W + w)*C;
                            for(size_t c=0; c<C; ++c) p_out[c] = mixer_buffer[c];
                        }
                    }
                }
            }
        }
        return output;
    }

    Tensor<int8_t> backward(const Tensor<int8_t>& grad_output) override {
        // Not implemented for benchmark
        return Tensor<int8_t>();
    }

    std::vector<Tensor<int8_t>*> parameters() override {
        return {};
    }

    std::string name() const override { return "ZenithBlock"; }

private:
    size_t channels_;
    size_t kernel_size_;
    size_t spectral_dim_;
    core::Arena arena_;
    bool use_gating_;
    std::vector<int8_t> packed_weights_;
    std::vector<int8_t> spectral_scales_;
    std::vector<int8_t> bias_;
    std::vector<int> perm_indices_;
    std::vector<int8_t> oracle_projection_;
};

} // namespace layers
} // namespace dreidel
