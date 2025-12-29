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
 *
 * Auto-Switching Connectivity:
 * - Backbone (Cin == Cout): Uses Depthwise Spatial Conv (O(C) params) + Spectral Mixing. "Alien Speed".
 * - Stem/Head (Cin != Cout): Uses Dense Spatial Conv (O(Cin*Cout) params) + Spectral Mixing.
 *
 * Pipeline:
 * 1. Oracle (Gating)
 * 2. Eyes (Spatial Conv)
 * 3. Mixer (Permute -> FWHT -> Scale -> Bias -> ReLU)
 */
class ZenithBlock : public Layer<int8_t> {
public:
    // Legacy Constructor (Cin == Cout -> Depthwise)
    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim, size_t arena_size = 1024*1024, bool use_gating = false)
        : ZenithBlock(channels, channels, kernel_size, spectral_dim, arena_size, use_gating) {}

    // Scalable Constructor (Auto-Detect Mode)
    ZenithBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t spectral_dim, size_t arena_size = 1024*1024, bool use_gating = false)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          arena_(arena_size),
          use_gating_(use_gating),
          spectral_scales_(out_channels),
          bias_(out_channels),
          perm_indices_(out_channels)
    {
        // Auto-Detect Mode
        // If dimensions match, we assume efficient backbone (Depthwise).
        // If dimensions differ, we assume stem/projection (Dense).
        is_depthwise_ = (in_channels_ == out_channels_);

        size_t total_weights = 0;
        if (is_depthwise_) {
            // Depthwise: 1 kernel per channel pair (Diagonal)
            total_weights = out_channels_ * kernel_size * kernel_size;
        } else {
            // Dense: In kernels per Out channel
            total_weights = out_channels_ * in_channels_ * kernel_size * kernel_size;
        }
        packed_weights_.resize(total_weights);

        // Random Init
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist_code(0, 255);

        for(auto& w : packed_weights_) w = static_cast<int8_t>(dist_code(gen));
        for(auto& s : spectral_scales_) s = static_cast<int8_t>(dist_code(gen));

        std::fill(bias_.begin(), bias_.end(), 0);
        std::iota(perm_indices_.begin(), perm_indices_.end(), 0);

        oracle_projection_.resize(in_channels_);
        for(auto& p : oracle_projection_) p = static_cast<int8_t>(dist_code(gen));
    }

#if defined(DREIDEL_ARCH_AVX2)
    // Intra-Register FWHT for 32 elements (YMM) - Unchanged
    static inline __m256i fwht_avx2_intra(__m256i v) {
        // Stride 1
        {
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14));
            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15));
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);
            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, _mm256_xor_si256(b, _mm256_set1_epi8(0x80)));
            v = _mm256_unpacklo_epi8(s, d);
        }
        // Stride 2
        {
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,1,4,5,8,9,12,13, 0,1,4,5,8,9,12,13));
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(2,3,6,7,10,11,14,15, 2,3,6,7,10,11,14,15));
            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);
            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, _mm256_xor_si256(b, _mm256_set1_epi8(0x80)));
            v = _mm256_unpacklo_epi16(s, d);
        }
        // Stride 4
        {
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,1,2,3,8,9,10,11, 0,1,2,3,8,9,10,11));
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(4,5,6,7,12,13,14,15, 4,5,6,7,12,13,14,15));
            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);
            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, _mm256_xor_si256(b, _mm256_set1_epi8(0x80)));
            v = _mm256_unpacklo_epi32(s, d);
        }
        // Stride 8
        {
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7));
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15));
            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);
            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, _mm256_xor_si256(b, _mm256_set1_epi8(0x80)));
            v = _mm256_unpacklo_epi64(s, d);
        }
        // Stride 16
        {
            __m128i lane0 = _mm256_castsi256_si128(v);
            __m128i lane1 = _mm256_extracti128_si256(v, 1);
            __m256i a = _mm256_broadcastsi128_si256(lane0);
            __m256i b = _mm256_broadcastsi128_si256(lane1);
            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, _mm256_xor_si256(b, _mm256_set1_epi8(0x80)));
            v = _mm256_permute2x128_si256(s, d, 0x20);
        }
        return v;
    }
#endif

    Tensor<int8_t> forward(const Tensor<int8_t>& input) override {
        auto shape = input.shape();
        size_t batch = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];

        Tensor<int8_t> output({batch, H, W, out_channels_});

        const int8_t* in_ptr = input.data();
        int8_t* out_ptr = output.data();
        const int8_t* w_ptr = packed_weights_.data();
        const int8_t* scale_ptr = spectral_scales_.data();
        const int8_t* bias_ptr = bias_.data();
        const int* perm_ptr = perm_indices_.data();
        const int8_t* oracle_ptr = oracle_projection_.data();

        int k_rad = kernel_size_ / 2;
        arena_.reset();
        int8_t* pixel_buffer = arena_.allocate<int8_t>(out_channels_);
        int8_t* mixer_buffer = arena_.allocate<int8_t>(out_channels_);

        constexpr int BLOCK_H = 8;
        constexpr int BLOCK_W = 8;

        for(size_t n=0; n<batch; ++n) {

            // Gating on Input (Always uses in_channels_)
            if (use_gating_) {
                size_t ch = H/2, cw = W/2;
                const int8_t* p_center = in_ptr + ((n*H + ch)*W + cw)*in_channels_;
                int dist = 0;
                size_t i = 0;
#if defined(DREIDEL_ARCH_AVX2)
                for(; i + 32 <= in_channels_; i += 32) {
                    __m256i v_in = _mm256_loadu_si256((const __m256i*)(p_center + i));
                    __m256i v_proj = _mm256_loadu_si256((const __m256i*)(oracle_ptr + i));
                    uint32_t mask_in = _mm256_movemask_epi8(v_in);
                    uint32_t mask_proj = _mm256_movemask_epi8(v_proj);
                    dist += hal::AlienOps::popcnt32(mask_in ^ mask_proj);
                }
#endif
                for(; i < in_channels_; ++i) {
                    bool s1 = (p_center[i] & 0x80);
                    bool s2 = (oracle_ptr[i] & 0x80);
                    if (s1 != s2) dist++;
                }
                if (dist > 16) {
                    int8_t* p_out_start = out_ptr + n * H * W * out_channels_;
                    std::fill(p_out_start, p_out_start + H * W * out_channels_, 0);
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

                            // 2. Eyes (Spatial)

                            // AUTO-MODE:
                            if (is_depthwise_) {
                                // DEPTHWISE PATH (Efficient O(C))
                                // Requires Cin == Cout.
                                size_t C = out_channels_;
                                std::fill(pixel_buffer, pixel_buffer + C, 0);

                                for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                    for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                        int ih = h + ky;
                                        int iw = w + kx;
                                        if (ih>=0 && ih<H && iw>=0 && iw<W) {
                                            const int8_t* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                            int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                            const int8_t* p_w = w_ptr + k_idx * C;
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
                            } else {
                                // DENSE PATH (Flexible O(Cin*Cout))
                                // Used for Expansion/Reduction.
                                std::fill(pixel_buffer, pixel_buffer + out_channels_, 0);
                                size_t w_stride_filter = in_channels_ * kernel_size_ * kernel_size_;
                                size_t w_stride_spatial = in_channels_;

                                for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                    for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                        int ih = h + ky;
                                        int iw = w + kx;
                                        if (ih>=0 && ih<H && iw>=0 && iw<W) {
                                            const int8_t* p_in_base = in_ptr + ((n*H + ih)*W + iw)*in_channels_;
                                            int k_idx_offset = ((ky+k_rad)*kernel_size_ + (kx+k_rad)) * w_stride_spatial;

                                            for(size_t o=0; o<out_channels_; ++o) {
                                                const int8_t* p_w = w_ptr + o * w_stride_filter + k_idx_offset;
                                                int8_t acc = pixel_buffer[o];

                                                // Dense Inner Loop: Reduce over Input Channels
                                                for(size_t i=0; i<in_channels_; ++i) {
                                                    int8_t val = p_in_base[i];
                                                    int8_t w = p_w[i];
                                                    int8_t prod = hal::AlienOps::apot_mul_lut(val, w);
                                                    acc = hal::AlienOps::apot_add_lut(acc, prod);
                                                }
                                                pixel_buffer[o] = acc;
                                            }
                                        }
                                    }
                                }
                            }

                            // 3. Mixer (Spectral) - Uses out_channels_
                            size_t C = out_channels_;
                            for(size_t c=0; c<C; ++c) mixer_buffer[c] = pixel_buffer[perm_ptr[c]];

                            // FWHT (Hybrid Strategy on 'C')
                            size_t c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                            for (; c + 32 <= C; c += 32) {
                                __m256i v = _mm256_loadu_si256((const __m256i*)(mixer_buffer + c));
                                v = fwht_avx2_intra(v);
                                _mm256_storeu_si256((__m256i*)(mixer_buffer + c), v);
                            }
#endif
                            if (c < C) {
                                size_t limit = C - c;
                                size_t sub_h = 1;
                                while (sub_h < 32 && sub_h < limit) {
                                    for(size_t i=c; i<C; i+=sub_h*2) {
                                        for(size_t j=i; j<i+sub_h; ++j) {
                                            if (j+sub_h < C) {
                                                int8_t x = mixer_buffer[j];
                                                int8_t y = mixer_buffer[j + sub_h];
                                                mixer_buffer[j] = hal::AlienOps::apot_add_lut(x, y);
                                                mixer_buffer[j + sub_h] = hal::AlienOps::apot_add_lut(x, y ^ 0x80);
                                            }
                                        }
                                    }
                                    sub_h *= 2;
                                }
                            }

                            // Inter-Register Pass
                            size_t h_len = 32;
                            while (h_len < C) {
                                bool handled = false;
#if defined(DREIDEL_ARCH_AVX2)
                                for (size_t i = 0; i < C; i += h_len * 2) {
                                    for (size_t j = i; j < i + h_len; j += 32) {
                                        __m256i x = _mm256_loadu_si256((const __m256i*)(mixer_buffer + j));
                                        __m256i y = _mm256_loadu_si256((const __m256i*)(mixer_buffer + j + h_len));
                                        __m256i sum = hal::AlienOps::vec_add_apot_avx2(x, y);
                                        __m256i neg_y = _mm256_xor_si256(y, _mm256_set1_epi8(0x80));
                                        __m256i sub = hal::AlienOps::vec_add_apot_avx2(x, neg_y);
                                        _mm256_storeu_si256((__m256i*)(mixer_buffer + j), sum);
                                        _mm256_storeu_si256((__m256i*)(mixer_buffer + j + h_len), sub);
                                    }
                                }
                                handled = true;
#endif
                                if (!handled) {
                                    for (size_t i = 0; i < C; i += h_len * 2) {
                                        for (size_t j = i; j < i + h_len; ++j) {
                                            int8_t x = mixer_buffer[j];
                                            int8_t y = mixer_buffer[j + h_len];
                                            mixer_buffer[j] = hal::AlienOps::apot_add_lut(x, y);
                                            mixer_buffer[j + h_len] = hal::AlienOps::apot_add_lut(x, y ^ 0x80);
                                        }
                                    }
                                }
                                h_len *= 2;
                            }

                            // C. Scale & Bias & ReLU
                            c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                            for(; c+32 <= C; c+=32) {
                                __m256i v_val = _mm256_loadu_si256((const __m256i*)(mixer_buffer + c));
                                __m256i v_scale = _mm256_loadu_si256((const __m256i*)(scale_ptr + c));
                                __m256i v_bias = _mm256_loadu_si256((const __m256i*)(bias_ptr + c));
                                __m128i val_lo = _mm256_castsi256_si128(v_val);
                                __m128i val_hi = _mm256_extracti128_si256(v_val, 1);
                                __m128i sc_lo = _mm256_castsi256_si128(v_scale);
                                __m128i sc_hi = _mm256_extracti128_si256(v_scale, 1);
                                __m128i res_lo = hal::AlienOps::vec_mul_apot_avx2(val_lo, sc_lo);
                                __m128i res_hi = hal::AlienOps::vec_mul_apot_avx2(val_hi, sc_hi);
                                __m256i v_res = _mm256_set_m128i(res_hi, res_lo);
                                v_res = hal::AlienOps::vec_add_apot_avx2(v_res, v_bias);
                                __m256i zero = _mm256_setzero_si256();
                                v_res = _mm256_blendv_epi8(v_res, zero, v_res);
                                _mm256_storeu_si256((__m256i*)(mixer_buffer + c), v_res);
                            }
#endif
                            for(; c<C; ++c) {
                                int8_t val = mixer_buffer[c];
                                val = hal::AlienOps::apot_mul_lut(val, scale_ptr[c]);
                                val = hal::AlienOps::apot_add_lut(val, bias_ptr[c]);
                                if (val & 0x80) val = 0;
                                mixer_buffer[c] = val;
                            }

                            int8_t* p_out = out_ptr + ((n*H + h)*W + w)*C;
                            for(size_t i=0; i<C; ++i) p_out[i] = mixer_buffer[i];
                        }
                    }
                }
            }
        }
        return output;
    }

    Tensor<int8_t> backward(const Tensor<int8_t>& grad_output) override { return Tensor<int8_t>(); }
    std::vector<Tensor<int8_t>*> parameters() override { return {}; }
    std::string name() const override { return "ZenithBlock"; }

    // Public setter for weights for Conversion Benchmark
    void set_weights(const std::vector<int8_t>& w) {
        if(w.size() == packed_weights_.size()) packed_weights_ = w;
    }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t spectral_dim_;
    core::Arena arena_;
    bool use_gating_;
    bool is_depthwise_;
    std::vector<int8_t> packed_weights_;
    std::vector<int8_t> spectral_scales_;
    std::vector<int8_t> bias_;
    std::vector<int> perm_indices_;
    std::vector<int8_t> oracle_projection_;
};

} // namespace layers
} // namespace dreidel
