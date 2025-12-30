#pragma once

#include "Layer.hpp"
#include "../core/Memory.hpp"
#include "../hal/ops.hpp"
#include "../algo/WHT.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <omp.h>

namespace dreidel {
namespace layers {

// Forward declare specialized ZenithBlock
template <typename T>
class ZenithBlock;

// -----------------------------------------------------------------------------
// ZenithBlock<float>: New Architecture (IFWHT, Dilated, Gating)
// -----------------------------------------------------------------------------
template <>
class ZenithBlock<float> : public Layer<float> {
public:
    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = true, bool use_gating = false)
        : channels_(channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_dilated_(use_dilated), use_gating_(use_gating),
          packed_weights_({channels, 1, kernel_size, kernel_size}),
          spectral_scales_({1, channels}),
          soft_perm_weights_({1, 3}),
          dilated_perm_weights_({1, 3}),
          bias_({1, channels}),
          oracle_projection_({1, channels}),

          grad_packed_weights_({channels, 1, kernel_size, kernel_size}),
          grad_spectral_scales_({1, channels}),
          grad_soft_perm_weights_({1, 3}),
          grad_dilated_perm_weights_({1, 3}),
          grad_bias_({1, channels})
    {
        // Init
        float stddev = std::sqrt(2.0f / (kernel_size * kernel_size * channels));
        packed_weights_.random(0, stddev);
        spectral_scales_.fill(1.0f);
        soft_perm_weights_.fill(0); soft_perm_weights_.data()[1] = 1.0f;
        dilated_perm_weights_.fill(0);
        bias_.fill(0);
        oracle_projection_.random(-1.0f, 1.0f);

        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_soft_perm_weights_.fill(0);
        grad_dilated_perm_weights_.fill(0);
        grad_bias_.fill(0);
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        // ... (The float implementation from previous step)
        input_cached_ = input;
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];

        if (C != channels_) throw std::runtime_error("Channel mismatch in ZenithBlock");

        Tensor<float> output(shape);
        float* out_ptr = output.data();
        const float* in_ptr = input.data();

        // 1. Gating
        std::vector<bool> active_mask(N, true);
        if (use_gating_) {
            const float* oracle_ptr = oracle_projection_.data();
            for(size_t n=0; n<N; ++n) {
                size_t ch = H/2, cw = W/2;
                const float* p_center = in_ptr + ((n*H + ch)*W + cw)*C;
                float dot = 0;
                for(size_t c=0; c<C; ++c) dot += p_center[c] * oracle_ptr[c];
                if (dot < 0) active_mask[n] = false;
            }
        }

        if (eyes_out_cached_.shape() != shape) eyes_out_cached_ = Tensor<float>(shape);

        // 2. Eyes
        int k_rad = kernel_size_ / 2;
        const float* w_ptr = packed_weights_.data();
        float* eyes_ptr = eyes_out_cached_.data();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    if (!active_mask[n]) {
                         for(size_t c=0; c<C; ++c) {
                             eyes_ptr[((n*H + h)*W + w)*C + c] = 0;
                             out_ptr[((n*H + h)*W + w)*C + c] = 0;
                         }
                         continue;
                    }
                    for(size_t c=0; c<C; ++c) {
                        float val = 0;
                        for(int ky=-k_rad; ky<=k_rad; ++ky) {
                            for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                int ih = h + ky; int iw = w + kx;
                                if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float pixel = in_ptr[((n*H + ih)*W + iw)*C + c];
                                    float weight = w_ptr[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                    val += pixel * weight;
                                }
                            }
                        }
                        eyes_ptr[((n*H + h)*W + w)*C + c] = val;
                    }
                }
            }
        }

        // 3. Mixer
        const float* scale_ptr = spectral_scales_.data();
        const float* bias_ptr = bias_.data();
        const float* sp_w = soft_perm_weights_.data();
        const float* dp_w = dilated_perm_weights_.data();
        int dilation = static_cast<int>(std::sqrt(C));

        #pragma omp parallel
        {
            std::vector<float> buf(C);
            std::vector<float> buf_temp(C);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H; ++h) {
                    for(size_t w=0; w<W; ++w) {
                        if (!active_mask[n]) continue;
                        size_t idx = ((n*H + h)*W + w)*C;
                        float* pixel = eyes_ptr + idx;

                        for(size_t c=0; c<C; ++c) buf[c] = pixel[c];
                        algo::WHT::fwht_1d(buf.data(), C);
                        for(size_t c=0; c<C; ++c) buf[c] *= scale_ptr[c];

                        std::copy(buf.begin(), buf.end(), buf_temp.begin());
                        for(size_t c=0; c<C; ++c) {
                            size_t prev = (c == 0) ? C - 1 : c - 1;
                            size_t next = (c == C - 1) ? 0 : c + 1;
                            float val = sp_w[0] * buf_temp[prev] + sp_w[1] * buf_temp[c] + sp_w[2] * buf_temp[next];
                            if (use_dilated_) {
                                size_t prev_d = (c < (size_t)dilation) ? C - dilation + c : c - dilation;
                                size_t next_d = (c + dilation >= C) ? c + dilation - C : c + dilation;
                                val += dp_w[0] * buf_temp[prev_d] + dp_w[1] * buf_temp[c] + dp_w[2] * buf_temp[next_d];
                            }
                            buf[c] = val;
                        }

                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(buf.data(), C);
                            float norm = 1.0f / std::sqrt(C);
                            for(size_t c=0; c<C; ++c) buf[c] *= norm;
                        }

                        for(size_t c=0; c<C; ++c) {
                            float v = buf[c] + bias_ptr[c];
                            if (v < 0) v = 0;
                            out_ptr[idx + c] = v;
                        }
                    }
                }
            }
        }
        return output;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        // ... (Optimized backward with thread locals)
        auto shape = input_cached_.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];

        Tensor<float> grad_input(shape);
        grad_input.fill(0);
        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_soft_perm_weights_.fill(0);
        grad_dilated_perm_weights_.fill(0);
        grad_bias_.fill(0);

        const float* go_ptr = grad_output.data();
        const float* eyes_ptr = eyes_out_cached_.data();
        const float* scale_ptr = spectral_scales_.data();
        const float* bias_ptr = bias_.data();
        const float* sp_w = soft_perm_weights_.data();
        const float* dp_w = dilated_perm_weights_.data();
        const float* in_ptr = input_cached_.data();
        int dilation = static_cast<int>(std::sqrt(C));

        std::vector<bool> active_mask(N, true);
        if (use_gating_) {
            const float* oracle_ptr = oracle_projection_.data();
            for(size_t n=0; n<N; ++n) {
                size_t ch = H/2, cw = W/2;
                const float* p_center = in_ptr + ((n*H + ch)*W + cw)*C;
                float dot = 0;
                for(size_t c=0; c<C; ++c) dot += p_center[c] * oracle_ptr[c];
                if (dot < 0) active_mask[n] = false;
            }
        }

        std::vector<float> acc_grad_sp(3, 0);
        std::vector<float> acc_grad_dp(3, 0);
        std::vector<float> acc_grad_scale(C, 0);
        std::vector<float> acc_grad_bias(C, 0);

        #pragma omp parallel
        {
            std::vector<float> t_grad_sp(3, 0);
            std::vector<float> t_grad_dp(3, 0);
            std::vector<float> t_grad_scale(C, 0);
            std::vector<float> t_grad_bias(C, 0);
            std::vector<float> t_grad_eyes(C);
            std::vector<float> t_grad_packed_weights(channels_ * kernel_size_ * kernel_size_, 0);

            std::vector<float> buf(C), buf_spectral(C), buf_scaled(C), dL_dPreAct(C), dL_dPerm(C), dL_dScaled(C), dL_dSpectral(C), dL_dEyes(C);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H; ++h) {
                    for(size_t w=0; w<W; ++w) {
                        if (!active_mask[n]) continue;
                        size_t idx = ((n*H + h)*W + w)*C;

                        for(size_t c=0; c<C; ++c) buf[c] = eyes_ptr[idx + c];
                        algo::WHT::fwht_1d(buf.data(), C);
                        for(size_t c=0; c<C; ++c) buf_spectral[c] = buf[c];
                        for(size_t c=0; c<C; ++c) buf[c] *= scale_ptr[c];
                        for(size_t c=0; c<C; ++c) buf_scaled[c] = buf[c];

                        for(size_t c=0; c<C; ++c) {
                            size_t prev = (c == 0) ? C - 1 : c - 1; size_t next = (c == C - 1) ? 0 : c + 1;
                            float val = sp_w[0] * buf_scaled[prev] + sp_w[1] * buf_scaled[c] + sp_w[2] * buf_scaled[next];
                            if (use_dilated_) {
                                size_t prev_d = (c < (size_t)dilation) ? C - dilation + c : c - dilation;
                                size_t next_d = (c + dilation >= C) ? c + dilation - C : c + dilation;
                                val += dp_w[0] * buf_scaled[prev_d] + dp_w[1] * buf_scaled[c] + dp_w[2] * buf_scaled[next_d];
                            }
                            buf[c] = val;
                        }

                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(buf.data(), C);
                            float norm = 1.0f / std::sqrt(C);
                            for(size_t c=0; c<C; ++c) buf[c] *= norm;
                        }

                        for(size_t c=0; c<C; ++c) {
                            float pre_act = buf[c] + bias_ptr[c];
                            float grad = go_ptr[idx+c];
                            if (pre_act < 0) grad = 0;
                            dL_dPreAct[c] = grad;
                            t_grad_bias[c] += grad;
                        }

                        for(size_t c=0; c<C; ++c) dL_dPerm[c] = dL_dPreAct[c];
                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(dL_dPerm.data(), C);
                            float norm = 1.0f / std::sqrt(C);
                            for(size_t c=0; c<C; ++c) dL_dPerm[c] *= norm;
                        }

                        std::fill(dL_dScaled.begin(), dL_dScaled.end(), 0);
                        for(size_t c=0; c<C; ++c) {
                            size_t prev = (c == 0) ? C - 1 : c - 1; size_t next = (c == C - 1) ? 0 : c + 1;
                            t_grad_sp[0] += dL_dPerm[c] * buf_scaled[prev];
                            t_grad_sp[1] += dL_dPerm[c] * buf_scaled[c];
                            t_grad_sp[2] += dL_dPerm[c] * buf_scaled[next];

                            dL_dScaled[prev] += dL_dPerm[c] * sp_w[0];
                            dL_dScaled[c] += dL_dPerm[c] * sp_w[1];
                            dL_dScaled[next] += dL_dPerm[c] * sp_w[2];

                            if (use_dilated_) {
                                size_t prev_d = (c < (size_t)dilation) ? C - dilation + c : c - dilation;
                                size_t next_d = (c + dilation >= C) ? c + dilation - C : c + dilation;
                                t_grad_dp[0] += dL_dPerm[c] * buf_scaled[prev_d];
                                t_grad_dp[1] += dL_dPerm[c] * buf_scaled[c];
                                t_grad_dp[2] += dL_dPerm[c] * buf_scaled[next_d];
                                dL_dScaled[prev_d] += dL_dPerm[c] * dp_w[0];
                                dL_dScaled[c] += dL_dPerm[c] * dp_w[1];
                                dL_dScaled[next_d] += dL_dPerm[c] * dp_w[2];
                            }
                        }

                        for(size_t c=0; c<C; ++c) {
                            t_grad_scale[c] += dL_dScaled[c] * buf_spectral[c];
                            dL_dSpectral[c] = dL_dScaled[c] * scale_ptr[c];
                        }

                        algo::WHT::fwht_1d(dL_dSpectral.data(), C);
                        for(size_t c=0; c<C; ++c) dL_dEyes[c] = dL_dSpectral[c];
                        for(size_t c=0; c<C; ++c) t_grad_eyes[c] = dL_dEyes[c];

                        int k_rad = kernel_size_ / 2;
                        for(int ky=-k_rad; ky<=k_rad; ++ky) {
                            for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                int ih = h + ky; int iw = w + kx;
                                if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    size_t in_idx_base = ((n*H + ih)*W + iw)*C;
                                    for(size_t c=0; c<C; ++c) {
                                        float inp = input_cached_.data()[in_idx_base + c];
                                        float grad = t_grad_eyes[c];
                                        size_t w_idx = c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                        t_grad_packed_weights[w_idx] += grad * inp;
                                        float w_val = packed_weights_.data()[w_idx];
                                        #pragma omp atomic
                                        grad_input.data()[in_idx_base + c] += grad * w_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            {
                for(size_t i=0; i<3; ++i) {
                    acc_grad_sp[i] += t_grad_sp[i];
                    acc_grad_dp[i] += t_grad_dp[i];
                }
                for(size_t i=0; i<C; ++i) {
                    acc_grad_scale[i] += t_grad_scale[i];
                    acc_grad_bias[i] += t_grad_bias[i];
                }
                size_t w_sz = grad_packed_weights_.size();
                float* gw_ptr = grad_packed_weights_.data();
                for(size_t i=0; i<w_sz; ++i) gw_ptr[i] += t_grad_packed_weights[i];
            }
        }

        std::copy(acc_grad_sp.begin(), acc_grad_sp.end(), grad_soft_perm_weights_.data());
        std::copy(acc_grad_dp.begin(), acc_grad_dp.end(), grad_dilated_perm_weights_.data());
        std::copy(acc_grad_scale.begin(), acc_grad_scale.end(), grad_spectral_scales_.data());
        std::copy(acc_grad_bias.begin(), acc_grad_bias.end(), grad_bias_.data());

        return grad_input;
    }

    std::vector<Tensor<float>*> parameters() override {
        return {&packed_weights_, &spectral_scales_, &soft_perm_weights_, &dilated_perm_weights_, &bias_, &oracle_projection_};
    }

    std::vector<Tensor<float>*> gradients() override {
        return {&grad_packed_weights_, &grad_spectral_scales_, &grad_soft_perm_weights_, &grad_dilated_perm_weights_, &grad_bias_};
    }

    std::string name() const override { return "ZenithBlock"; }

private:
    size_t channels_;
    size_t kernel_size_;
    size_t spectral_dim_;
    bool use_ifwht_;
    bool use_dilated_;
    bool use_gating_;

    Tensor<float> packed_weights_;
    Tensor<float> spectral_scales_;
    Tensor<float> soft_perm_weights_;
    Tensor<float> dilated_perm_weights_;
    Tensor<float> bias_;
    Tensor<float> oracle_projection_;

    Tensor<float> grad_packed_weights_;
    Tensor<float> grad_spectral_scales_;
    Tensor<float> grad_soft_perm_weights_;
    Tensor<float> grad_dilated_perm_weights_;
    Tensor<float> grad_bias_;

    Tensor<float> input_cached_;
    Tensor<float> eyes_out_cached_;
};

// -----------------------------------------------------------------------------
// ZenithBlock<int8_t>: Optimized APoT Implementation (Preserved)
// -----------------------------------------------------------------------------
template <>
class ZenithBlock<int8_t> : public Layer<int8_t> {
public:
    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                size_t arena_size = 1024*1024, bool use_gating = false)
        : channels_(channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          arena_(arena_size),
          use_gating_(use_gating),
          packed_weights_(channels * kernel_size * kernel_size),
          spectral_scales_(channels),
          bias_(channels),
          perm_indices_(channels)
    {
        // Random Init Weights
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist_code(0, 255);

        for(auto& w : packed_weights_) w = static_cast<int8_t>(dist_code(gen));
        for(auto& s : spectral_scales_) s = static_cast<int8_t>(dist_code(gen));

        std::fill(bias_.begin(), bias_.end(), 0);
        std::iota(perm_indices_.begin(), perm_indices_.end(), 0);

        oracle_projection_.resize(channels);
        for(auto& p : oracle_projection_) p = static_cast<int8_t>(dist_code(gen));
    }

#if defined(DREIDEL_ARCH_AVX2)
    // Intra-Register FWHT for 32 elements (YMM)
    static inline __m256i fwht_avx2_intra(__m256i v) {
        // Stride 1
        {
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14));
            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15));
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);

            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);

            v = _mm256_unpacklo_epi8(s, d);
        }
        // Stride 2
        {
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,1,4,5,8,9,12,13, 0,1,4,5,8,9,12,13));
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(2,3,6,7,10,11,14,15, 2,3,6,7,10,11,14,15));
            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);
            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);
            v = _mm256_unpacklo_epi16(s, d);
        }
        // Stride 4
        {
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,1,2,3,8,9,10,11, 0,1,2,3,8,9,10,11));
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(4,5,6,7,12,13,14,15, 4,5,6,7,12,13,14,15));
            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);
            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);
            v = _mm256_unpacklo_epi32(s, d);
        }
        // Stride 8
        {
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7));
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15));
            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);
            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);
            v = _mm256_unpacklo_epi64(s, d);
        }
        // Stride 16
        {
            __m128i lane0 = _mm256_castsi256_si128(v);
            __m128i lane1 = _mm256_extracti128_si256(v, 1);
            __m256i a = _mm256_broadcastsi128_si256(lane0);
            __m256i b = _mm256_broadcastsi128_si256(lane1);
            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);
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

        for(size_t n=0; n<batch; ++n) {

            if (use_gating_) {
                size_t ch = H/2, cw = W/2;
                const int8_t* p_center = in_ptr + ((n*H + ch)*W + cw)*C;

                int dist = 0;
                size_t i = 0;
#if defined(DREIDEL_ARCH_AVX2)
                for(; i + 32 <= C; i += 32) {
                    __m256i v_in = _mm256_loadu_si256((const __m256i*)(p_center + i));
                    __m256i v_proj = _mm256_loadu_si256((const __m256i*)(oracle_ptr + i));

                    uint32_t mask_in = _mm256_movemask_epi8(v_in);
                    uint32_t mask_proj = _mm256_movemask_epi8(v_proj);

                    dist += hal::AlienOps::popcnt32(mask_in ^ mask_proj);
                }
#endif
                for(; i < C; ++i) {
                    bool s1 = (p_center[i] & 0x80);
                    bool s2 = (oracle_ptr[i] & 0x80);
                    if (s1 != s2) dist++;
                }

                if (dist > 16) {
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

                            for(size_t c=0; c<C; ++c) mixer_buffer[c] = pixel_buffer[perm_ptr[c]];

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

                            size_t h_len = 32;
                            while (h_len < C) {
                                bool handled = false;
#if defined(DREIDEL_ARCH_AVX2)
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
                                if (val & 0x80) val = 0; // ReLU
                                mixer_buffer[c] = val;
                            }

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
