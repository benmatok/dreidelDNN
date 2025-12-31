#pragma once

#include "Layer.hpp"
#include "../core/Memory.hpp"
#include "../core/Allocator.hpp"
#include "../hal/ops.hpp"
#include "../algo/WHT.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <omp.h>

#ifdef __AVX2__
#include <immintrin.h>
#include "../hal/x86.hpp"
#endif

namespace dreidel {
namespace layers {

template <typename T>
class ZenithBlock : public Layer<T> {
public:
    ZenithBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = true, bool use_gating = false, size_t stride = 1, size_t upscale = 1)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_dilated_(use_dilated), use_gating_(use_gating), stride_(stride), upscale_(upscale),
          packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          spectral_scales_({1, in_channels}),
          soft_perm_weights_({1, 3}),
          dilated_perm_weights_({1, 3}),
          bias_({1, out_channels}),
          oracle_projection_({1, in_channels}),

          grad_packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          grad_spectral_scales_({1, in_channels}),
          grad_soft_perm_weights_({1, 3}),
          grad_dilated_perm_weights_({1, 3}),
          grad_bias_({1, out_channels}),
          grad_oracle_projection_({1, in_channels})
    {
        T stddev = std::sqrt(static_cast<T>(2.0) / (kernel_size * kernel_size));
        packed_weights_.random(0, stddev);

        // Initialization update: Mean 1.0, std 0.2
        spectral_scales_.random(1.0, 0.2);

        soft_perm_weights_.fill(0); soft_perm_weights_.data()[1] = 1.0;
        dilated_perm_weights_.fill(0);

        // Initialization update: Bias 0.01 to prevent dead ReLU
        bias_.fill(0.01);

        oracle_projection_.random(-1.0, 1.0);

        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_soft_perm_weights_.fill(0);
        grad_dilated_perm_weights_.fill(0);
        grad_bias_.fill(0);
        grad_oracle_projection_.fill(0);

        optimized_weights_cache_.resize(in_channels * kernel_size * kernel_size);
    }

    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = true, bool use_gating = false)
        : ZenithBlock(channels, channels, kernel_size, spectral_dim, use_ifwht, use_dilated, use_gating, 1, 1) {}


    Tensor<T> forward(const Tensor<T>& input) override {
        input_cached_ = input;
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];

        if (C != in_channels_) throw std::runtime_error("Channel mismatch in ZenithBlock");

        size_t H_out = (H + stride_ - 1) / stride_;
        size_t W_out = (W + stride_ - 1) / stride_;

        int up_shift = 0;
        if (upscale_ > 1) {
            H_out = H * upscale_;
            W_out = W * upscale_;
            if (upscale_ == 2) up_shift = 1;
            else if (upscale_ == 4) up_shift = 2;
            else if (upscale_ == 8) up_shift = 3;
        }

        output_cached_pre_relu_ = Tensor<T>({N, H_out, W_out, out_channels_});
        Tensor<T> output({N, H_out, W_out, out_channels_});
        T* out_ptr = output.data();
        T* pre_relu_ptr = output_cached_pre_relu_.data();
        const T* in_ptr = input.data();

        // Prepare Eyes Output Buffer
        eyes_out_cached_ = Tensor<T>({N, H_out, W_out, in_channels_});
        eyes_out_cached_.fill(0);
        T* eyes_ptr = eyes_out_cached_.data();

        std::vector<bool> active_mask(N, true);
        if (use_gating_) {
            const T* oracle_ptr = oracle_projection_.data();
            for(size_t n=0; n<N; ++n) {
                size_t ch = H/2, cw = W/2;
                const T* p_center = in_ptr + ((n*H + ch)*W + cw)*C;
                T dot = 0;
                for(size_t c=0; c<C; ++c) dot += p_center[c] * oracle_ptr[c];
                if (dot < 0) active_mask[n] = false;
            }
        }

        const T* scale_ptr = spectral_scales_.data();
        const T* bias_ptr = bias_.data();
        const T* sp_w = soft_perm_weights_.data();
        const T* dp_w = dilated_perm_weights_.data();
        const T* w_ptr = packed_weights_.data();
        int k_rad = kernel_size_ / 2;

        bool eyes_done = false;
        bool mixer_done = false;

#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            // Eyes Optimization
            if (in_channels_ >= 8 && in_channels_ % 8 == 0 && kernel_size_ == 3 && stride_ == 1) {
                repack_weights();

                if (upscale_ == 1) {
                    #pragma omp parallel for collapse(2)
                    for(size_t n=0; n<N; ++n) {
                        for(size_t h_out=0; h_out<H_out; ++h_out) {
                            if (!active_mask[n]) continue;

                            size_t w_out = 0;
                            // Left Boundary
                            {
                                int h_in_center = h_out;
                                int w_in_center = w_out;
                                for(size_t c=0; c<in_channels_; ++c) {
                                    float val = 0;
                                    for(int ky=-1; ky<=1; ++ky) {
                                        for(int kx=-1; kx<=1; ++kx) {
                                            int ih = h_in_center + ky;
                                            int iw = w_in_center + kx;
                                            if(ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                                float pixel = in_ptr[((n*H + ih)*W + iw)*in_channels_ + c];
                                                float weight = w_ptr[c*9 + (ky+1)*3 + (kx+1)];
                                                val += pixel * weight;
                                            }
                                        }
                                    }
                                    eyes_ptr[((n*H_out + h_out)*W_out + w_out)*in_channels_ + c] = val;
                                }
                                w_out++;
                            }
                            // Center
                            if (h_out >= 1 && h_out < H_out - 1) {
                                for (; w_out + 4 < W_out; w_out += 4) {
                                    float* out_p = eyes_ptr + ((n*H_out + h_out)*W_out + w_out)*in_channels_;
                                    const float* in_base = in_ptr + ((n*H + (h_out-1))*W + (w_out-1))*in_channels_;
                                    forward_avx2_eyes_sliding_window(in_base, out_p, W, in_channels_);
                                }
                            }
                            // Right Boundary
                            for (; w_out < W_out; ++w_out) {
                                int h_in_center = h_out;
                                int w_in_center = w_out;
                                for(size_t c=0; c<in_channels_; ++c) {
                                    float val = 0;
                                    for(int ky=-1; ky<=1; ++ky) {
                                        for(int kx=-1; kx<=1; ++kx) {
                                            int ih = h_in_center + ky;
                                            int iw = w_in_center + kx;
                                            if(ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                                float pixel = in_ptr[((n*H + ih)*W + iw)*in_channels_ + c];
                                                float weight = w_ptr[c*9 + (ky+1)*3 + (kx+1)];
                                                val += pixel * weight;
                                            }
                                        }
                                    }
                                    eyes_ptr[((n*H_out + h_out)*W_out + w_out)*in_channels_ + c] = val;
                                }
                            }
                        }
                    }
                    eyes_done = true;
                } else if (upscale_ == 4) {
                    forward_avx2_eyes_upscale(N, H, W, H_out, W_out, in_ptr, eyes_ptr, active_mask, in_channels_);
                    eyes_done = true;
                }
            }
        }
#endif

        if (!eyes_done) {
            #pragma omp parallel for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h_out=0; h_out<H_out; ++h_out) {
                    for(size_t w_out=0; w_out<W_out; ++w_out) {
                        if (!active_mask[n]) continue;

                        int h_in_center, w_in_center;
                        if (upscale_ > 1) {
                            h_in_center = (up_shift > 0) ? (h_out >> up_shift) : (h_out / upscale_);
                            w_in_center = (up_shift > 0) ? (w_out >> up_shift) : (w_out / upscale_);
                        } else {
                            h_in_center = h_out * stride_;
                            w_in_center = w_out * stride_;
                        }

                        for(size_t c=0; c<in_channels_; ++c) {
                            T val = 0;
                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                int ih = h_in_center + ky;
                                if(ih < 0 || ih >= (int)H) continue;
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int iw = w_in_center + kx;
                                    if(iw < 0 || iw >= (int)W) continue;
                                    T pixel = in_ptr[((n*H + ih)*W + iw)*C + c];
                                    T weight = w_ptr[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                    val += pixel * weight;
                                }
                            }
                            eyes_ptr[((n*H_out + h_out)*W_out + w_out)*in_channels_ + c] = val;
                        }
                    }
                }
            }
        }

#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
             if (in_channels_ == out_channels_) {
                 if (in_channels_ == 32) {
                     forward_avx2_c32_mixer(N, H_out, W_out, out_ptr, pre_relu_ptr, eyes_ptr, scale_ptr, sp_w, dp_w, bias_ptr, active_mask);
                     mixer_done = true;
                 } else if (in_channels_ == 64) {
                     forward_avx2_c64_mixer(N, H_out, W_out, out_ptr, pre_relu_ptr, eyes_ptr, scale_ptr, sp_w, dp_w, bias_ptr, active_mask);
                     mixer_done = true;
                 } else if (in_channels_ == 128) {
                     forward_avx2_c128_mixer(N, H_out, W_out, out_ptr, pre_relu_ptr, eyes_ptr, scale_ptr, sp_w, dp_w, bias_ptr, active_mask);
                     mixer_done = true;
                 } else if (in_channels_ > 128 && in_channels_ % 8 == 0) {
                     forward_avx2_generic_large_mixer(N, H_out, W_out, out_ptr, pre_relu_ptr, eyes_ptr, scale_ptr, sp_w, dp_w, bias_ptr, active_mask, in_channels_);
                     mixer_done = true;
                 }
             }
             else if (in_channels_ == 64 && out_channels_ == 1) {
                  forward_avx2_c64_to_1_mixer(N, H_out, W_out, out_ptr, pre_relu_ptr, eyes_ptr, scale_ptr, sp_w, dp_w, bias_ptr, active_mask);
                  mixer_done = true;
             }
        }
#endif

        if (!mixer_done) {
            #pragma omp parallel
            {
                std::vector<T> buf(in_channels_);
                std::vector<T> buf_out(out_channels_);
                #pragma omp for collapse(3)
                for(size_t n=0; n<N; ++n) {
                    for(size_t h=0; h<H_out; ++h) {
                        for(size_t w=0; w<W_out; ++w) {
                            size_t idx = ((n*H_out + h)*W_out + w);
                            if (!active_mask[n]) {
                                // Gated
                                for(size_t c=0; c<out_channels_; ++c) {
                                    out_ptr[idx*out_channels_ + c] = 0;
                                    pre_relu_ptr[idx*out_channels_ + c] = -1.0;
                                }
                                continue;
                            }

                            const T* e_p = eyes_ptr + idx * in_channels_;
                            T* o_p = out_ptr + idx * out_channels_;
                            T* pr_p = pre_relu_ptr + idx * out_channels_;

                            for(size_t c=0; c<in_channels_; ++c) buf[c] = e_p[c];
                            algo::WHT::fwht_1d(buf.data(), in_channels_);

                            for(size_t c=0; c<in_channels_; ++c) buf[c] *= scale_ptr[c];

                            if (in_channels_ == out_channels_) {
                                std::vector<T> temp = buf; // Copy for perm
                                for(size_t c=0; c<in_channels_; ++c) {
                                    size_t p = (c==0)?in_channels_-1:c-1;
                                    size_t nx = (c==in_channels_-1)?0:c+1;
                                    buf[c] = sp_w[0]*temp[p] + sp_w[1]*temp[c] + sp_w[2]*temp[nx];
                                }
                            }

                            if (in_channels_ != out_channels_) {
                                 std::fill(buf_out.begin(), buf_out.end(), 0);
                                 size_t min_c = std::min(in_channels_, out_channels_);
                                 for(size_t c=0; c<min_c; ++c) buf_out[c] = buf[c];
                            } else {
                                for(size_t c=0; c<in_channels_; ++c) buf_out[c] = buf[c];
                            }

                            if (use_ifwht_) {
                                algo::WHT::fwht_1d(buf_out.data(), out_channels_);
                                T inv_norm = 1.0f / static_cast<T>(out_channels_);
                                for(size_t c=0; c<out_channels_; ++c) buf_out[c] *= inv_norm;
                            }

                            for(size_t c=0; c<out_channels_; ++c) {
                                T v = buf_out[c] + bias_ptr[c];
                                pr_p[c] = v;
                                if (v < 0) v = 0;
                                o_p[c] = v;
                            }
                        }
                    }
                }
            }
        }

        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        size_t N = grad_output.shape()[0];
        size_t H_out = grad_output.shape()[1];
        size_t W_out = grad_output.shape()[2];
        size_t C_out = grad_output.shape()[3];

        size_t H_in = input_cached_.shape()[1];
        size_t W_in = input_cached_.shape()[2];

        Tensor<T> grad_input(input_cached_.shape());
        grad_input.fill(0);

        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();
        const T* pre_relu_ptr = output_cached_pre_relu_.data();
        const T* eyes_ptr = eyes_out_cached_.data();
        const T* scale_ptr = spectral_scales_.data();
        const T* in_ptr = input_cached_.data();
        const T* sp_w = soft_perm_weights_.data();

        T* g_bias_ptr = grad_bias_.data();
        T* g_scale_ptr = grad_spectral_scales_.data();
        T* g_weights_ptr = grad_packed_weights_.data();
        T* g_sp_ptr = grad_soft_perm_weights_.data();

        Tensor<T> grad_eyes({N, H_out, W_out, in_channels_});
        T* ge_ptr = grad_eyes.data();

        static bool printed_debug = false;

        #pragma omp parallel
        {
            std::vector<T> buf(out_channels_);
            std::vector<T> buf_in(in_channels_);
            std::vector<T> temp_grad(in_channels_);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H_out; ++h) {
                    for(size_t w=0; w<W_out; ++w) {
                        size_t idx = ((n*H_out + h)*W_out + w);
                        const T* go_p = go_ptr + idx * out_channels_;
                        const T* pr_p = pre_relu_ptr + idx * out_channels_;
                        T* ge_p = ge_ptr + idx * in_channels_;

                        for(size_t c=0; c<out_channels_; ++c) {
                            T d = (pr_p[c] > 0) ? go_p[c] : 0;
                            buf[c] = d;
                            #pragma omp atomic
                            g_bias_ptr[c] += d;
                        }

                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(buf.data(), out_channels_);
                            T inv_norm = 1.0f / static_cast<T>(out_channels_);
                            for(size_t c=0; c<out_channels_; ++c) buf[c] *= inv_norm;
                        }

                        std::fill(buf_in.begin(), buf_in.end(), 0);
                        size_t min_c = std::min(in_channels_, out_channels_);
                        for(size_t c=0; c<min_c; ++c) buf_in[c] = buf[c];

                        std::vector<T> x_spectral(in_channels_);
                        const T* e_p = eyes_ptr + idx * in_channels_;
                        for(size_t c=0; c<in_channels_; ++c) x_spectral[c] = e_p[c];
                        algo::WHT::fwht_1d(x_spectral.data(), in_channels_);
                        for(size_t c=0; c<in_channels_; ++c) x_spectral[c] *= scale_ptr[c];

                        if (in_channels_ == out_channels_) {
                            for(size_t c=0; c<in_channels_; ++c) temp_grad[c] = buf_in[c];

                            for(size_t c=0; c<in_channels_; ++c) {
                                size_t p = (c==0)?in_channels_-1:c-1;
                                size_t nx = (c==in_channels_-1)?0:c+1;
                                buf_in[c] = temp_grad[c] * sp_w[1] + temp_grad[nx] * sp_w[0] + temp_grad[p] * sp_w[2];

                                T dy = temp_grad[c];
                                #pragma omp atomic
                                g_sp_ptr[0] += dy * x_spectral[p];
                                #pragma omp atomic
                                g_sp_ptr[1] += dy * x_spectral[c];
                                #pragma omp atomic
                                g_sp_ptr[2] += dy * x_spectral[nx];
                            }
                        }

                        for(size_t c=0; c<in_channels_; ++c) x_spectral[c] = e_p[c];
                        algo::WHT::fwht_1d(x_spectral.data(), in_channels_);

                        for(size_t c=0; c<in_channels_; ++c) {
                             T dy = buf_in[c];
                             #pragma omp atomic
                             g_scale_ptr[c] += dy * x_spectral[c];
                             buf_in[c] = dy * scale_ptr[c];
                        }

                        algo::WHT::fwht_1d(buf_in.data(), in_channels_);

                        for(size_t c=0; c<in_channels_; ++c) ge_p[c] = buf_in[c];
                    }
                }
            }
        }

        int k_rad = kernel_size_ / 2;

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {

                    int h_in_center, w_in_center;
                    if (upscale_ > 1) {
                        h_in_center = h_out / upscale_;
                        w_in_center = w_out / upscale_;
                    } else {
                        h_in_center = h_out * stride_;
                        w_in_center = w_out * stride_;
                    }

                    for(size_t c=0; c<in_channels_; ++c) {
                        T d_val = ge_ptr[((n*H_out + h_out)*W_out + w_out)*in_channels_ + c];

                        for(int ky=-k_rad; ky<=k_rad; ++ky) {
                            int ih = h_in_center + ky;
                            if(ih < 0 || ih >= (int)H_in) continue;
                            for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                int iw = w_in_center + kx;
                                if(iw < 0 || iw >= (int)W_in) continue;

                                T pixel = in_ptr[((n*H_in + ih)*W_in + iw)*in_channels_ + c];
                                #pragma omp atomic
                                g_weights_ptr[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)] += d_val * pixel;

                                T weight = packed_weights_.data()[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                #pragma omp atomic
                                gi_ptr[((n*H_in + ih)*W_in + iw)*in_channels_ + c] += d_val * weight;
                            }
                        }
                    }
                }
            }
        }

        if (!printed_debug) {
            T bias_g_sum = 0;
            for(size_t i=0; i<grad_bias_.size(); ++i) bias_g_sum += grad_bias_.data()[i];
            T eyes_g_sum = 0;
            for(size_t i=0; i<grad_eyes.size(); ++i) eyes_g_sum += grad_eyes.data()[i];

            std::cout << "[DEBUG BACKWARD] Bias Grad Sum: " << bias_g_sum << " Eyes Grad Sum: " << eyes_g_sum << std::endl;
            printed_debug = true;
        }

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&packed_weights_, &spectral_scales_, &soft_perm_weights_, &bias_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_packed_weights_, &grad_spectral_scales_, &grad_soft_perm_weights_, &grad_bias_};
    }

    std::string name() const override { return "ZenithBlock"; }

private:
#ifdef __AVX2__
    static inline __m256 shift_right_1(__m256 curr, __m256 prev_reg) {
        __m256 t_mix = _mm256_permute2f128_ps(prev_reg, curr, 0x21);
        __m256i t_mix_i = _mm256_castps_si256(t_mix);
        __m256i curr_i = _mm256_castps_si256(curr);
        __m256i res = _mm256_alignr_epi8(curr_i, t_mix_i, 12);
        return _mm256_castsi256_ps(res);
    }

    static inline __m256 shift_left_1(__m256 curr, __m256 next_reg) {
        __m256 t_mix = _mm256_permute2f128_ps(curr, next_reg, 0x21);
        __m256i t_mix_i = _mm256_castps_si256(t_mix);
        __m256i curr_i = _mm256_castps_si256(curr);
        __m256i res = _mm256_alignr_epi8(t_mix_i, curr_i, 4);
        return _mm256_castsi256_ps(res);
    }

    void repack_weights() {
        if (packed_weights_.size() != optimized_weights_cache_.size()) {
             optimized_weights_cache_.resize(packed_weights_.size());
        }
        const float* src = packed_weights_.data();
        float* dst = optimized_weights_cache_.data();
        size_t C = in_channels_;
        size_t K = kernel_size_;
        for (size_t ky = 0; ky < K; ++ky) {
            for (size_t kx = 0; kx < K; ++kx) {
                for (size_t c = 0; c < C; ++c) {
                    dst[(ky * K + kx) * C + c] = src[c * K * K + ky * K + kx];
                }
            }
        }
    }

    void forward_avx2_eyes_sliding_window(const float* in_base, float* out_p, size_t input_stride_w, size_t C) {
        using namespace dreidel::hal::x86;
        const float* w_base = optimized_weights_cache_.data();
        for (size_t c = 0; c < C; c += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            for (int ky = 0; ky < 3; ++ky) {
                const float* row_ptr = in_base + ky * input_stride_w * C + c;
                __m256 v0 = _mm256_loadu_ps(row_ptr + 0*C);
                __m256 v1 = _mm256_loadu_ps(row_ptr + 1*C);
                __m256 v2 = _mm256_loadu_ps(row_ptr + 2*C);
                __m256 v3 = _mm256_loadu_ps(row_ptr + 3*C);
                __m256 v4 = _mm256_loadu_ps(row_ptr + 4*C);
                __m256 v5 = _mm256_loadu_ps(row_ptr + 5*C);
                const float* w_row = w_base + (ky * 3 + 0) * C + c;
                __m256 w0 = _mm256_load_ps(w_row + 0*C);
                __m256 w1 = _mm256_load_ps(w_row + 1*C);
                __m256 w2 = _mm256_load_ps(w_row + 2*C);
                acc0 = _mm256_fmadd_ps(v0, w0, acc0);
                acc0 = _mm256_fmadd_ps(v1, w1, acc0);
                acc0 = _mm256_fmadd_ps(v2, w2, acc0);
                acc1 = _mm256_fmadd_ps(v1, w0, acc1);
                acc1 = _mm256_fmadd_ps(v2, w1, acc1);
                acc1 = _mm256_fmadd_ps(v3, w2, acc1);
                acc2 = _mm256_fmadd_ps(v2, w0, acc2);
                acc2 = _mm256_fmadd_ps(v3, w1, acc2);
                acc2 = _mm256_fmadd_ps(v4, w2, acc2);
                acc3 = _mm256_fmadd_ps(v3, w0, acc3);
                acc3 = _mm256_fmadd_ps(v4, w1, acc3);
                acc3 = _mm256_fmadd_ps(v5, w2, acc3);
            }
            _mm256_storeu_ps(out_p + 0*C + c, acc0);
            _mm256_storeu_ps(out_p + 1*C + c, acc1);
            _mm256_storeu_ps(out_p + 2*C + c, acc2);
            _mm256_storeu_ps(out_p + 3*C + c, acc3);
        }
    }

    void forward_avx2_eyes_upscale(size_t N, size_t H, size_t W, size_t H_out, size_t W_out, const float* in_ptr, float* eyes_ptr, const std::vector<bool>& active_mask, size_t C) {
        using namespace dreidel::hal::x86;
        const float* w_base = optimized_weights_cache_.data();

        #pragma omp parallel for collapse(2)
        for (size_t n = 0; n < N; ++n) {
            for (size_t h_in = 0; h_in < H; ++h_in) {
                if (!active_mask[n]) continue;
                for (size_t w_in = 0; w_in < W; ++w_in) {
                    size_t out_h_base = 4 * h_in;
                    size_t out_w_base = 4 * w_in;

                    for (int dy = 0; dy < 4; ++dy) {
                        for (int dx = 0; dx < 4; ++dx) {
                            size_t oh = out_h_base + dy;
                            size_t ow = out_w_base + dx;
                            float* out_p = eyes_ptr + ((n*H_out + oh)*W_out + ow)*C;

                            for (size_t c = 0; c < C; c += 8) {
                                __m256 val = _mm256_setzero_ps();
                                for (int ky = -1; ky <= 1; ++ky) {
                                    int v_h = (int)oh + ky;
                                    if (v_h < 0 || v_h >= (int)H_out) continue;
                                    int ih = v_h >> 2;
                                    for (int kx = -1; kx <= 1; ++kx) {
                                        int v_w_idx = (int)ow + kx;
                                        if (v_w_idx < 0 || v_w_idx >= (int)W_out) continue;
                                        int iw = v_w_idx >> 2;
                                        const float* in_pixel = in_ptr + ((n*H + ih)*W + iw)*C + c;
                                        __m256 vec_in = _mm256_loadu_ps(in_pixel);
                                        const float* w_ptr = w_base + ((ky+1)*3 + (kx+1))*C + c;
                                        __m256 vec_w = _mm256_load_ps(w_ptr);
                                        val = _mm256_fmadd_ps(vec_in, vec_w, val);
                                    }
                                }
                                _mm256_storeu_ps(out_p + c, val);
                            }
                        }
                    }
                }
            }
        }
    }

    void forward_avx2_c32_mixer(size_t N, size_t H, size_t W, float* out_ptr, float* pre_relu_ptr, const float* eyes_ptr, const float* scale_ptr, const float* sp_w, const float* dp_w, const float* bias_ptr, const std::vector<bool>& active_mask) {
        using namespace dreidel::hal::x86;
        __m256 w_sp0 = _mm256_broadcast_ss(sp_w + 0), w_sp1 = _mm256_broadcast_ss(sp_w + 1), w_sp2 = _mm256_broadcast_ss(sp_w + 2);
        __m256 bias[4], scale[4];
        for(int i=0; i<4; ++i) bias[i] = _mm256_loadu_ps(bias_ptr + i*8);
        for(int i=0; i<4; ++i) scale[i] = _mm256_loadu_ps(scale_ptr + i*8);
        __m256 norm = _mm256_set1_ps(1.0f/32.0f);
        __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    size_t out_idx = ((n*H + h)*W + w)*32;
                    if (!active_mask[n]) {
                        for(int i=0; i<4; ++i) {
                            _mm256_storeu_ps(out_ptr + out_idx + i*8, zero);
                            _mm256_storeu_ps(pre_relu_ptr + out_idx + i*8, _mm256_set1_ps(-1.0f));
                        }
                        continue;
                    }

                    __m256 r[4];
                    const float* ptr = eyes_ptr + out_idx;
                    for(int i=0; i<4; ++i) r[i] = _mm256_loadu_ps(ptr + i*8);

                    fwht8_avx2(r[0]); fwht8_avx2(r[1]); fwht8_avx2(r[2]); fwht8_avx2(r[3]);
                    Ops::butterfly(r[0], r[1]); Ops::butterfly(r[2], r[3]);
                    Ops::butterfly(r[0], r[2]); Ops::butterfly(r[1], r[3]);

                    for(int i=0; i<4; ++i) r[i] = _mm256_mul_ps(r[i], scale[i]);

                    __m256 t[4]; for(int i=0; i<4; ++i) t[i] = r[i];
                    auto mix = [&](__m256& curr, __m256 prev, __m256 next) {
                        __m256 p = shift_right_1(curr, prev);
                        __m256 n = shift_left_1(curr, next);
                        __m256 res = _mm256_mul_ps(curr, w_sp1);
                        res = _mm256_fmadd_ps(p, w_sp0, res);
                        res = _mm256_fmadd_ps(n, w_sp2, res);
                        return res;
                    };
                    r[0] = mix(t[0], t[3], t[1]);
                    r[1] = mix(t[1], t[0], t[2]);
                    r[2] = mix(t[2], t[1], t[3]);
                    r[3] = mix(t[3], t[2], t[0]);

                    fwht8_avx2(r[0]); fwht8_avx2(r[1]); fwht8_avx2(r[2]); fwht8_avx2(r[3]);
                    Ops::butterfly(r[0], r[1]); Ops::butterfly(r[2], r[3]);
                    Ops::butterfly(r[0], r[2]); Ops::butterfly(r[1], r[3]);

                    for(int i=0; i<4; ++i) r[i] = _mm256_mul_ps(r[i], norm);
                    for(int i=0; i<4; ++i) {
                        r[i] = _mm256_add_ps(r[i], bias[i]);
                        _mm256_storeu_ps(pre_relu_ptr + out_idx + i*8, r[i]); // Store pre-ReLU
                        r[i] = _mm256_max_ps(r[i], zero);
                    }
                    for(int i=0; i<4; ++i) _mm256_storeu_ps(out_ptr + out_idx + i*8, r[i]);
                }
            }
        }
    }

    void forward_avx2_c64_mixer(
        size_t N, size_t H_out, size_t W_out,
        float* out_ptr, float* pre_relu_ptr, const float* eyes_ptr,
        const float* scale_ptr, const float* sp_w, const float* dp_w, const float* bias_ptr,
        const std::vector<bool>& active_mask
    ) {
        using namespace dreidel::hal::x86;
        __m256 w_sp0 = _mm256_broadcast_ss(sp_w + 0);
        __m256 w_sp1 = _mm256_broadcast_ss(sp_w + 1);
        __m256 w_sp2 = _mm256_broadcast_ss(sp_w + 2);
        __m256 bias_r0 = _mm256_loadu_ps(bias_ptr + 0);
        __m256 bias_r1 = _mm256_loadu_ps(bias_ptr + 8);
        __m256 bias_r2 = _mm256_loadu_ps(bias_ptr + 16);
        __m256 bias_r3 = _mm256_loadu_ps(bias_ptr + 24);
        __m256 bias_r4 = _mm256_loadu_ps(bias_ptr + 32);
        __m256 bias_r5 = _mm256_loadu_ps(bias_ptr + 40);
        __m256 bias_r6 = _mm256_loadu_ps(bias_ptr + 48);
        __m256 bias_r7 = _mm256_loadu_ps(bias_ptr + 56);
        __m256 scale_r0 = _mm256_loadu_ps(scale_ptr + 0);
        __m256 scale_r1 = _mm256_loadu_ps(scale_ptr + 8);
        __m256 scale_r2 = _mm256_loadu_ps(scale_ptr + 16);
        __m256 scale_r3 = _mm256_loadu_ps(scale_ptr + 24);
        __m256 scale_r4 = _mm256_loadu_ps(scale_ptr + 32);
        __m256 scale_r5 = _mm256_loadu_ps(scale_ptr + 40);
        __m256 scale_r6 = _mm256_loadu_ps(scale_ptr + 48);
        __m256 scale_r7 = _mm256_loadu_ps(scale_ptr + 56);
        __m256 norm = _mm256_set1_ps(1.0f/64.0f);
        __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H_out; ++h) {
                for(size_t w=0; w<W_out; ++w) {
                    size_t out_idx = ((n*H_out + h)*W_out + w)*64;
                    if (!active_mask[n]) {
                        for(int i=0; i<8; ++i) {
                            _mm256_storeu_ps(out_ptr + out_idx + i*8, zero);
                            _mm256_storeu_ps(pre_relu_ptr + out_idx + i*8, _mm256_set1_ps(-1.0f));
                        }
                        continue;
                    }
                    size_t eyes_idx = ((n*H_out + h)*W_out + w)*64;
                    const float* ptr = eyes_ptr + eyes_idx;
                    __m256 r0 = _mm256_loadu_ps(ptr + 0);
                    __m256 r1 = _mm256_loadu_ps(ptr + 8);
                    __m256 r2 = _mm256_loadu_ps(ptr + 16);
                    __m256 r3 = _mm256_loadu_ps(ptr + 24);
                    __m256 r4 = _mm256_loadu_ps(ptr + 32);
                    __m256 r5 = _mm256_loadu_ps(ptr + 40);
                    __m256 r6 = _mm256_loadu_ps(ptr + 48);
                    __m256 r7 = _mm256_loadu_ps(ptr + 56);
                    fwht8_avx2(r0); fwht8_avx2(r1); fwht8_avx2(r2); fwht8_avx2(r3);
                    fwht8_avx2(r4); fwht8_avx2(r5); fwht8_avx2(r6); fwht8_avx2(r7);
                    Ops::butterfly(r0, r1); Ops::butterfly(r2, r3); Ops::butterfly(r4, r5); Ops::butterfly(r6, r7);
                    Ops::butterfly(r0, r2); Ops::butterfly(r1, r3); Ops::butterfly(r4, r6); Ops::butterfly(r5, r7);
                    Ops::butterfly(r0, r4); Ops::butterfly(r1, r5); Ops::butterfly(r2, r6); Ops::butterfly(r3, r7);
                    r0 = _mm256_mul_ps(r0, scale_r0);
                    r1 = _mm256_mul_ps(r1, scale_r1);
                    r2 = _mm256_mul_ps(r2, scale_r2);
                    r3 = _mm256_mul_ps(r3, scale_r3);
                    r4 = _mm256_mul_ps(r4, scale_r4);
                    r5 = _mm256_mul_ps(r5, scale_r5);
                    r6 = _mm256_mul_ps(r6, scale_r6);
                    r7 = _mm256_mul_ps(r7, scale_r7);
                    __m256 t0 = r0, t1 = r1, t2 = r2, t3 = r3, t4 = r4, t5 = r5, t6 = r6, t7 = r7;
                    auto sp_mix = [&](__m256& curr, __m256 prev_reg, __m256 next_reg) {
                        __m256 prev = shift_right_1(curr, prev_reg);
                        __m256 next = shift_left_1(curr, next_reg);
                        __m256 res = _mm256_mul_ps(curr, w_sp1);
                        res = _mm256_fmadd_ps(prev, w_sp0, res);
                        res = _mm256_fmadd_ps(next, w_sp2, res);
                        return res;
                    };
                    r0 = sp_mix(t0, t7, t1);
                    r1 = sp_mix(t1, t0, t2);
                    r2 = sp_mix(t2, t1, t3);
                    r3 = sp_mix(t3, t2, t4);
                    r4 = sp_mix(t4, t3, t5);
                    r5 = sp_mix(t5, t4, t6);
                    r6 = sp_mix(t6, t5, t7);
                    r7 = sp_mix(t7, t6, t0);

                    fwht8_avx2(r0); fwht8_avx2(r1); fwht8_avx2(r2); fwht8_avx2(r3);
                    fwht8_avx2(r4); fwht8_avx2(r5); fwht8_avx2(r6); fwht8_avx2(r7);
                    Ops::butterfly(r0, r1); Ops::butterfly(r2, r3); Ops::butterfly(r4, r5); Ops::butterfly(r6, r7);
                    Ops::butterfly(r0, r2); Ops::butterfly(r1, r3); Ops::butterfly(r4, r6); Ops::butterfly(r5, r7);
                    Ops::butterfly(r0, r4); Ops::butterfly(r1, r5); Ops::butterfly(r2, r6); Ops::butterfly(r3, r7);
                    r0 = _mm256_mul_ps(r0, norm);
                    r1 = _mm256_mul_ps(r1, norm);
                    r2 = _mm256_mul_ps(r2, norm);
                    r3 = _mm256_mul_ps(r3, norm);
                    r4 = _mm256_mul_ps(r4, norm);
                    r5 = _mm256_mul_ps(r5, norm);
                    r6 = _mm256_mul_ps(r6, norm);
                    r7 = _mm256_mul_ps(r7, norm);
                    auto bias_relu = [&](__m256& r, __m256 b, float* pre_p) {
                        r = _mm256_add_ps(r, b);
                        _mm256_storeu_ps(pre_p, r);
                        r = _mm256_max_ps(r, zero);
                    };
                    bias_relu(r0, bias_r0, pre_relu_ptr + out_idx + 0);
                    bias_relu(r1, bias_r1, pre_relu_ptr + out_idx + 8);
                    bias_relu(r2, bias_r2, pre_relu_ptr + out_idx + 16);
                    bias_relu(r3, bias_r3, pre_relu_ptr + out_idx + 24);
                    bias_relu(r4, bias_r4, pre_relu_ptr + out_idx + 32);
                    bias_relu(r5, bias_r5, pre_relu_ptr + out_idx + 40);
                    bias_relu(r6, bias_r6, pre_relu_ptr + out_idx + 48);
                    bias_relu(r7, bias_r7, pre_relu_ptr + out_idx + 56);
                    _mm256_storeu_ps(out_ptr + out_idx + 0, r0);
                    _mm256_storeu_ps(out_ptr + out_idx + 8, r1);
                    _mm256_storeu_ps(out_ptr + out_idx + 16, r2);
                    _mm256_storeu_ps(out_ptr + out_idx + 24, r3);
                    _mm256_storeu_ps(out_ptr + out_idx + 32, r4);
                    _mm256_storeu_ps(out_ptr + out_idx + 40, r5);
                    _mm256_storeu_ps(out_ptr + out_idx + 48, r6);
                    _mm256_storeu_ps(out_ptr + out_idx + 56, r7);
                }
            }
        }
    }

    void forward_avx2_c128_mixer(size_t N, size_t H, size_t W, float* out_ptr, float* pre_relu_ptr, const float* eyes_ptr, const float* scale_ptr, const float* sp_w, const float* dp_w, const float* bias_ptr, const std::vector<bool>& active_mask) {
        using namespace dreidel::hal::x86;
        __m256 w_sp0 = _mm256_broadcast_ss(sp_w + 0), w_sp1 = _mm256_broadcast_ss(sp_w + 1), w_sp2 = _mm256_broadcast_ss(sp_w + 2);
        __m256 norm = _mm256_set1_ps(1.0f/128.0f);
        __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    size_t out_idx = ((n*H + h)*W + w)*128;
                    if (!active_mask[n]) {
                        for(int i=0; i<16; ++i) {
                            _mm256_storeu_ps(out_ptr + out_idx + i*8, zero);
                            _mm256_storeu_ps(pre_relu_ptr + out_idx + i*8, _mm256_set1_ps(-1.0f));
                        }
                        continue;
                    }

                    __m256 r[16];
                    const float* ptr = eyes_ptr + out_idx;
                    for(int i=0; i<16; ++i) r[i] = _mm256_loadu_ps(ptr + i*8);

                    alignas(32) float buf[128];
                    for(int i=0; i<16; ++i) _mm256_store_ps(buf + i*8, r[i]);
                    fwht128_avx2(buf);
                    for(int i=0; i<16; ++i) r[i] = _mm256_load_ps(buf + i*8);

                    for(int i=0; i<16; ++i) {
                        __m256 s = _mm256_loadu_ps(scale_ptr + i*8);
                        r[i] = _mm256_mul_ps(r[i], s);
                    }

                    __m256 t[16]; for(int i=0; i<16; ++i) t[i] = r[i];
                    auto mix = [&](__m256& curr, __m256 prev, __m256 next) {
                        __m256 p = shift_right_1(curr, prev);
                        __m256 n = shift_left_1(curr, next);
                        __m256 res = _mm256_mul_ps(curr, w_sp1);
                        res = _mm256_fmadd_ps(p, w_sp0, res);
                        res = _mm256_fmadd_ps(n, w_sp2, res);
                        return res;
                    };
                    for(int i=0; i<16; ++i) {
                        int p = (i==0)?15:i-1;
                        int n = (i==15)?0:i+1;
                        r[i] = mix(t[i], t[p], t[n]);
                    }

                    for(int i=0; i<16; ++i) _mm256_store_ps(buf + i*8, r[i]);
                    fwht128_avx2(buf);
                    for(int i=0; i<16; ++i) r[i] = _mm256_load_ps(buf + i*8);

                    for(int i=0; i<16; ++i) {
                        __m256 b = _mm256_loadu_ps(bias_ptr + i*8);
                        r[i] = _mm256_mul_ps(r[i], norm);
                        r[i] = _mm256_add_ps(r[i], b);
                        _mm256_storeu_ps(pre_relu_ptr + out_idx + i*8, r[i]);
                        r[i] = _mm256_max_ps(r[i], zero);
                        _mm256_storeu_ps(out_ptr + out_idx + i*8, r[i]);
                    }
                }
            }
        }
    }

    void forward_avx2_generic_large_mixer(size_t N, size_t H, size_t W, float* out_ptr, float* pre_relu_ptr, const float* eyes_ptr, const float* scale_ptr, const float* sp_w, const float* dp_w, const float* bias_ptr, const std::vector<bool>& active_mask, size_t C) {
        using namespace dreidel::hal::x86;
        __m256 w_sp0 = _mm256_broadcast_ss(sp_w + 0), w_sp1 = _mm256_broadcast_ss(sp_w + 1), w_sp2 = _mm256_broadcast_ss(sp_w + 2);
        __m256 norm = _mm256_set1_ps(1.0f/C);
        __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel
        {
            std::vector<float, core::AlignedAllocator<float>> buf(C);
            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H; ++h) {
                    for(size_t w=0; w<W; ++w) {
                        size_t idx = ((n*H + h)*W + w)*C;
                        if (!active_mask[n]) {
                            for(size_t i=0; i<C; i+=8) {
                                _mm256_storeu_ps(out_ptr + idx + i, zero);
                                _mm256_storeu_ps(pre_relu_ptr + idx + i, _mm256_set1_ps(-1.0f));
                            }
                            continue;
                        }

                        std::copy(eyes_ptr + idx, eyes_ptr + idx + C, buf.begin());
                        algo::WHT::fwht_1d(buf.data(), C);

                        for(size_t i=0; i<C; i+=8) {
                            __m256 v = _mm256_load_ps(buf.data() + i);
                            __m256 s = _mm256_loadu_ps(scale_ptr + i);
                            _mm256_store_ps(buf.data() + i, _mm256_mul_ps(v, s));
                        }

                        std::vector<float, core::AlignedAllocator<float>> temp = buf;
                        for(size_t c=0; c<C; ++c) {
                            size_t p = (c==0)?C-1:c-1;
                            size_t nx = (c==C-1)?0:c+1;
                            buf[c] = sp_w[0]*temp[p] + sp_w[1]*temp[c] + sp_w[2]*temp[nx];
                        }

                        algo::WHT::fwht_1d(buf.data(), C);

                        for(size_t i=0; i<C; i+=8) {
                            __m256 v = _mm256_load_ps(buf.data() + i);
                            __m256 b = _mm256_loadu_ps(bias_ptr + i);
                            v = _mm256_mul_ps(v, norm);
                            v = _mm256_add_ps(v, b);
                            _mm256_storeu_ps(pre_relu_ptr + idx + i, v);
                            v = _mm256_max_ps(v, zero);
                            _mm256_storeu_ps(out_ptr + idx + i, v);
                        }
                    }
                }
            }
        }
    }

    void forward_avx2_c64_to_1_mixer(
        size_t N, size_t H_out, size_t W_out,
        float* out_ptr, float* pre_relu_ptr, const float* eyes_ptr,
        const float* scale_ptr, const float* sp_w, const float* dp_w, const float* bias_ptr,
        const std::vector<bool>& active_mask
    ) {
        using namespace dreidel::hal::x86;
        float bias_val = bias_ptr[0];
        __m256 scale_r0 = _mm256_loadu_ps(scale_ptr + 0);
        __m256 scale_r1 = _mm256_loadu_ps(scale_ptr + 8);
        __m256 scale_r2 = _mm256_loadu_ps(scale_ptr + 16);
        __m256 scale_r3 = _mm256_loadu_ps(scale_ptr + 24);
        __m256 scale_r4 = _mm256_loadu_ps(scale_ptr + 32);
        __m256 scale_r5 = _mm256_loadu_ps(scale_ptr + 40);
        __m256 scale_r6 = _mm256_loadu_ps(scale_ptr + 48);
        __m256 scale_r7 = _mm256_loadu_ps(scale_ptr + 56);

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H_out; ++h) {
                for(size_t w=0; w<W_out; ++w) {
                    size_t out_idx = ((n*H_out + h)*W_out + w);
                    if (!active_mask[n]) {
                        out_ptr[out_idx] = 0.0f;
                        pre_relu_ptr[out_idx] = -1.0f;
                        continue;
                    }
                    size_t eyes_idx = ((n*H_out + h)*W_out + w)*64;
                    const float* ptr = eyes_ptr + eyes_idx;

                    __m256 r0 = _mm256_loadu_ps(ptr + 0);
                    __m256 r1 = _mm256_loadu_ps(ptr + 8);
                    __m256 r2 = _mm256_loadu_ps(ptr + 16);
                    __m256 r3 = _mm256_loadu_ps(ptr + 24);
                    __m256 r4 = _mm256_loadu_ps(ptr + 32);
                    __m256 r5 = _mm256_loadu_ps(ptr + 40);
                    __m256 r6 = _mm256_loadu_ps(ptr + 48);
                    __m256 r7 = _mm256_loadu_ps(ptr + 56);

                    fwht8_avx2(r0); fwht8_avx2(r1); fwht8_avx2(r2); fwht8_avx2(r3);
                    fwht8_avx2(r4); fwht8_avx2(r5); fwht8_avx2(r6); fwht8_avx2(r7);
                    Ops::butterfly(r0, r1); Ops::butterfly(r2, r3); Ops::butterfly(r4, r5); Ops::butterfly(r6, r7);
                    Ops::butterfly(r0, r2); Ops::butterfly(r1, r3); Ops::butterfly(r4, r6); Ops::butterfly(r5, r7);
                    Ops::butterfly(r0, r4); Ops::butterfly(r1, r5); Ops::butterfly(r2, r6); Ops::butterfly(r3, r7);

                    r0 = _mm256_mul_ps(r0, scale_r0);
                    r1 = _mm256_mul_ps(r1, scale_r1);
                    r2 = _mm256_mul_ps(r2, scale_r2);
                    r3 = _mm256_mul_ps(r3, scale_r3);
                    r4 = _mm256_mul_ps(r4, scale_r4);
                    r5 = _mm256_mul_ps(r5, scale_r5);
                    r6 = _mm256_mul_ps(r6, scale_r6);
                    r7 = _mm256_mul_ps(r7, scale_r7);

                    float buf0 = _mm256_cvtss_f32(r0);
                    __m256 r0_shuf = _mm256_permute_ps(r0, 0x01);
                    float buf1 = _mm256_cvtss_f32(r0_shuf);
                    __m256 r7_high = _mm256_permute2f128_ps(r7, r7, 0x11);
                    __m256 r7_perm = _mm256_permute_ps(r7_high, 0xFF);
                    float buf63 = _mm256_cvtss_f32(r7_perm);

                    float val = sp_w[0] * buf63 + sp_w[1] * buf0 + sp_w[2] * buf1;
                    val += bias_val;
                    pre_relu_ptr[out_idx] = val;
                    if (val < 0) val = 0;
                    out_ptr[out_idx] = val;
                }
            }
        }
    }
#endif
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t spectral_dim_;
    bool use_ifwht_;
    bool use_dilated_;
    bool use_gating_;
    size_t stride_;
    size_t upscale_;

    Tensor<T> packed_weights_;
    Tensor<T> spectral_scales_;
    Tensor<T> soft_perm_weights_;
    Tensor<T> dilated_perm_weights_;
    Tensor<T> bias_;
    Tensor<T> oracle_projection_;

    Tensor<T> grad_packed_weights_;
    Tensor<T> grad_spectral_scales_;
    Tensor<T> grad_soft_perm_weights_;
    Tensor<T> grad_dilated_perm_weights_;
    Tensor<T> grad_bias_;
    Tensor<T> grad_oracle_projection_;

    Tensor<T> input_cached_;
    Tensor<T> eyes_out_cached_;
    Tensor<T> output_cached_pre_relu_;

    std::vector<float, core::AlignedAllocator<float>> optimized_weights_cache_;
};

} // namespace layers
} // namespace dreidel
