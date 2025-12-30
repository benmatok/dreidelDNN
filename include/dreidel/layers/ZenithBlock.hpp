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

/**
 * @brief The ZenithBlock (Spectral ResNet Block).
 *
 * Pipeline:
 * 1. Oracle (Gating)
 * 2. Eyes (Spatial Depthwise) - now supports stride
 * 3. Mixer (FWHT -> Scale -> SoftPerm (Standard + Dilated) -> IFWHT -> Bias -> ReLU)
 */
template <typename T>
class ZenithBlock : public Layer<T> {
public:
    // Main constructor with explicit in/out channels and stride
    ZenithBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = true, bool use_gating = false, size_t stride = 1)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_dilated_(use_dilated), use_gating_(use_gating), stride_(stride),
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
        // Init
        T stddev = std::sqrt(static_cast<T>(2.0) / (kernel_size * kernel_size));
        packed_weights_.random(0, stddev);
        spectral_scales_.fill(1.0);
        soft_perm_weights_.fill(0); soft_perm_weights_.data()[1] = 1.0;
        dilated_perm_weights_.fill(0);
        bias_.fill(0);
        oracle_projection_.random(-1.0, 1.0);

        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_soft_perm_weights_.fill(0);
        grad_dilated_perm_weights_.fill(0);
        grad_bias_.fill(0);
        grad_oracle_projection_.fill(0);

        // Allocate cache for repacked weights
        optimized_weights_cache_.resize(in_channels * kernel_size * kernel_size);
    }

    // Compatibility constructor (in == out, stride=1)
    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = true, bool use_gating = false)
        : ZenithBlock(channels, channels, kernel_size, spectral_dim, use_ifwht, use_dilated, use_gating, 1) {}


    Tensor<T> forward(const Tensor<T>& input) override {
        input_cached_ = input;
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];

        if (C != in_channels_) throw std::runtime_error("Channel mismatch in ZenithBlock");

        size_t H_out = (H + stride_ - 1) / stride_;
        size_t W_out = (W + stride_ - 1) / stride_;

        Tensor<T> output({N, H_out, W_out, out_channels_});
        T* out_ptr = output.data();
        const T* in_ptr = input.data();

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

        if (eyes_out_cached_.shape().size() != 4 ||
            eyes_out_cached_.shape()[0] != N ||
            eyes_out_cached_.shape()[1] != H_out ||
            eyes_out_cached_.shape()[2] != W_out ||
            eyes_out_cached_.shape()[3] != in_channels_) {
             eyes_out_cached_ = Tensor<T>({N, H_out, W_out, in_channels_});
        }

        int k_rad = kernel_size_ / 2;
        const T* w_ptr = packed_weights_.data();
        T* eyes_ptr = eyes_out_cached_.data();

        // Declare pointers used by both paths
        const T* scale_ptr = spectral_scales_.data();
        const T* bias_ptr = bias_.data();
        const T* sp_w = soft_perm_weights_.data();
        const T* dp_w = dilated_perm_weights_.data();

        bool is_downsample = (out_channels_ == in_channels_ / 2);
        bool is_upsample = (out_channels_ == in_channels_ * 2);

        // --- AVX2 Optimized Path ---
#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            if (in_channels_ == 64 && out_channels_ == 64 && !is_downsample && !is_upsample && kernel_size_ == 3 && stride_ == 1) {
                repack_weights();

                #pragma omp parallel for collapse(2)
                for(size_t n=0; n<N; ++n) {
                    for(size_t h_out=0; h_out<H_out; ++h_out) {
                        if (!active_mask[n]) {
                            std::fill(eyes_ptr + ((n*H_out + h_out)*W_out)*in_channels_,
                                      eyes_ptr + ((n*H_out + h_out + 1)*W_out)*in_channels_, 0.0f);
                            continue;
                        }

                        size_t w_out = 0;

                        // Left Boundary
                        {
                            int h_in_center = h_out;
                            int w_in_center = w_out;
                            for(size_t c=0; c<64; ++c) {
                                float val = 0;
                                for(int ky=-1; ky<=1; ++ky) {
                                    for(int kx=-1; kx<=1; ++kx) {
                                        int ih = h_in_center + ky;
                                        int iw = w_in_center + kx;
                                        if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                            float pixel = in_ptr[((n*H + ih)*W + iw)*64 + c];
                                            float weight = w_ptr[c*9 + (ky+1)*3 + (kx+1)];
                                            val += pixel * weight;
                                        }
                                    }
                                }
                                eyes_ptr[((n*H_out + h_out)*W_out + w_out)*64 + c] = val;
                            }
                            w_out++;
                        }

                        // Center (Sliding Window)
                        if (h_out >= 1 && h_out < H_out - 1) {
                            for (; w_out + 4 < W_out; w_out += 4) {
                                float* out_p = eyes_ptr + ((n*H_out + h_out)*W_out + w_out)*64;
                                const float* in_base = in_ptr + ((n*H + (h_out-1))*W + (w_out-1))*64;
                                forward_avx2_eyes_sliding_window(in_base, out_p, W);
                            }
                        }

                        // Right Boundary
                        for (; w_out < W_out; ++w_out) {
                            int h_in_center = h_out;
                            int w_in_center = w_out;
                            for(size_t c=0; c<64; ++c) {
                                float val = 0;
                                for(int ky=-1; ky<=1; ++ky) {
                                    for(int kx=-1; kx<=1; ++kx) {
                                        int ih = h_in_center + ky;
                                        int iw = w_in_center + kx;
                                        if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                            float pixel = in_ptr[((n*H + ih)*W + iw)*64 + c];
                                            float weight = w_ptr[c*9 + (ky+1)*3 + (kx+1)];
                                            val += pixel * weight;
                                        }
                                    }
                                }
                                eyes_ptr[((n*H_out + h_out)*W_out + w_out)*64 + c] = val;
                            }
                        }
                    }
                }

                forward_avx2_c64_mixer(N, H_out, W_out, out_ptr, eyes_ptr, scale_ptr, sp_w, dp_w, bias_ptr, active_mask);
                return output;
            }
        }
#endif

        int dilation_in = static_cast<int>(std::sqrt(in_channels_));
        int dilation_out = static_cast<int>(std::sqrt(out_channels_));

        #pragma omp parallel
        {
            // Generic Eyes Loop
            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h_out=0; h_out<H_out; ++h_out) {
                    for(size_t w_out=0; w_out<W_out; ++w_out) {
                        if (!active_mask[n]) {
                             for(size_t c=0; c<C; ++c) {
                                 eyes_ptr[((n*H_out + h_out)*W_out + w_out)*C + c] = 0;
                             }
                             continue;
                        }

                        int h_in_center = h_out * stride_;
                        int w_in_center = w_out * stride_;

                        for(size_t c=0; c<C; ++c) {
                            T val = 0;
                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int ih = h_in_center + ky;
                                    int iw = w_in_center + kx;
                                    if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                        T pixel = in_ptr[((n*H + ih)*W + iw)*C + c];
                                        T weight = w_ptr[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                        val += pixel * weight;
                                    }
                                }
                            }
                            eyes_ptr[((n*H_out + h_out)*W_out + w_out)*C + c] = val;
                        }
                    }
                }
            }

            // Generic Mixer Loop
            std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_);
            std::vector<T, core::AlignedAllocator<T>> buf_out(out_channels_);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H_out; ++h) {
                    for(size_t w=0; w<W_out; ++w) {
                        size_t out_idx = ((n*H_out + h)*W_out + w)*out_channels_;
                        if (!active_mask[n]) {
                            for(size_t c=0; c<out_channels_; ++c) out_ptr[out_idx + c] = 0;
                            continue;
                        }

                        size_t eyes_idx = ((n*H_out + h)*W_out + w)*in_channels_;
                        T* pixel = eyes_ptr + eyes_idx;

                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] = pixel[c];
                        algo::WHT::fwht_1d(buf_in.data(), in_channels_);
                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] *= scale_ptr[c];

                        if (in_channels_ == out_channels_) {
                            for(size_t c=0; c<in_channels_; ++c) {
                                size_t prev = (c == 0) ? in_channels_ - 1 : c - 1;
                                size_t next = (c == in_channels_ - 1) ? 0 : c + 1;
                                T val = sp_w[0] * buf_in[prev] + sp_w[1] * buf_in[c] + sp_w[2] * buf_in[next];
                                if (use_dilated_) {
                                    size_t prev_d = (c < (size_t)dilation_in) ? in_channels_ - dilation_in + c : c - dilation_in;
                                    size_t next_d = (c + dilation_in >= in_channels_) ? c + dilation_in - in_channels_ : c + dilation_in;
                                    val += dp_w[0] * buf_in[prev_d] + dp_w[1] * buf_in[c] + dp_w[2] * buf_in[next_d];
                                }
                                buf_out[c] = val;
                            }
                        } else if (is_downsample) {
                             for(size_t c=0; c<out_channels_; ++c) {
                                size_t ci = c * 2;
                                size_t prev = (ci == 0) ? in_channels_ - 1 : ci - 1;
                                size_t next = (ci == in_channels_ - 1) ? 0 : ci + 1;
                                T val = sp_w[0] * buf_in[prev] + sp_w[1] * buf_in[ci] + sp_w[2] * buf_in[next];
                                if (use_dilated_) {
                                    size_t prev_d = (ci < (size_t)dilation_in) ? in_channels_ - dilation_in + ci : ci - dilation_in;
                                    size_t next_d = (ci + dilation_in >= in_channels_) ? ci + dilation_in - in_channels_ : ci + dilation_in;
                                    val += dp_w[0] * buf_in[prev_d] + dp_w[1] * buf_in[ci] + dp_w[2] * buf_in[next_d];
                                }
                                buf_out[c] = val;
                            }
                        } else if (is_upsample) {
                            std::fill(buf_out.begin(), buf_out.end(), 0);
                            for(size_t c=0; c<in_channels_; ++c) {
                                T val = buf_in[c];
                                size_t co = c * 2;
                                size_t prev = (co == 0) ? out_channels_ - 1 : co - 1;
                                size_t next = (co == out_channels_ - 1) ? 0 : co + 1;
                                buf_out[prev] += val * sp_w[0];
                                buf_out[co]   += val * sp_w[1];
                                buf_out[next] += val * sp_w[2];
                                if (use_dilated_) {
                                    size_t prev_d = (co < (size_t)dilation_out) ? out_channels_ - dilation_out + co : co - dilation_out;
                                    size_t next_d = (co + dilation_out >= out_channels_) ? co + dilation_out - out_channels_ : co + dilation_out;
                                    buf_out[prev_d] += val * dp_w[0];
                                    buf_out[co]     += val * dp_w[1];
                                    buf_out[next_d] += val * dp_w[2];
                                }
                            }
                        } else {
                            // Fallback
                            std::fill(buf_out.begin(), buf_out.end(), 0);
                            size_t min_c = std::min(in_channels_, out_channels_);
                            for(size_t c=0; c<min_c; ++c) {
                                size_t prev = (c == 0) ? in_channels_ - 1 : c - 1;
                                size_t next = (c == in_channels_ - 1) ? 0 : c + 1;
                                T val = sp_w[0] * buf_in[prev] + sp_w[1] * buf_in[c] + sp_w[2] * buf_in[next];
                                buf_out[c] = val;
                            }
                        }

                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(buf_out.data(), out_channels_);
                            T norm = 1.0f / static_cast<T>(out_channels_);
                            for(size_t c=0; c<out_channels_; ++c) buf_out[c] *= norm;
                        }

                        for(size_t c=0; c<out_channels_; ++c) {
                            T v = buf_out[c] + bias_ptr[c];
                            if (v < 0) v = 0;
                            out_ptr[out_idx + c] = v;
                        }
                    }
                }
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // ... (Full implementation from prior step, ensuring generic support) ...
        auto g_shape = grad_output.shape();
        size_t H_out = g_shape[1];
        size_t W_out = g_shape[2];
        auto shape = input_cached_.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];

        Tensor<T> grad_input(shape);
        grad_input.fill(0);
        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_soft_perm_weights_.fill(0);
        grad_dilated_perm_weights_.fill(0);
        grad_bias_.fill(0);

        const T* go_ptr = grad_output.data();
        const T* eyes_ptr = eyes_out_cached_.data();
        const T* scale_ptr = spectral_scales_.data();
        const T* bias_ptr = bias_.data();
        const T* sp_w = soft_perm_weights_.data();
        const T* dp_w = dilated_perm_weights_.data();
        const T* in_ptr = input_cached_.data();

        int dilation_in = static_cast<int>(std::sqrt(in_channels_));
        int dilation_out = static_cast<int>(std::sqrt(out_channels_));
        bool is_downsample = (out_channels_ == in_channels_ / 2);
        bool is_upsample = (out_channels_ == in_channels_ * 2);

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

        std::vector<T> acc_grad_sp(3, 0);
        std::vector<T> acc_grad_dp(3, 0);
        std::vector<T> acc_grad_scale(in_channels_, 0);
        std::vector<T> acc_grad_bias(out_channels_, 0);

        #pragma omp parallel
        {
            std::vector<T> t_grad_sp(3, 0);
            std::vector<T> t_grad_dp(3, 0);
            std::vector<T> t_grad_scale(in_channels_, 0);
            std::vector<T> t_grad_bias(out_channels_, 0);
            std::vector<T, core::AlignedAllocator<T>> t_grad_eyes(in_channels_);
            std::vector<T> t_grad_packed_weights(in_channels_ * kernel_size_ * kernel_size_, 0);

            std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_), buf_scaled(in_channels_);
            std::vector<T, core::AlignedAllocator<T>> dL_dPerm(out_channels_);
            std::vector<T> dL_dScaled(in_channels_);
            std::vector<T, core::AlignedAllocator<T>> dL_dSpectral(in_channels_);
            std::vector<T, core::AlignedAllocator<T>> dL_dEyes(in_channels_);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H_out; ++h) {
                    for(size_t w=0; w<W_out; ++w) {
                        if (!active_mask[n]) continue;

                        size_t eyes_idx = ((n*H_out + h)*W_out + w)*in_channels_;
                        size_t out_idx = ((n*H_out + h)*W_out + w)*out_channels_;

                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] = eyes_ptr[eyes_idx + c];
                        algo::WHT::fwht_1d(buf_in.data(), in_channels_);
                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] *= scale_ptr[c];
                        for(size_t c=0; c<in_channels_; ++c) buf_scaled[c] = buf_in[c];

                        for(size_t c=0; c<out_channels_; ++c) {
                            T grad = go_ptr[out_idx+c];
                            dL_dPerm[c] = grad;
                            t_grad_bias[c] += grad;
                        }

                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(dL_dPerm.data(), out_channels_);
                            T norm = 1.0f / static_cast<T>(out_channels_);
                            for(size_t c=0; c<out_channels_; ++c) dL_dPerm[c] *= norm;
                        }

                        std::fill(dL_dScaled.begin(), dL_dScaled.end(), 0);

                        if (in_channels_ == out_channels_) {
                             for(size_t c=0; c<in_channels_; ++c) {
                                size_t prev = (c == 0) ? in_channels_ - 1 : c - 1;
                                size_t next = (c == in_channels_ - 1) ? 0 : c + 1;
                                T grad = dL_dPerm[c];
                                dL_dScaled[prev] += grad * sp_w[0];
                                dL_dScaled[c]    += grad * sp_w[1];
                                dL_dScaled[next] += grad * sp_w[2];
                                t_grad_sp[0] += grad * buf_scaled[prev];
                                t_grad_sp[1] += grad * buf_scaled[c];
                                t_grad_sp[2] += grad * buf_scaled[next];

                                if (use_dilated_) {
                                    size_t prev_d = (c < (size_t)dilation_in) ? in_channels_ - dilation_in + c : c - dilation_in;
                                    size_t next_d = (c + dilation_in >= in_channels_) ? c + dilation_in - in_channels_ : c + dilation_in;
                                    dL_dScaled[prev_d] += grad * dp_w[0];
                                    dL_dScaled[c]      += grad * dp_w[1];
                                    dL_dScaled[next_d] += grad * dp_w[2];
                                    t_grad_dp[0] += grad * buf_scaled[prev_d];
                                    t_grad_dp[1] += grad * buf_scaled[c];
                                    t_grad_dp[2] += grad * buf_scaled[next_d];
                                }
                            }
                        } else if (is_downsample) {
                             for(size_t c=0; c<out_channels_; ++c) {
                                T grad = dL_dPerm[c];
                                size_t ci = c * 2;
                                size_t prev = (ci == 0) ? in_channels_ - 1 : ci - 1;
                                size_t next = (ci == in_channels_ - 1) ? 0 : ci + 1;
                                dL_dScaled[prev] += grad * sp_w[0];
                                dL_dScaled[ci]   += grad * sp_w[1];
                                dL_dScaled[next] += grad * sp_w[2];
                                t_grad_sp[0] += grad * buf_scaled[prev];
                                t_grad_sp[1] += grad * buf_scaled[ci];
                                t_grad_sp[2] += grad * buf_scaled[next];
                                if (use_dilated_) {
                                    size_t prev_d = (ci < (size_t)dilation_in) ? in_channels_ - dilation_in + ci : ci - dilation_in;
                                    size_t next_d = (ci + dilation_in >= in_channels_) ? ci + dilation_in - in_channels_ : ci + dilation_in;
                                    dL_dScaled[prev_d] += grad * dp_w[0];
                                    dL_dScaled[ci]     += grad * dp_w[1];
                                    dL_dScaled[next_d] += grad * dp_w[2];
                                    t_grad_dp[0] += grad * buf_scaled[prev_d];
                                    t_grad_dp[1] += grad * buf_scaled[ci];
                                    t_grad_dp[2] += grad * buf_scaled[next_d];
                                }
                            }
                        } else if (is_upsample) {
                            for(size_t c=0; c<in_channels_; ++c) {
                                size_t co = c * 2;
                                size_t prev = (co == 0) ? out_channels_ - 1 : co - 1;
                                size_t next = (co == out_channels_ - 1) ? 0 : co + 1;
                                T grad = 0;
                                grad += dL_dPerm[prev] * sp_w[0];
                                grad += dL_dPerm[co]   * sp_w[1];
                                grad += dL_dPerm[next] * sp_w[2];
                                t_grad_sp[0] += dL_dPerm[prev] * buf_scaled[c];
                                t_grad_sp[1] += dL_dPerm[co]   * buf_scaled[c];
                                t_grad_sp[2] += dL_dPerm[next] * buf_scaled[c];
                                if (use_dilated_) {
                                    size_t prev_d = (co < (size_t)dilation_out) ? out_channels_ - dilation_out + co : co - dilation_out;
                                    size_t next_d = (co + dilation_out >= out_channels_) ? co + dilation_out - out_channels_ : co + dilation_out;
                                    grad += dL_dPerm[prev_d] * dp_w[0];
                                    grad += dL_dPerm[co]     * dp_w[1];
                                    grad += dL_dPerm[next_d] * dp_w[2];
                                    t_grad_dp[0] += dL_dPerm[prev_d] * buf_scaled[c];
                                    t_grad_dp[1] += dL_dPerm[co]     * buf_scaled[c];
                                    t_grad_dp[2] += dL_dPerm[next_d] * buf_scaled[c];
                                }
                                dL_dScaled[c] = grad;
                            }
                        } else {
                            size_t min_c = std::min(in_channels_, out_channels_);
                            for(size_t c=0; c<min_c; ++c) {
                                size_t prev = (c == 0) ? in_channels_ - 1 : c - 1;
                                size_t next = (c == in_channels_ - 1) ? 0 : c + 1;
                                T grad = dL_dPerm[c];
                                dL_dScaled[prev] += grad * sp_w[0];
                                dL_dScaled[c]    += grad * sp_w[1];
                                dL_dScaled[next] += grad * sp_w[2];
                                t_grad_sp[0] += grad * buf_scaled[prev];
                                t_grad_sp[1] += grad * buf_scaled[c];
                                t_grad_sp[2] += grad * buf_scaled[next];
                            }
                        }

                        for(size_t c=0; c<in_channels_; ++c) {
                             T val_spectral = buf_scaled[c] / scale_ptr[c];
                             t_grad_scale[c] += dL_dScaled[c] * val_spectral;
                             dL_dSpectral[c] = dL_dScaled[c] * scale_ptr[c];
                        }

                        algo::WHT::fwht_1d(dL_dSpectral.data(), in_channels_);
                        for(size_t c=0; c<in_channels_; ++c) dL_dEyes[c] = dL_dSpectral[c];
                        for(size_t c=0; c<in_channels_; ++c) t_grad_eyes[c] = dL_dEyes[c];

                        int k_rad = kernel_size_ / 2;
                        int h_in_center = h * stride_;
                        int w_in_center = w * stride_;

                        for(int ky=-k_rad; ky<=k_rad; ++ky) {
                            for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                int ih = h_in_center + ky;
                                int iw = w_in_center + kx;
                                if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    size_t in_idx_base = ((n*H + ih)*W + iw)*in_channels_;
                                    for(size_t c=0; c<in_channels_; ++c) {
                                        T inp = input_cached_.data()[in_idx_base + c];
                                        T grad = t_grad_eyes[c];
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
                for(size_t i=0; i<in_channels_; ++i) {
                    acc_grad_scale[i] += t_grad_scale[i];
                }
                for(size_t i=0; i<out_channels_; ++i) {
                    acc_grad_bias[i] += t_grad_bias[i];
                }
                size_t w_sz = grad_packed_weights_.size();
                T* gw_ptr = grad_packed_weights_.data();
                for(size_t i=0; i<w_sz; ++i) gw_ptr[i] += t_grad_packed_weights[i];
            }
        }

        std::copy(acc_grad_sp.begin(), acc_grad_sp.end(), grad_soft_perm_weights_.data());
        std::copy(acc_grad_dp.begin(), acc_grad_dp.end(), grad_dilated_perm_weights_.data());
        std::copy(acc_grad_scale.begin(), acc_grad_scale.end(), grad_spectral_scales_.data());
        std::copy(acc_grad_bias.begin(), acc_grad_bias.end(), grad_bias_.data());

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&packed_weights_, &spectral_scales_, &soft_perm_weights_, &dilated_perm_weights_, &bias_, &oracle_projection_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_packed_weights_, &grad_spectral_scales_, &grad_soft_perm_weights_, &grad_dilated_perm_weights_, &grad_bias_, &grad_oracle_projection_};
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

    void forward_avx2_eyes_sliding_window(const float* in_base, float* out_p, size_t input_stride_w) {
        using namespace dreidel::hal::x86;

        const float* w_base = optimized_weights_cache_.data();

        for (size_t c = 0; c < 64; c += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int ky = 0; ky < 3; ++ky) {
                const float* row_ptr = in_base + ky * input_stride_w * 64 + c;

                __m256 v0 = _mm256_loadu_ps(row_ptr + 0*64);
                __m256 v1 = _mm256_loadu_ps(row_ptr + 1*64);
                __m256 v2 = _mm256_loadu_ps(row_ptr + 2*64);
                __m256 v3 = _mm256_loadu_ps(row_ptr + 3*64);
                __m256 v4 = _mm256_loadu_ps(row_ptr + 4*64);
                __m256 v5 = _mm256_loadu_ps(row_ptr + 5*64);

                const float* w_row = w_base + (ky * 3 + 0) * 64 + c;

                __m256 w0 = _mm256_load_ps(w_row + 0*64);
                __m256 w1 = _mm256_load_ps(w_row + 1*64);
                __m256 w2 = _mm256_load_ps(w_row + 2*64);

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

            _mm256_storeu_ps(out_p + 0*64 + c, acc0);
            _mm256_storeu_ps(out_p + 1*64 + c, acc1);
            _mm256_storeu_ps(out_p + 2*64 + c, acc2);
            _mm256_storeu_ps(out_p + 3*64 + c, acc3);
        }
    }

    void forward_avx2_c64_mixer(
        size_t N, size_t H_out, size_t W_out,
        float* out_ptr, const float* eyes_ptr,
        const float* scale_ptr, const float* sp_w, const float* dp_w, const float* bias_ptr,
        const std::vector<bool>& active_mask
    ) {
        using namespace dreidel::hal::x86;
        __m256 w_sp0 = _mm256_broadcast_ss(sp_w + 0);
        __m256 w_sp1 = _mm256_broadcast_ss(sp_w + 1);
        __m256 w_sp2 = _mm256_broadcast_ss(sp_w + 2);
        __m256 w_dp0 = _mm256_broadcast_ss(dp_w + 0);
        __m256 w_dp1 = _mm256_broadcast_ss(dp_w + 1);
        __m256 w_dp2 = _mm256_broadcast_ss(dp_w + 2);
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
                        _mm256_storeu_ps(out_ptr + out_idx + 0, zero);
                        _mm256_storeu_ps(out_ptr + out_idx + 8, zero);
                        _mm256_storeu_ps(out_ptr + out_idx + 16, zero);
                        _mm256_storeu_ps(out_ptr + out_idx + 24, zero);
                        _mm256_storeu_ps(out_ptr + out_idx + 32, zero);
                        _mm256_storeu_ps(out_ptr + out_idx + 40, zero);
                        _mm256_storeu_ps(out_ptr + out_idx + 48, zero);
                        _mm256_storeu_ps(out_ptr + out_idx + 56, zero);
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
                    if (use_dilated_) {
                        auto dilated_add = [&](__m256& acc, __m256 curr, __m256 prev, __m256 next) {
                            __m256 dp_res = _mm256_mul_ps(curr, w_dp1);
                            dp_res = _mm256_fmadd_ps(prev, w_dp0, dp_res);
                            dp_res = _mm256_fmadd_ps(next, w_dp2, dp_res);
                            acc = _mm256_add_ps(acc, dp_res);
                        };
                        dilated_add(r0, t0, t7, t1);
                        dilated_add(r1, t1, t0, t2);
                        dilated_add(r2, t2, t1, t3);
                        dilated_add(r3, t3, t2, t4);
                        dilated_add(r4, t4, t3, t5);
                        dilated_add(r5, t5, t4, t6);
                        dilated_add(r6, t6, t5, t7);
                        dilated_add(r7, t7, t6, t0);
                    }
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
                    auto bias_relu = [&](__m256& r, __m256 b) {
                        r = _mm256_add_ps(r, b);
                        r = _mm256_max_ps(r, zero);
                    };
                    bias_relu(r0, bias_r0);
                    bias_relu(r1, bias_r1);
                    bias_relu(r2, bias_r2);
                    bias_relu(r3, bias_r3);
                    bias_relu(r4, bias_r4);
                    bias_relu(r5, bias_r5);
                    bias_relu(r6, bias_r6);
                    bias_relu(r7, bias_r7);
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
#endif

    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t spectral_dim_;
    bool use_ifwht_;
    bool use_dilated_;
    bool use_gating_;
    size_t stride_;

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

    // Cache for optimized weight layout
    std::vector<float, core::AlignedAllocator<float>> optimized_weights_cache_;
};

} // namespace layers
} // namespace dreidel
