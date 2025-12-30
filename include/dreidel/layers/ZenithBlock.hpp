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

/**
 * @brief The ZenithBlock (Spectral ResNet Block).
 *
 * Pipeline:
 * 1. Oracle (Gating)
 * 2. Eyes (Spatial Depthwise)
 * 3. Mixer (FWHT -> Scale -> SoftPerm (Standard + Dilated) -> IFWHT -> Bias -> ReLU)
 */
template <typename T>
class ZenithBlock : public Layer<T> {
public:
    // Main constructor with explicit in/out channels
    ZenithBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = true, bool use_gating = false)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_dilated_(use_dilated), use_gating_(use_gating),
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
          grad_bias_({1, out_channels})
    {
        // Init
        T stddev = std::sqrt(static_cast<T>(2.0) / (kernel_size * kernel_size * in_channels));
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
    }

    // Compatibility constructor (in == out)
    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = true, bool use_gating = false)
        : ZenithBlock(channels, channels, kernel_size, spectral_dim, use_ifwht, use_dilated, use_gating) {}


    Tensor<T> forward(const Tensor<T>& input) override {
        input_cached_ = input;
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];

        if (C != in_channels_) throw std::runtime_error("Channel mismatch in ZenithBlock");

        Tensor<T> output({N, H, W, out_channels_});
        T* out_ptr = output.data();
        const T* in_ptr = input.data();

        // 1. Gating
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

        if (eyes_out_cached_.shape() != shape) eyes_out_cached_ = Tensor<T>(shape);

        // 2. Eyes (Spatial on In Channels)
        int k_rad = kernel_size_ / 2;
        const T* w_ptr = packed_weights_.data();
        T* eyes_ptr = eyes_out_cached_.data();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    if (!active_mask[n]) {
                         for(size_t c=0; c<C; ++c) {
                             eyes_ptr[((n*H + h)*W + w)*C + c] = 0;
                         }
                         // Output will be zeroed in Mixer loop
                         continue;
                    }
                    for(size_t c=0; c<C; ++c) {
                        T val = 0;
                        for(int ky=-k_rad; ky<=k_rad; ++ky) {
                            for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                int ih = h + ky; int iw = w + kx;
                                if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    T pixel = in_ptr[((n*H + ih)*W + iw)*C + c];
                                    T weight = w_ptr[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
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
        const T* scale_ptr = spectral_scales_.data();
        const T* bias_ptr = bias_.data();
        const T* sp_w = soft_perm_weights_.data();
        const T* dp_w = dilated_perm_weights_.data();

        int dilation_in = static_cast<int>(std::sqrt(in_channels_));
        int dilation_out = static_cast<int>(std::sqrt(out_channels_));

        bool is_downsample = (out_channels_ == in_channels_ / 2);
        bool is_upsample = (out_channels_ == in_channels_ * 2);

        #pragma omp parallel
        {
            std::vector<T> buf_in(in_channels_);
            std::vector<T> buf_out(out_channels_);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H; ++h) {
                    for(size_t w=0; w<W; ++w) {
                        size_t out_idx = ((n*H + h)*W + w)*out_channels_;
                        if (!active_mask[n]) {
                            for(size_t c=0; c<out_channels_; ++c) out_ptr[out_idx + c] = 0;
                            continue;
                        }

                        size_t eyes_idx = ((n*H + h)*W + w)*in_channels_;
                        T* pixel = eyes_ptr + eyes_idx;

                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] = pixel[c];
                        algo::WHT::fwht_1d(buf_in.data(), in_channels_);
                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] *= scale_ptr[c];

                        // Soft Permutation (Resize)
                        if (is_downsample) {
                            // Strided Conv (Stride 2)
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
                            // Transpose Conv (Stride 2)
                            std::fill(buf_out.begin(), buf_out.end(), 0);
                            for(size_t c=0; c<in_channels_; ++c) {
                                T val = buf_in[c];
                                size_t co = c * 2;

                                size_t prev = (co == 0) ? out_channels_ - 1 : co - 1;
                                size_t next = (co == out_channels_ - 1) ? 0 : co + 1;

                                buf_out[prev] += val * sp_w[0]; // Left
                                buf_out[co]   += val * sp_w[1]; // Center
                                buf_out[next] += val * sp_w[2]; // Right

                                if (use_dilated_) {
                                    size_t prev_d = (co < (size_t)dilation_out) ? out_channels_ - dilation_out + co : co - dilation_out;
                                    size_t next_d = (co + dilation_out >= out_channels_) ? co + dilation_out - out_channels_ : co + dilation_out;
                                    buf_out[prev_d] += val * dp_w[0];
                                    buf_out[co]     += val * dp_w[1]; // Center (Dilated) usually adds to center too?
                                    // In standard code: val += dp_w[1] * buf_in[c]. So yes.
                                    buf_out[next_d] += val * dp_w[2];
                                }
                            }
                        } else {
                            // Identity size
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
        auto shape = input_cached_.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3]; // C is in_channels

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
            std::vector<T> t_grad_eyes(in_channels_);
            std::vector<T> t_grad_packed_weights(in_channels_ * kernel_size_ * kernel_size_, 0);

            std::vector<T> buf_in(in_channels_), buf_scaled(in_channels_);
            std::vector<T> dL_dPerm(out_channels_); // Gradient coming from IFWHT
            std::vector<T> dL_dScaled(in_channels_); // Gradient w.r.t scaled input
            std::vector<T> dL_dSpectral(in_channels_); // Gradient w.r.t FWHT output
            std::vector<T> dL_dEyes(in_channels_);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H; ++h) {
                    for(size_t w=0; w<W; ++w) {
                        if (!active_mask[n]) continue;

                        size_t eyes_idx = ((n*H + h)*W + w)*in_channels_;
                        size_t out_idx = ((n*H + h)*W + w)*out_channels_;

                        // Recompute forward pass partially to get activation for gradients
                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] = eyes_ptr[eyes_idx + c];
                        algo::WHT::fwht_1d(buf_in.data(), in_channels_);
                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] *= scale_ptr[c];
                        // buf_in is now 'scaled'
                        for(size_t c=0; c<in_channels_; ++c) buf_scaled[c] = buf_in[c];

                        // Recompute dL_dPerm (which is dL/d(IFWHT_input))
                        // dL/dOutput * Activation' * IFWHT
                        // We need intermediate buf_out to check ReLU?
                        // Actually: Output = ReLU(IFWHT(buf_out) + Bias)
                        // Gradient flows through ReLU, then Bias, then IFWHT.

                        // We need the PRE-RELU output.
                        // Ideally we should cache it, but to save memory we recompute.
                        // Or we check output > 0? output = max(0, pre). If output > 0, grad passes.
                        // If output == 0, pre <= 0.
                        // So we can use output data.

                        // But wait, if output is 0, we don't know the exact pre value, but we know grad is 0.
                        // So checking go_ptr[c] is not enough if we need correct mask?
                        // Standard ReLU backward: grad = (output > 0) ? grad_out : 0.

                        // Wait, if we use output to determine mask, it's fine.
                        // But we don't have output here, we have grad_output.
                        // We need to recompute output or use output tensor if we had it.
                        // We don't pass output tensor to backward.
                        // So recompute pre-act.

                        // We need full forward path recompute for SoftPerm part to get `buf_out`.
                        // Let's do it.
                        std::vector<T> buf_out(out_channels_);
                         if (is_downsample) {
                            for(size_t c=0; c<out_channels_; ++c) {
                                size_t ci = c * 2;
                                size_t prev = (ci == 0) ? in_channels_ - 1 : ci - 1;
                                size_t next = (ci == in_channels_ - 1) ? 0 : ci + 1;
                                T val = sp_w[0] * buf_scaled[prev] + sp_w[1] * buf_scaled[ci] + sp_w[2] * buf_scaled[next];
                                if (use_dilated_) {
                                    size_t prev_d = (ci < (size_t)dilation_in) ? in_channels_ - dilation_in + ci : ci - dilation_in;
                                    size_t next_d = (ci + dilation_in >= in_channels_) ? ci + dilation_in - in_channels_ : ci + dilation_in;
                                    val += dp_w[0] * buf_scaled[prev_d] + dp_w[1] * buf_scaled[ci] + dp_w[2] * buf_scaled[next_d];
                                }
                                buf_out[c] = val;
                            }
                        } else if (is_upsample) {
                            std::fill(buf_out.begin(), buf_out.end(), 0);
                            for(size_t c=0; c<in_channels_; ++c) {
                                T val = buf_scaled[c];
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
                            for(size_t c=0; c<in_channels_; ++c) {
                                size_t prev = (c == 0) ? in_channels_ - 1 : c - 1;
                                size_t next = (c == in_channels_ - 1) ? 0 : c + 1;
                                T val = sp_w[0] * buf_scaled[prev] + sp_w[1] * buf_scaled[c] + sp_w[2] * buf_scaled[next];
                                if (use_dilated_) {
                                    size_t prev_d = (c < (size_t)dilation_in) ? in_channels_ - dilation_in + c : c - dilation_in;
                                    size_t next_d = (c + dilation_in >= in_channels_) ? c + dilation_in - in_channels_ : c + dilation_in;
                                    val += dp_w[0] * buf_scaled[prev_d] + dp_w[1] * buf_scaled[c] + dp_w[2] * buf_scaled[next_d];
                                }
                                buf_out[c] = val; // buf_out is size in_channels here
                            }
                        }

                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(buf_out.data(), out_channels_);
                            T norm = 1.0f / static_cast<T>(out_channels_);
                            for(size_t c=0; c<out_channels_; ++c) buf_out[c] *= norm;
                        }

                        // Now calc grad through ReLU + Bias
                        for(size_t c=0; c<out_channels_; ++c) {
                            T pre_act = buf_out[c] + bias_ptr[c];
                            T grad = go_ptr[out_idx+c];
                            if (pre_act < 0) grad = 0;
                            dL_dPerm[c] = grad;
                            t_grad_bias[c] += grad;
                        }

                        // Backprop IFWHT
                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(dL_dPerm.data(), out_channels_);
                            T norm = 1.0f / static_cast<T>(out_channels_);
                            for(size_t c=0; c<out_channels_; ++c) dL_dPerm[c] *= norm;
                        }

                        // Backprop SoftPerm (dL_dPerm -> dL_dScaled)
                        std::fill(dL_dScaled.begin(), dL_dScaled.end(), 0);

                        if (is_downsample) {
                            // Forward: Downsample (Strided Gather)
                            // Backward: Upsample (Strided Scatter)
                            for(size_t c=0; c<out_channels_; ++c) {
                                // dL_dPerm[c] contributes to prev, ci, next in dL_dScaled
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
                            // Forward: Upsample (Strided Scatter)
                            // Backward: Downsample (Strided Gather)
                            for(size_t c=0; c<in_channels_; ++c) {
                                // We are gathering gradient from dL_dPerm (out) to dL_dScaled (in)
                                // Forward: buf_out[prev] += val * sp_w[0]
                                // So dL_dScaled[c] += dL_dPerm[prev] * sp_w[0]
                                size_t co = c * 2;
                                size_t prev = (co == 0) ? out_channels_ - 1 : co - 1;
                                size_t next = (co == out_channels_ - 1) ? 0 : co + 1;

                                T grad = 0;
                                grad += dL_dPerm[prev] * sp_w[0];
                                grad += dL_dPerm[co]   * sp_w[1];
                                grad += dL_dPerm[next] * sp_w[2];

                                // Weights gradient: dL/dw = dL/dy * dy/dw
                                // dy[prev]/dw0 = val. So dL/dw0 += dL_dPerm[prev] * val.
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
                            // Identity
                             for(size_t c=0; c<in_channels_; ++c) {
                                size_t prev = (c == 0) ? in_channels_ - 1 : c - 1;
                                size_t next = (c == in_channels_ - 1) ? 0 : c + 1;

                                // Backward conv
                                // dL_dScaled[prev] += dL_dPerm[c] * sp_w[0];
                                // ...
                                // Easier: dL_dScaled[c] += dL_dPerm[next] * sp_w[0] (since next of someone is c)
                                // Let's use Scatter accumulation like Forward
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
                        }

                        for(size_t c=0; c<in_channels_; ++c) {
                            // Scale
                            t_grad_scale[c] += dL_dScaled[c] * (buf_in[c] / scale_ptr[c]); // buf_in was scaled, need unscaled.
                            // Better: we computed buf_scaled.
                            // buf_spectral (from forward FWHT) = buf_in / scale_ptr? No.
                            // Forward: FWHT -> buf_in. Scale -> buf_in * scale.
                            // So unscaled is what we want.
                            // We need to store 'buf_spectral' in forward pass (recompute).
                            // Let's grab it from partial recompute above?
                            // `buf_in` is currently scaled.
                            // Unscaled = buf_in / scale. Or better, just store it before scaling.
                            // I overwrote it.
                            // Let's just divide.
                             T val_spectral = buf_scaled[c] / scale_ptr[c]; // Assuming non-zero scale.
                             t_grad_scale[c] += dL_dScaled[c] * val_spectral;
                             dL_dSpectral[c] = dL_dScaled[c] * scale_ptr[c];
                        }

                        algo::WHT::fwht_1d(dL_dSpectral.data(), in_channels_);
                        for(size_t c=0; c<in_channels_; ++c) dL_dEyes[c] = dL_dSpectral[c];
                        for(size_t c=0; c<in_channels_; ++c) t_grad_eyes[c] = dL_dEyes[c];

                        int k_rad = kernel_size_ / 2;
                        for(int ky=-k_rad; ky<=k_rad; ++ky) {
                            for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                int ih = h + ky; int iw = w + kx;
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
        return {&grad_packed_weights_, &grad_spectral_scales_, &grad_soft_perm_weights_, &grad_dilated_perm_weights_, &grad_bias_};
    }

    std::string name() const override { return "ZenithBlock"; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t spectral_dim_;
    bool use_ifwht_;
    bool use_dilated_;
    bool use_gating_;

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

    Tensor<T> input_cached_;
    Tensor<T> eyes_out_cached_;
};

} // namespace layers
} // namespace dreidel
