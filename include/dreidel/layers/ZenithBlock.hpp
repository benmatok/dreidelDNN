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

/**
 * @brief The Zenith Block (Spectral ResNet Block).
 *
 * Pipeline:
 * 1. Oracle (Gating) - Optional
 * 2. Eyes (Spatial Depthwise)
 * 3. Mixer (FWHT -> Scale -> SoftPerm -> IFWHT -> Bias -> ReLU)
 */
template <typename T>
class ZenithBlock : public Layer<T> {
public:
    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = true, bool use_gating = false)
        : channels_(channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_dilated_(use_dilated), use_gating_(use_gating),
          packed_weights_({channels, 1, kernel_size, kernel_size}), // Depthwise
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
        T stddev = std::sqrt(2.0 / (kernel_size * kernel_size * channels));
        packed_weights_.random(0, stddev);

        spectral_scales_.fill(1.0);

        // Soft Perm Init (Identity: [0, 1, 0])
        soft_perm_weights_.fill(0);
        soft_perm_weights_.data()[1] = 1.0;

        dilated_perm_weights_.fill(0);

        bias_.fill(0);

        oracle_projection_.random(-1.0, 1.0); // Random projection for gating

        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_soft_perm_weights_.fill(0);
        grad_dilated_perm_weights_.fill(0);
        grad_bias_.fill(0);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Input: (N, H, W, C)
        input_cached_ = input;
        auto shape = input.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3];

        if (C != channels_) throw std::runtime_error("Channel mismatch in ZenithBlock");

        Tensor<T> output(shape);
        T* out_ptr = output.data();
        const T* in_ptr = input.data();

        // 1. Gating (Oracle)
        std::vector<bool> active_mask(N, true);
        if (use_gating_) {
            const T* oracle_ptr = oracle_projection_.data();
            for(size_t n=0; n<N; ++n) {
                // Check center pixel
                size_t ch = H/2, cw = W/2;
                const T* p_center = in_ptr + ((n*H + ch)*W + cw)*C;
                T dot = 0;
                for(size_t c=0; c<C; ++c) dot += p_center[c] * oracle_ptr[c];
                if (dot < 0) { // Simple hyperplane check
                    active_mask[n] = false;
                }
            }
        }

        // Output buffer for Eyes
        if (eyes_out_cached_.shape() != shape) {
            eyes_out_cached_ = Tensor<T>(shape);
        }

        // 2. Eyes (Spatial Depthwise)
        int k_rad = kernel_size_ / 2;
        const T* w_ptr = packed_weights_.data();
        T* eyes_ptr = eyes_out_cached_.data();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    if (!active_mask[n]) {
                         // Zero out if gated
                         for(size_t c=0; c<C; ++c) {
                             eyes_ptr[((n*H + h)*W + w)*C + c] = 0;
                             out_ptr[((n*H + h)*W + w)*C + c] = 0;
                         }
                         continue;
                    }

                    for(size_t c=0; c<C; ++c) {
                        T val = 0;
                        for(int ky=-k_rad; ky<=k_rad; ++ky) {
                            for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                int ih = h + ky;
                                int iw = w + kx;
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
        int dilation = static_cast<int>(std::sqrt(C));

        #pragma omp parallel
        {
            std::vector<T> buf(C);
            std::vector<T> buf_temp(C);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H; ++h) {
                    for(size_t w=0; w<W; ++w) {
                        if (!active_mask[n]) continue;

                        size_t idx = ((n*H + h)*W + w)*C;
                        T* pixel = eyes_ptr + idx;

                        for(size_t c=0; c<C; ++c) buf[c] = pixel[c];

                        algo::WHT::fwht_1d(buf.data(), C);

                        for(size_t c=0; c<C; ++c) buf[c] *= scale_ptr[c];

                        std::copy(buf.begin(), buf.end(), buf_temp.begin());
                        for(size_t c=0; c<C; ++c) {
                            size_t prev = (c == 0) ? C - 1 : c - 1;
                            size_t next = (c == C - 1) ? 0 : c + 1;
                            T val = sp_w[0] * buf_temp[prev] + sp_w[1] * buf_temp[c] + sp_w[2] * buf_temp[next];

                            if (use_dilated_) {
                                size_t prev_d = (c < (size_t)dilation) ? C - dilation + c : c - dilation;
                                size_t next_d = (c + dilation >= C) ? c + dilation - C : c + dilation;
                                val += dp_w[0] * buf_temp[prev_d] + dp_w[1] * buf_temp[c] + dp_w[2] * buf_temp[next_d];
                            }
                            buf[c] = val;
                        }

                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(buf.data(), C);
                            T norm = 1.0f / std::sqrt(C);
                            for(size_t c=0; c<C; ++c) buf[c] *= norm;
                        }

                        for(size_t c=0; c<C; ++c) {
                            T v = buf[c] + bias_ptr[c];
                            if (v < 0) v = 0;
                            out_ptr[idx + c] = v;
                        }
                    }
                }
            }
        }

        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        auto shape = input_cached_.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3];

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
        int dilation = static_cast<int>(std::sqrt(C));

        // Recompute Mask
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
        std::vector<T> acc_grad_scale(C, 0);
        std::vector<T> acc_grad_bias(C, 0);

        #pragma omp parallel
        {
            std::vector<T> t_grad_sp(3, 0);
            std::vector<T> t_grad_dp(3, 0);
            std::vector<T> t_grad_scale(C, 0);
            std::vector<T> t_grad_bias(C, 0);
            std::vector<T> t_grad_eyes(C);
            // Thread-local weight gradients
            std::vector<T> t_grad_packed_weights(channels_ * kernel_size_ * kernel_size_, 0);

            std::vector<T> buf(C);
            std::vector<T> buf_spectral(C);
            std::vector<T> buf_scaled(C);
            std::vector<T> dL_dPreAct(C);
            std::vector<T> dL_dPerm(C);
            std::vector<T> dL_dScaled(C);
            std::vector<T> dL_dSpectral(C);
            std::vector<T> dL_dEyes(C);

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
                            size_t prev = (c == 0) ? C - 1 : c - 1;
                            size_t next = (c == C - 1) ? 0 : c + 1;
                            T val = sp_w[0] * buf_scaled[prev] + sp_w[1] * buf_scaled[c] + sp_w[2] * buf_scaled[next];
                            if (use_dilated_) {
                                size_t prev_d = (c < (size_t)dilation) ? C - dilation + c : c - dilation;
                                size_t next_d = (c + dilation >= C) ? c + dilation - C : c + dilation;
                                val += dp_w[0] * buf_scaled[prev_d] + dp_w[1] * buf_scaled[c] + dp_w[2] * buf_scaled[next_d];
                            }
                            buf[c] = val;
                        }

                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(buf.data(), C);
                            T norm = 1.0f / std::sqrt(C);
                            for(size_t c=0; c<C; ++c) buf[c] *= norm;
                        }

                        for(size_t c=0; c<C; ++c) {
                            T pre_act = buf[c] + bias_ptr[c];
                            T grad = go_ptr[idx+c];
                            if (pre_act < 0) grad = 0;
                            dL_dPreAct[c] = grad;
                            t_grad_bias[c] += grad;
                        }

                        for(size_t c=0; c<C; ++c) dL_dPerm[c] = dL_dPreAct[c];
                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(dL_dPerm.data(), C);
                            T norm = 1.0f / std::sqrt(C);
                            for(size_t c=0; c<C; ++c) dL_dPerm[c] *= norm;
                        }

                        std::fill(dL_dScaled.begin(), dL_dScaled.end(), 0);
                        for(size_t c=0; c<C; ++c) {
                            size_t prev = (c == 0) ? C - 1 : c - 1;
                            size_t next = (c == C - 1) ? 0 : c + 1;

                            t_grad_sp[0] += dL_dPerm[c] * buf_scaled[prev];
                            t_grad_sp[1] += dL_dPerm[c] * buf_scaled[c];
                            t_grad_sp[2] += dL_dPerm[c] * buf_scaled[next];

                            size_t p_idx = prev;
                            size_t c_idx = c;
                            size_t n_idx = next;

                            dL_dScaled[p_idx] += dL_dPerm[c] * sp_w[0];
                            dL_dScaled[c_idx] += dL_dPerm[c] * sp_w[1];
                            dL_dScaled[n_idx] += dL_dPerm[c] * sp_w[2];

                            if (use_dilated_) {
                                size_t prev_d = (c < (size_t)dilation) ? C - dilation + c : c - dilation;
                                size_t next_d = (c + dilation >= C) ? c + dilation - C : c + dilation;

                                t_grad_dp[0] += dL_dPerm[c] * buf_scaled[prev_d];
                                t_grad_dp[1] += dL_dPerm[c] * buf_scaled[c];
                                t_grad_dp[2] += dL_dPerm[c] * buf_scaled[next_d];

                                dL_dScaled[prev_d] += dL_dPerm[c] * dp_w[0];
                                dL_dScaled[c]      += dL_dPerm[c] * dp_w[1];
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
                                int ih = h + ky;
                                int iw = w + kx;
                                if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    size_t in_idx_base = ((n*H + ih)*W + iw)*C;
                                    for(size_t c=0; c<C; ++c) {
                                        T inp = input_cached_.data()[in_idx_base + c];
                                        T grad = t_grad_eyes[c];

                                        size_t w_idx = c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                        // Accumulate to thread local
                                        t_grad_packed_weights[w_idx] += grad * inp;

                                        T w_val = packed_weights_.data()[w_idx];
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
                // Accumulate weight grads
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
    size_t channels_;
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
