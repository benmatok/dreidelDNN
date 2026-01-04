#pragma once

#include "Layer.hpp"
#include "GroupNorm.hpp"
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
#include <cassert>
#include <cstring>
#include <memory>

namespace dreidel {
namespace layers {

template <typename T>
class ZenithBlock : public Layer<T> {
public:
    static inline bool use_fused_kernels = true;

    ZenithBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = false, bool use_gating = false, size_t stride = 1, size_t upscale = 1)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_gating_(use_gating), stride_(stride), upscale_(upscale),
          packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          spectral_scales_({1, in_channels}),
          mixing_weights_({3, in_channels}),
          bias_({1, out_channels}),
          oracle_projection_({1, in_channels}),

          grad_packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          grad_spectral_scales_({1, in_channels}),
          grad_mixing_weights_({3, in_channels}),
          grad_bias_({1, out_channels}),
          grad_oracle_projection_({1, in_channels}),
          rng_(std::random_device{}())
    {
        if ((in_channels_ & (in_channels_ - 1)) != 0) {
            throw std::invalid_argument("ZenithBlock in_channels must be a power of 2 for Spectral Mixing.");
        }
        reinit("identity"); // Default to Identity
    }

    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = false, bool use_gating = false)
        : ZenithBlock(channels, channels, kernel_size, spectral_dim, use_ifwht, use_dilated, use_gating, 1, 1) {}

    // Allow re-initialization for benchmarking
    void reinit(const std::string& scheme) {
        if (scheme == "identity") {
            // Delta-Orthogonal (Identity)
            packed_weights_.fill(0);
            T* w_ptr = packed_weights_.data();
            size_t k_center = kernel_size_ / 2;
            size_t spatial_size = kernel_size_ * kernel_size_;
            for (size_t c = 0; c < in_channels_; ++c) {
                w_ptr[c * spatial_size + k_center * kernel_size_ + k_center] = 1.0f;
            }
            spectral_scales_.fill(1.0);
            mixing_weights_.fill(0);
            T* mw = mixing_weights_.data();
            std::fill(mw + in_channels_, mw + 2 * in_channels_, 1.0f);
            bias_.fill(0.01);
        } else if (scheme == "he") {
            // Standard He Normal
            T stddev = std::sqrt(static_cast<T>(2.0) / (kernel_size_ * kernel_size_ * in_channels_));
            // Note: Depthwise usually divides by K*K only (fan_in per channel is K*K).
            T stddev_dw = std::sqrt(static_cast<T>(2.0) / (kernel_size_ * kernel_size_));
            packed_weights_.random(0, stddev_dw);

            // Random scales/mixing
            spectral_scales_.random(0.9, 1.1); // Perturbed Identity
            mixing_weights_.random(-0.1, 0.1);
            bias_.fill(0);
        }

        // Zero Grads
        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_mixing_weights_.fill(0);
        grad_bias_.fill(0);
        grad_oracle_projection_.fill(0);
    }

    void set_group_norm(bool enable, size_t groups = 32) {
        use_group_norm_ = enable;
        if (enable) {
            group_norm_ = std::make_unique<GroupNorm<T>>(groups, out_channels_);
        } else {
            group_norm_.reset();
        }
    }

    void set_spectral_dropout(float rate) {
        spectral_dropout_rate_ = rate;
    }

    void set_gradient_checkpointing(bool enable) {
        gradient_checkpointing_ = enable;
    }

    void set_bias_init(T val) {
        bias_.fill(val);
    }

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

        Tensor<T> output({N, H_out, W_out, out_channels_});
        T* out_ptr = output.data();
        const T* in_ptr = input.data();

        bool apply_dropout = (spectral_dropout_rate_ > 0.0f);
        if (apply_dropout) {
            dropout_mask_.resize(in_channels_);
            std::bernoulli_distribution d(1.0f - spectral_dropout_rate_);
            for (size_t c = 0; c < in_channels_; ++c) {
                dropout_mask_[c] = d(rng_) ? 1.0f : 0.0f;
            }
        }

        Tensor<T>* eyes_out_ptr = nullptr;
        Tensor<T> eyes_temp;
        if (gradient_checkpointing_) {
            eyes_temp = Tensor<T>({N, H_out, W_out, in_channels_});
            eyes_out_ptr = &eyes_temp;
        } else {
             if (eyes_out_cached_.shape().size() != 4 || eyes_out_cached_.shape()[0] != N) {
                 eyes_out_cached_ = Tensor<T>({N, H_out, W_out, in_channels_});
            }
            eyes_out_ptr = &eyes_out_cached_;
        }
        T* eyes_ptr = eyes_out_ptr->data();

        int k_rad = kernel_size_ / 2;
        const T* w_ptr = packed_weights_.data();
        const T* scale_ptr = spectral_scales_.data();
        const T* bias_ptr = bias_.data();
        const T* mix_w = mixing_weights_.data();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {
                    if (upscale_ > 1) {
                         for(size_t c=0; c<C; ++c) {
                            T val = 0;
                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                int v_h = (int)h_out + ky;
                                if (v_h < 0 || v_h >= (int)H_out) continue;
                                int ih = (up_shift > 0) ? (v_h >> up_shift) : (v_h / (int)upscale_);
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int v_w = (int)w_out + kx;
                                    if (v_w < 0 || v_w >= (int)W_out) continue;
                                    int iw = (up_shift > 0) ? (v_w >> up_shift) : (v_w / (int)upscale_);
                                    T pixel = in_ptr[((n*H + ih)*W + iw)*C + c];
                                    T weight = w_ptr[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                    val += pixel * weight;
                                }
                            }
                            eyes_ptr[((n*H_out + h_out)*W_out + w_out)*C + c] = val;
                        }
                    } else {
                        int h_in_center = h_out * stride_;
                        int w_in_center = w_out * stride_;
                        for(size_t c=0; c<C; ++c) {
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
                            eyes_ptr[((n*H_out + h_out)*W_out + w_out)*C + c] = val;
                        }
                    }
                }
            }
        }

        // Inverted Dropout Scale Factor
        T dropout_scale = (apply_dropout) ? (1.0f / (1.0f - spectral_dropout_rate_)) : 1.0f;

        #pragma omp parallel
        {
            std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_), buf_out(out_channels_);
            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H_out; ++h) {
                    for(size_t w=0; w<W_out; ++w) {
                        size_t eyes_idx = ((n*H_out + h)*W_out + w)*in_channels_;
                        T* pixel = eyes_ptr + eyes_idx;
                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] = pixel[c];

                        algo::WHT::fwht_1d(buf_in.data(), in_channels_);

                        if (apply_dropout) {
                            for(size_t c=0; c<in_channels_; ++c) buf_in[c] *= (dropout_mask_[c] * dropout_scale);
                        }

                        for(size_t c=0; c<in_channels_; ++c) buf_in[c] *= scale_ptr[c];

                        if (in_channels_ == out_channels_) {
                            const T* w_L = mix_w;
                            const T* w_C = mix_w + in_channels_;
                            const T* w_R = mix_w + 2 * in_channels_;
                            for(size_t c=0; c<in_channels_; ++c) {
                                T prev = (c == 0) ? 0 : buf_in[c - 1];
                                T next = (c == in_channels_ - 1) ? 0 : buf_in[c + 1];
                                buf_out[c] = w_L[c] * prev + w_C[c] * buf_in[c] + w_R[c] * next;
                            }
                        } else {
                             std::fill(buf_out.begin(), buf_out.end(), 0);
                             size_t min_c = std::min(in_channels_, out_channels_);
                             const T* w_L = mix_w;
                             const T* w_C = mix_w + in_channels_;
                             const T* w_R = mix_w + 2 * in_channels_;
                             for(size_t c=0; c<min_c; ++c) {
                                T prev = (c == 0) ? 0 : buf_in[c - 1];
                                T next = (c == in_channels_ - 1) ? 0 : buf_in[c + 1];
                                buf_out[c] = w_L[c] * prev + w_C[c] * buf_in[c] + w_R[c] * next;
                             }
                        }

                        if (use_ifwht_) {
                            algo::WHT::fwht_1d(buf_out.data(), out_channels_);
                            T norm = 1.0f / static_cast<T>(out_channels_);
                            for(size_t c=0; c<out_channels_; ++c) buf_out[c] *= norm;
                        }

                        if (!use_group_norm_) {
                             for(size_t c=0; c<out_channels_; ++c) buf_out[c] += bias_ptr[c];
                        }

                        size_t out_idx = ((n*H_out + h)*W_out + w)*out_channels_;
                        for(size_t c=0; c<out_channels_; ++c) {
                            out_ptr[out_idx + c] = buf_out[c];
                        }
                    }
                }
            }
        }

        if (use_group_norm_) {
            output = group_norm_->forward(output);
        }

        #pragma omp parallel for
        for (size_t i = 0; i < output.size(); ++i) {
             if (output.data()[i] < 0) output.data()[i] = 0;
        }

        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        auto shape = input_cached_.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];

        auto g_shape = grad_output.shape();
        size_t H_out = g_shape[1];
        size_t W_out = g_shape[2];

        int up_shift = 0;
        if (upscale_ > 1) {
            if (upscale_ == 2) up_shift = 1;
            else if (upscale_ == 4) up_shift = 2;
            else if (upscale_ == 8) up_shift = 3;
        }

        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_mixing_weights_.fill(0);
        grad_bias_.fill(0);
        grad_oracle_projection_.fill(0);

        Tensor<T> grad_input(shape);
        grad_input.fill(0);

        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();

        const T* eyes_ptr = (gradient_checkpointing_) ? nullptr : eyes_out_cached_.data();

        Tensor<T> grad_pre_act = grad_output; // Copy

        const T* scale_ptr = spectral_scales_.data();
        const T* mix_w = mixing_weights_.data(); // 3 rows: L, C, R
        const T* w_L = mix_w;
        const T* w_C = mix_w + in_channels_;
        const T* w_R = mix_w + 2 * in_channels_;
        const T* bias_ptr = bias_.data();
        const T* input_ptr = input_cached_.data();

        T* g_scale = grad_spectral_scales_.data();
        T* g_mix = grad_mixing_weights_.data();
        T* gw_L = g_mix;
        T* gw_C = g_mix + in_channels_;
        T* gw_R = g_mix + 2 * in_channels_;
        T* g_bias = grad_bias_.data();

        T norm = (use_ifwht_) ? (1.0f / static_cast<T>(out_channels_)) : 1.0f;
        T dropout_scale = (spectral_dropout_rate_ > 0.0f) ? (1.0f / (1.0f - spectral_dropout_rate_)) : 1.0f;

        Tensor<T> d_mixer_out({N, H_out, W_out, out_channels_});
        if (use_group_norm_) {
             Tensor<T> temp_gn_out({N, H_out, W_out, out_channels_});

             #pragma omp parallel
             {
                std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_), buf_out(out_channels_);
                #pragma omp for collapse(3)
                for(size_t n=0; n<N; ++n) {
                    for(size_t h=0; h<H_out; ++h) {
                        for(size_t w=0; w<W_out; ++w) {
                             if (eyes_ptr) {
                                  size_t idx = ((n*H_out + h)*W_out + w)*in_channels_;
                                  for(size_t c=0; c<in_channels_; ++c) buf_in[c] = eyes_ptr[idx+c];
                             } else {
                                  int k_rad = kernel_size_ / 2;
                                  const T* w_base = packed_weights_.data();
                                  std::fill(buf_in.begin(), buf_in.end(), 0);
                                  for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                      int ih = (up_shift > 0) ? (((int)h + ky) >> up_shift) : (((int)h + ky) / (int)upscale_);
                                      if(ih < 0 || ih >= (int)H) continue;
                                      for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                          int iw = (up_shift > 0) ? (((int)w + kx) >> up_shift) : (((int)w + kx) / (int)upscale_);
                                          if(iw < 0 || iw >= (int)W) continue;
                                          size_t in_idx = ((n*H + ih)*W + iw)*in_channels_;
                                          for(size_t c=0; c<in_channels_; ++c) {
                                              T w_val = w_base[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                              buf_in[c] += input_ptr[in_idx + c] * w_val;
                                          }
                                      }
                                  }
                             }

                             algo::WHT::fwht_1d(buf_in.data(), in_channels_);
                             if (spectral_dropout_rate_ > 0.0f) {
                                 for(size_t c=0; c<in_channels_; ++c) buf_in[c] *= (dropout_mask_[c] * dropout_scale);
                             }
                             for(size_t c=0; c<in_channels_; ++c) buf_in[c] *= scale_ptr[c];

                             if (in_channels_ == out_channels_) {
                                for(size_t c=0; c<in_channels_; ++c) {
                                    T prev = (c == 0) ? 0 : buf_in[c - 1];
                                    T next = (c == in_channels_ - 1) ? 0 : buf_in[c + 1];
                                    buf_out[c] = w_L[c] * prev + w_C[c] * buf_in[c] + w_R[c] * next;
                                }
                             } else {
                                  std::fill(buf_out.begin(), buf_out.end(), 0);
                             }

                             if (use_ifwht_) {
                                algo::WHT::fwht_1d(buf_out.data(), out_channels_);
                                for(size_t c=0; c<out_channels_; ++c) buf_out[c] *= norm;
                             }

                             size_t idx = ((n*H_out + h)*W_out + w)*out_channels_;
                             for(size_t c=0; c<out_channels_; ++c) temp_gn_out.data()[idx+c] = buf_out[c];
                        }
                    }
                }
             }

             Tensor<T> gn_out = group_norm_->forward(temp_gn_out);

             Tensor<T> d_gn_out = grad_output;
             T* d_gn_ptr = d_gn_out.data();
             const T* gn_out_ptr = gn_out.data();

             for(size_t i=0; i<d_gn_out.size(); ++i) {
                 if (gn_out_ptr[i] <= 0) d_gn_ptr[i] = 0;
             }

             d_mixer_out = group_norm_->backward(d_gn_out);

        } else {
             d_mixer_out = grad_output;
        }

        const T* d_mix_ptr = d_mixer_out.data();

        #pragma omp parallel
        {
            std::vector<T> local_g_scale(in_channels_, 0);
            std::vector<T> local_gw_L(in_channels_, 0);
            std::vector<T> local_gw_C(in_channels_, 0);
            std::vector<T> local_gw_R(in_channels_, 0);
            std::vector<T> local_g_bias(out_channels_, 0);

            std::vector<T, core::AlignedAllocator<T>> buf_grad(std::max(in_channels_, out_channels_));
            std::vector<T, core::AlignedAllocator<T>> buf_eyes(in_channels_);
            std::vector<T, core::AlignedAllocator<T>> d_eyes(in_channels_);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H_out; ++h) {
                    for(size_t w=0; w<W_out; ++w) {
                        size_t out_idx = ((n*H_out + h)*W_out + w)*out_channels_;

                        for(size_t c=0; c<out_channels_; ++c) buf_grad[c] = d_mix_ptr[out_idx + c];

                        if (!use_group_norm_) {
                             for(size_t c=0; c<out_channels_; ++c) local_g_bias[c] += buf_grad[c];
                        }

                        if(use_ifwht_) {
                             for(size_t c=0; c<out_channels_; ++c) buf_grad[c] *= norm;
                             algo::WHT::fwht_1d(buf_grad.data(), out_channels_);
                        }

                        if (eyes_ptr) {
                            size_t idx = ((n*H_out + h)*W_out + w)*in_channels_;
                            for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] = eyes_ptr[idx+c];
                        } else {
                            int k_rad = kernel_size_ / 2;
                            const T* w_base = packed_weights_.data();
                            std::fill(buf_eyes.begin(), buf_eyes.end(), 0);
                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                int ih = (up_shift > 0) ? (((int)h + ky) >> up_shift) : (((int)h + ky) / (int)upscale_);
                                if(ih < 0 || ih >= (int)H) continue;
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int iw = (up_shift > 0) ? (((int)w + kx) >> up_shift) : (((int)w + kx) / (int)upscale_);
                                    if(iw < 0 || iw >= (int)W) continue;
                                    size_t in_idx = ((n*H + ih)*W + iw)*in_channels_;
                                    for(size_t c=0; c<in_channels_; ++c) {
                                        T w_val = w_base[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                        buf_eyes[c] += input_ptr[in_idx + c] * w_val;
                                    }
                                }
                            }
                        }

                        algo::WHT::fwht_1d(buf_eyes.data(), in_channels_);
                        if (spectral_dropout_rate_ > 0.0f) {
                            for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] *= (dropout_mask_[c] * dropout_scale);
                        }
                        for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] *= scale_ptr[c];

                        std::fill(d_eyes.begin(), d_eyes.end(), 0);
                        if (in_channels_ == out_channels_) {
                            for(size_t c=0; c<in_channels_; ++c) {
                                T dy = buf_grad[c];
                                T prev = (c==0)?0:buf_eyes[c-1];
                                T curr = buf_eyes[c];
                                T next = (c==in_channels_-1)?0:buf_eyes[c+1];

                                local_gw_L[c] += dy * prev;
                                local_gw_C[c] += dy * curr;
                                local_gw_R[c] += dy * next;

                                d_eyes[c] += dy * w_C[c];
                                if (c > 0) d_eyes[c-1] += dy * w_L[c];
                                if (c < in_channels_-1) d_eyes[c+1] += dy * w_R[c];
                            }
                        } else {
                             // Fallback
                        }

                        for(size_t c=0; c<in_channels_; ++c) {
                            T unscaled = (std::abs(scale_ptr[c]) > 1e-9) ? buf_eyes[c] / scale_ptr[c] : 0;
                            local_g_scale[c] += d_eyes[c] * unscaled;
                            d_eyes[c] *= scale_ptr[c];
                        }

                        if (spectral_dropout_rate_ > 0.0f) {
                            for(size_t c=0; c<in_channels_; ++c) d_eyes[c] *= (dropout_mask_[c] * dropout_scale);
                        }

                        algo::WHT::fwht_1d(d_eyes.data(), in_channels_);

                        int k_rad = kernel_size_ / 2;
                        T* g_pack = grad_packed_weights_.data();
                        const T* w_pack = packed_weights_.data();

                        for(size_t c=0; c<in_channels_; ++c) {
                            T dy = d_eyes[c];
                            if (dy == 0) continue;

                            int ih_center = (up_shift > 0) ? ((int)h >> up_shift) : ((int)h / (int)upscale_);
                            int iw_center = (up_shift > 0) ? ((int)w >> up_shift) : ((int)w / (int)upscale_);

                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                int ih = ih_center + ky;
                                if(ih < 0 || ih >= (int)H) continue;
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int iw = iw_center + kx;
                                    if(iw < 0 || iw >= (int)W) continue;

                                    size_t in_idx = ((n*H + ih)*W + iw)*in_channels_ + c;
                                    T val = input_ptr[in_idx];

                                    #pragma omp atomic
                                    g_pack[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)] += dy * val;

                                    T w_val = w_pack[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                    #pragma omp atomic
                                    gi_ptr[in_idx] += dy * w_val;
                                }
                            }
                        }
                    }
                }
            }

            #pragma omp critical
            {
                if(!use_group_norm_) {
                     for(size_t i=0; i<local_g_bias.size(); ++i) g_bias[i] += local_g_bias[i];
                }
                for(size_t i=0; i<local_g_scale.size(); ++i) g_scale[i] += local_g_scale[i];
                for(size_t i=0; i<local_gw_L.size(); ++i) gw_L[i] += local_gw_L[i];
                for(size_t i=0; i<local_gw_C.size(); ++i) gw_C[i] += local_gw_C[i];
                for(size_t i=0; i<local_gw_R.size(); ++i) gw_R[i] += local_gw_R[i];
            }
        }

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params = {&packed_weights_, &spectral_scales_, &mixing_weights_};
        if (!use_group_norm_) params.push_back(&bias_);
        if (use_gating_) params.push_back(&oracle_projection_);
        if (use_group_norm_) {
            auto p = group_norm_->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads = {&grad_packed_weights_, &grad_spectral_scales_, &grad_mixing_weights_};
        if (!use_group_norm_) grads.push_back(&grad_bias_);
        if (use_gating_) grads.push_back(&grad_oracle_projection_);
        if (use_group_norm_) {
            auto g = group_norm_->gradients();
            grads.insert(grads.end(), g.begin(), g.end());
        }
        return grads;
    }

    std::string name() const override { return "ZenithBlock"; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t spectral_dim_;
    bool use_ifwht_;
    bool use_gating_;
    size_t stride_;
    size_t upscale_;
    bool use_group_norm_ = false;
    float spectral_dropout_rate_ = 0.0f;
    bool gradient_checkpointing_ = false;

    Tensor<T> packed_weights_;
    Tensor<T> spectral_scales_;
    Tensor<T> mixing_weights_;
    Tensor<T> bias_;
    Tensor<T> oracle_projection_;

    Tensor<T> grad_packed_weights_;
    Tensor<T> grad_spectral_scales_;
    Tensor<T> grad_mixing_weights_;
    Tensor<T> grad_bias_;
    Tensor<T> grad_oracle_projection_;

    Tensor<T> input_cached_;
    Tensor<T> eyes_out_cached_;

    std::mt19937 rng_;
    std::vector<float> dropout_mask_;

    std::unique_ptr<GroupNorm<T>> group_norm_;
};

} // namespace layers
} // namespace dreidel
