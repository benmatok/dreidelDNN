#pragma once

#include "Layer.hpp"
#include "GroupNorm.hpp"
#include "../core/Memory.hpp"
#include "../core/Allocator.hpp"
#include "../hal/ops.hpp"
#include "../hal/x86.hpp"
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
                bool use_ifwht = true, bool use_dilated = false, bool use_gating = false, size_t stride = 1, size_t upscale = 1,
                const std::string& init_scheme = "identity")
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_gating_(use_gating), stride_(stride), upscale_(upscale),
          packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          spectral_scales_({1, in_channels}),
          mixing_weights_({3, in_channels}),
          oracle_projection_({1, in_channels}),

          grad_packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          grad_spectral_scales_({1, in_channels}),
          grad_mixing_weights_({3, in_channels}),
          grad_oracle_projection_({1, in_channels}),
          rng_(std::random_device{}())
    {
        if ((in_channels_ & (in_channels_ - 1)) != 0) {
            throw std::invalid_argument("ZenithBlock in_channels must be a power of 2 for Spectral Mixing.");
        }

        initialize(init_scheme);

        // Gating
        oracle_projection_.random(-1.0, 1.0);

        // Zero Grads
        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_mixing_weights_.fill(0);
        grad_oracle_projection_.fill(0);

        // 2. GroupNorm (Hardcoded Standard)
        size_t groups = 32;
        if (out_channels_ % groups != 0) groups = 1;
        group_norm_ = std::make_unique<GroupNorm<T>>(groups, out_channels_);
    }

    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = false, bool use_gating = false)
        : ZenithBlock(channels, channels, kernel_size, spectral_dim, use_ifwht, use_dilated, use_gating, 1, 1) {}

    void initialize(const std::string& scheme) {
        // 1. Packed Weights (Eyes)
        packed_weights_.fill(0);
        if (scheme == "identity") {
            // Delta-Orthogonal
            T* w_ptr = packed_weights_.data();
            size_t k_center = kernel_size_ / 2;
            size_t spatial_size = kernel_size_ * kernel_size_;
            for (size_t c = 0; c < in_channels_; ++c) {
                w_ptr[c * spatial_size + k_center * kernel_size_ + k_center] = 1.0f;
            }
        } else if (scheme == "he") {
            // Kaiming / He
            T stddev = std::sqrt(2.0f / (in_channels_ * kernel_size_ * kernel_size_));
            packed_weights_.random(0, stddev);
        } else {
            // Random small
            packed_weights_.random(-0.01, 0.01);
        }

        // 2. Spectral Scales & Mixer
        // For Identity, we use spectral identity (scale=1, mix=1)
        // For He, we still want spectral path to be open, but maybe we add noise?
        // Standard Zenith implementation uses Identity for the mixing path initially to pass gradients.

        T norm_factor = (use_ifwht_) ? (1.0f / static_cast<T>(out_channels_)) : 1.0f;
        spectral_scales_.fill(1.0f * norm_factor);

        mixing_weights_.fill(0);
        T* mw = mixing_weights_.data();
        // Center band = 1.0 (Identity mixing)
        std::fill(mw + in_channels_, mw + 2 * in_channels_, 1.0f);

        if (scheme == "he" || scheme == "random") {
             // Maybe add noise to mixing weights if requested, but "Identity Mix" is structural.
             // We keep Mixer as Identity for stability, only vary Eyes initialization.
        }
    }

    void set_spectral_dropout(float rate) {
        spectral_dropout_rate_ = rate;
    }

    void set_training(bool training) {
        training_ = training;
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

        // 3. Spectral Dropout (Only in Training)
        bool apply_dropout = training_ && (spectral_dropout_rate_ > 0.0f);
        if (apply_dropout) {
            dropout_mask_.resize(in_channels_);
            std::bernoulli_distribution d(1.0f - spectral_dropout_rate_);
            for (size_t c = 0; c < in_channels_; ++c) {
                dropout_mask_[c] = d(rng_) ? 1.0f : 0.0f;
            }
        }

        if (eyes_out_cached_.shape().size() != 4 || eyes_out_cached_.shape()[0] != N) {
             eyes_out_cached_ = Tensor<T>({N, H_out, W_out, in_channels_});
        }
        T* eyes_ptr = eyes_out_cached_.data();

        int k_rad = kernel_size_ / 2;
        const T* w_ptr = packed_weights_.data();
        const T* scale_ptr = spectral_scales_.data();
        const T* mix_w = mixing_weights_.data();

        // Inverted Dropout Scale
        T dropout_scale = (apply_dropout) ? (1.0f / (1.0f - spectral_dropout_rate_)) : 1.0f;

        // Fused Step 1 (Eyes/Depthwise) & Step 2 (Mixer) & ReLU
        #pragma omp parallel
        {
            std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_), buf_out(out_channels_);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h_out=0; h_out<H_out; ++h_out) {
                    for(size_t w_out=0; w_out<W_out; ++w_out) {
                        // --- Part 1: Eyes (Depthwise) ---
                        // Compute and write to eyes_out_cached_ (needed for backward)
                        // Also keep in buf_in for immediate mixing (L1 cache locality)

                        size_t eyes_idx = ((n*H_out + h_out)*W_out + w_out)*in_channels_;
                        T* eyes_store_ptr = eyes_ptr + eyes_idx;

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
                                eyes_store_ptr[c] = val;
                                buf_in[c] = val;
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
                                eyes_store_ptr[c] = val;
                                buf_in[c] = val;
                            }
                        }

                        // --- Part 2: Mixer (Spectral) ---
                        // buf_in already has the data.

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
                        }

                        size_t out_idx = ((n*H_out + h_out)*W_out + w_out)*out_channels_;

                        // --- Part 3: ReLU ---
                        for(size_t c=0; c<out_channels_; ++c) {
                            T v = buf_out[c];
                            if (v < 0) v = 0;
                            out_ptr[out_idx + c] = v;
                        }
                    }
                }
            }
        }

        // Step 3: GroupNorm
        output = group_norm_->forward(output);

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
        grad_oracle_projection_.fill(0);

        Tensor<T> grad_input(shape);
        grad_input.fill(0);

        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();

        const T* eyes_ptr = eyes_out_cached_.data();

        const T* scale_ptr = spectral_scales_.data();
        const T* mix_w = mixing_weights_.data();
        const T* w_L = mix_w;
        const T* w_C = mix_w + in_channels_;
        const T* w_R = mix_w + 2 * in_channels_;
        const T* input_ptr = input_cached_.data();

        T* g_scale = grad_spectral_scales_.data();
        T* g_mix = grad_mixing_weights_.data();
        T* gw_L = g_mix;
        T* gw_C = g_mix + in_channels_;
        T* gw_R = g_mix + 2 * in_channels_;

        // Lazy Norm: Norm is fused, so factor here is 1.0 (or we need to account for it if backward needs unscaling?
        // Backward of y = IFWHT(x) * (1/N) is x_grad = FWHT(y_grad) * (1/N).
        // Since we fused 1/N into weights, the forward was y = IFWHT(x * (1/N)).
        // Gradient of y w.r.t x is IFWHT(1/N).
        // So backward prop is correct as is, provided `spectral_scales_` contains the 1/N.
        // Wait, backward uses `scale_ptr`. If `scale_ptr` is `1/N`, then `buf_eyes` in backward
        // will be scaled by `1/N` again.
        // Forward: x -> WHT -> *S -> IFWHT -> y.
        // Backward: dy -> FWHT -> *S -> IFWHT -> dx.
        // If S has 1/N, then backward applies 1/N too.
        // Standard WHT/IFWHT are symmetric (up to N).
        // The previous code applied 1/N explicitly at end of forward.
        // And 1/N explicitly at start of backward (before FWHT on grads).
        // Let's check previous backward:
        // `for(size_t c=0; c<out_channels_; ++c) buf_grad[c] *= norm;`
        // `algo::WHT::fwht_1d(buf_grad.data(), out_channels_);`
        // So yes, backward also needs scaling.
        // By fusing into `spectral_scales_`, we use `scale_ptr` in the middle.
        // Forward: x -> WHT -> *S(1/N) -> IFWHT.
        // Backward: dy -> FWHT -> *S(1/N) -> IFWHT.
        // This is mathematically consistent! The 1/N is applied via S in both passes.
        // So removing explicit norm in backward is also correct.

        T dropout_scale = (training_ && spectral_dropout_rate_ > 0.0f) ? (1.0f / (1.0f - spectral_dropout_rate_)) : 1.0f;

        Tensor<T> d_mixer_out({N, H_out, W_out, out_channels_});

        {
             Tensor<T> gn_in({N, H_out, W_out, out_channels_});

             #pragma omp parallel
             {
                std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_), buf_out(out_channels_);
                #pragma omp for collapse(3)
                for(size_t n=0; n<N; ++n) {
                    for(size_t h=0; h<H_out; ++h) {
                        for(size_t w=0; w<W_out; ++w) {
                             size_t idx = ((n*H_out + h)*W_out + w)*in_channels_;
                             for(size_t c=0; c<in_channels_; ++c) buf_in[c] = eyes_ptr[idx+c];

                             algo::WHT::fwht_1d(buf_in.data(), in_channels_);
                             if (training_ && spectral_dropout_rate_ > 0.0f) {
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
                                // Lazy Norm: fused
                             }

                             size_t out_idx = ((n*H_out + h)*W_out + w)*out_channels_;
                             for(size_t c=0; c<out_channels_; ++c) gn_in.data()[out_idx+c] = buf_out[c];
                        }
                    }
                }
             }

             // 2. Forward GN
             Tensor<T> gn_out = group_norm_->forward(gn_in);

             // 3. Backward ReLU
             Tensor<T> d_gn_out = grad_output; // Copy
             T* d_gn_ptr = d_gn_out.data();
             const T* gn_out_ptr = gn_out.data();

             for(size_t i=0; i<d_gn_out.size(); ++i) {
                 if (gn_out_ptr[i] <= 0) d_gn_ptr[i] = 0; // ReLU
             }

             // 4. Backward GN
             d_mixer_out = group_norm_->backward(d_gn_out);
        }

        const T* d_mix_ptr = d_mixer_out.data();

        #pragma omp parallel
        {
            std::vector<T> local_g_scale(in_channels_, 0);
            std::vector<T> local_gw_L(in_channels_, 0);
            std::vector<T> local_gw_C(in_channels_, 0);
            std::vector<T> local_gw_R(in_channels_, 0);

            std::vector<T, core::AlignedAllocator<T>> buf_grad(std::max(in_channels_, out_channels_));
            std::vector<T, core::AlignedAllocator<T>> buf_eyes(in_channels_);
            std::vector<T, core::AlignedAllocator<T>> d_eyes(in_channels_);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H_out; ++h) {
                    for(size_t w=0; w<W_out; ++w) {
                        size_t out_idx = ((n*H_out + h)*W_out + w)*out_channels_;

                        for(size_t c=0; c<out_channels_; ++c) buf_grad[c] = d_mix_ptr[out_idx + c];

                        if(use_ifwht_) {
                             // Lazy Norm: No explicit 1/N multiply. WHT is unitary up to N.
                             algo::WHT::fwht_1d(buf_grad.data(), out_channels_);
                        }

                        size_t idx = ((n*H_out + h)*W_out + w)*in_channels_;
                        for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] = eyes_ptr[idx+c];

                        algo::WHT::fwht_1d(buf_eyes.data(), in_channels_);
                        if (training_ && spectral_dropout_rate_ > 0.0f) {
                            for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] *= (dropout_mask_[c] * dropout_scale);
                        }
                        for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] *= scale_ptr[c];

                        std::fill(d_eyes.begin(), d_eyes.end(), 0);
                        if (in_channels_ == out_channels_) {
                            for(size_t c=0; c<in_channels_; ++c) {
                                T dy = buf_grad[c];

                                // dL/dW mixing
                                T prev = (c==0)?0:buf_eyes[c-1];
                                T curr = buf_eyes[c];
                                T next = (c==in_channels_-1)?0:buf_eyes[c+1];

                                local_gw_L[c] += dy * prev;
                                local_gw_C[c] += dy * curr;
                                local_gw_R[c] += dy * next;

                                // dL/dInput (propagate back)
                                d_eyes[c] += dy * w_C[c];
                                if (c > 0) d_eyes[c-1] += dy * w_L[c];
                                if (c < in_channels_-1) d_eyes[c+1] += dy * w_R[c];
                            }
                        }

                        for(size_t c=0; c<in_channels_; ++c) {
                            // dL/dScale
                            // d_eyes is the gradient w.r.t. the scaled spectral input (backpropagated through Mixer).
                            // buf_eyes is the scaled spectral input (Forward pass value).
                            // We need unscaled input: WHT(eyes).
                            T unscaled = (std::abs(scale_ptr[c]) > 1e-9) ? buf_eyes[c] / scale_ptr[c] : 0;
                            local_g_scale[c] += d_eyes[c] * unscaled;

                            // Apply scale for next step (Backprop to input)
                            d_eyes[c] *= scale_ptr[c];
                        }

                        if (training_ && spectral_dropout_rate_ > 0.0f) {
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
        if (use_gating_) params.push_back(&oracle_projection_);
        auto p = group_norm_->parameters();
        params.insert(params.end(), p.begin(), p.end());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads = {&grad_packed_weights_, &grad_spectral_scales_, &grad_mixing_weights_};
        if (use_gating_) grads.push_back(&grad_oracle_projection_);
        auto g = group_norm_->gradients();
        grads.insert(grads.end(), g.begin(), g.end());
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
    float spectral_dropout_rate_ = 0.1f;
    bool training_ = true;

    Tensor<T> packed_weights_;
    Tensor<T> spectral_scales_;
    Tensor<T> mixing_weights_;
    Tensor<T> oracle_projection_;

    Tensor<T> grad_packed_weights_;
    Tensor<T> grad_spectral_scales_;
    Tensor<T> grad_mixing_weights_;
    Tensor<T> grad_oracle_projection_;

    Tensor<T> input_cached_;
    Tensor<T> eyes_out_cached_;

    std::mt19937 rng_;
    std::vector<float> dropout_mask_;

    std::unique_ptr<GroupNorm<T>> group_norm_;
};

} // namespace layers
} // namespace dreidel
