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
#include <cassert>
#include <cstring> // for std::memset, std::memcpy

#ifdef __AVX2__
#include <immintrin.h>
#include "../hal/x86.hpp"
#endif

namespace dreidel {
namespace layers {

template <typename T>
class ZenithBlock : public Layer<T> {
public:
    // Global flag to toggle fused kernels for testing/debugging
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
          grad_oracle_projection_({1, in_channels})
    {
        // Enforce Power-of-Two for in_channels (required for FWHT)
        if ((in_channels_ & (in_channels_ - 1)) != 0) {
            throw std::invalid_argument("ZenithBlock in_channels must be a power of 2 for Spectral Mixing.");
        }

        T stddev = std::sqrt(static_cast<T>(2.0) / (kernel_size * kernel_size));
        packed_weights_.random(0, stddev);
        spectral_scales_.fill(1.0);

        // Initialize Mixing Weights: Center=1, Neighbors=0 (Identity Mixing)
        mixing_weights_.fill(0);
        T* mw = mixing_weights_.data();
        // Layout: Row 0 (Left), Row 1 (Center), Row 2 (Right)
        // Center row is at offset in_channels_
        std::fill(mw + in_channels_, mw + 2 * in_channels_, 1.0f);

        bias_.fill(0);
        oracle_projection_.random(-1.0, 1.0);

        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_mixing_weights_.fill(0);
        grad_bias_.fill(0);
        grad_oracle_projection_.fill(0);

        // optimized_weights_cache_.resize(in_channels * kernel_size * kernel_size); // Lazy init
    }

    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = false, bool use_gating = false)
        : ZenithBlock(channels, channels, kernel_size, spectral_dim, use_ifwht, use_dilated, use_gating, 1, 1) {}


    Tensor<T> forward(const Tensor<T>& input) override {
        // std::cout << "ZenithBlock::forward " << in_channels_ << "->" << out_channels_ << std::endl;
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

        // Optimization: Skip Eyes Cache allocation for Fused Path
        bool use_fused_path = false;
#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            if (use_fused_kernels && in_channels_ == 4096 && out_channels_ == 4096 && kernel_size_ == 3 && stride_ == 1 && upscale_ == 1) {
                use_fused_path = true;
            }
        }
#endif

        if (!use_fused_path) {
            if (eyes_out_cached_.shape().size() != 4 ||
                eyes_out_cached_.shape()[0] != N ||
                eyes_out_cached_.shape()[1] != H_out ||
                eyes_out_cached_.shape()[2] != W_out ||
                eyes_out_cached_.shape()[3] != in_channels_) {
                 // std::cout << "Allocating eyes_out_cached_: " << N << "x" << H_out << "x" << W_out << "x" << in_channels_ << std::endl;
                 eyes_out_cached_ = Tensor<T>({N, H_out, W_out, in_channels_});
            }
        }

        int k_rad = kernel_size_ / 2;
        const T* w_ptr = packed_weights_.data();
        T* eyes_ptr = eyes_out_cached_.data();

        const T* scale_ptr = spectral_scales_.data();
        const T* bias_ptr = bias_.data();
        const T* mix_w = mixing_weights_.data();

        bool is_downsample = (out_channels_ == in_channels_ / 2);
        bool is_upsample = (out_channels_ == in_channels_ * 2);

        bool eyes_done = false;
        bool mixer_done = false;

#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
            // Strategy 1: Fused L1-Resident Path for C=4096
            if (use_fused_kernels && in_channels_ == 4096 && out_channels_ == 4096 && kernel_size_ == 3 && stride_ == 1 && upscale_ == 1) {
                repack_weights();
                forward_avx2_fused_c4096(N, H, W, out_ptr, in_ptr, scale_ptr, mix_w, bias_ptr, active_mask);
                eyes_done = true;
                mixer_done = true;
            }
            // Eyes Optimization: Only for C >= 8 and multiples of 8 for now
            else if (in_channels_ >= 8 && in_channels_ % 8 == 0 && kernel_size_ == 3 && stride_ == 1) {
                repack_weights();

                if (upscale_ == 1) {
                    #pragma omp parallel for collapse(2)
                    for(size_t n=0; n<N; ++n) {
                        for(size_t h_out=0; h_out<H_out; ++h_out) {
                            if (!active_mask[n]) {
                                std::fill(eyes_ptr + ((n*H_out + h_out)*W_out)*in_channels_,
                                          eyes_ptr + ((n*H_out + h_out + 1)*W_out)*in_channels_, 0.0f);
                                continue;
                            }
                            // Call Generic AVX2 Sliding Window
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
                                            if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
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
                                            if(ih >= 0 && ih < H && iw >= 0 && iw < W) {
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
                        if (!active_mask[n]) {
                             for(size_t c=0; c<C; ++c) eyes_ptr[((n*H_out + h_out)*W_out + w_out)*C + c] = 0;
                             continue;
                        }
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
        }

#ifdef __AVX2__
        if constexpr (std::is_same_v<T, float>) {
             if (in_channels_ == out_channels_) {
                 if (in_channels_ == 32) {
                     forward_avx2_c32_mixer(N, H_out, W_out, out_ptr, eyes_ptr, scale_ptr, mix_w, bias_ptr, active_mask);
                     mixer_done = true;
                 } else if (in_channels_ == 64) {
                     forward_avx2_c64_mixer(N, H_out, W_out, out_ptr, eyes_ptr, scale_ptr, mix_w, bias_ptr, active_mask);
                     mixer_done = true;
                 } else if (in_channels_ == 128) {
                     forward_avx2_c128_mixer(N, H_out, W_out, out_ptr, eyes_ptr, scale_ptr, mix_w, bias_ptr, active_mask);
                     mixer_done = true;
                 } else if (in_channels_ > 128 && in_channels_ % 8 == 0) {
                     forward_avx2_generic_large_mixer(N, H_out, W_out, out_ptr, eyes_ptr, scale_ptr, mix_w, bias_ptr, active_mask, in_channels_);
                     mixer_done = true;
                 }
             }
             else if (in_channels_ == 64 && out_channels_ == 1) {
                  forward_avx2_c64_to_1_mixer(N, H_out, W_out, out_ptr, eyes_ptr, scale_ptr, mix_w, bias_ptr, active_mask);
                  mixer_done = true;
             }
        }
#endif

        if (!mixer_done) {
            // Fallback generic with Locally Connected Mixing
            #pragma omp parallel
            {
                std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_), buf_out(out_channels_);
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
                                // Locally Connected Mixing with Zero Padding at Boundaries
                                const T* w_L = mix_w;
                                const T* w_C = mix_w + in_channels_;
                                const T* w_R = mix_w + 2 * in_channels_;

                                // Buffer copy to avoid modifying while reading neighbors
                                // Or use buf_out as temp if in-place?
                                // mixing is C->C. buf_out is destination.

                                for(size_t c=0; c<in_channels_; ++c) {
                                    T prev = (c == 0) ? 0 : buf_in[c - 1];
                                    T next = (c == in_channels_ - 1) ? 0 : buf_in[c + 1];

                                    T val = w_L[c] * prev + w_C[c] * buf_in[c] + w_R[c] * next;
                                    buf_out[c] = val;
                                }
                            } else {
                                // Fallback for non-matching channels
                                std::fill(buf_out.begin(), buf_out.end(), 0);
                                size_t min_c = std::min(in_channels_, out_channels_);
                                const T* w_L = mix_w;
                                const T* w_C = mix_w + in_channels_;
                                const T* w_R = mix_w + 2 * in_channels_;

                                for(size_t c=0; c<min_c; ++c) {
                                    T prev = (c == 0) ? 0 : buf_in[c - 1];
                                    T next = (c == in_channels_ - 1) ? 0 : buf_in[c + 1];
                                    T val = w_L[c] * prev + w_C[c] * buf_in[c] + w_R[c] * next;
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
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        auto shape = input_cached_.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];

        auto g_shape = grad_output.shape();
        size_t H_out = g_shape[1];
        size_t W_out = g_shape[2];

        // Ensure gradients are zeroed before accumulation
        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_mixing_weights_.fill(0);
        grad_bias_.fill(0);
        grad_oracle_projection_.fill(0);

        Tensor<T> grad_input(shape);
        grad_input.fill(0);

        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();

        const T* eyes_ptr = eyes_out_cached_.data(); // Input to Mixer
        const T* input_ptr = input_cached_.data();   // Input to Eyes

        // Mixer Weights
        const T* scale_ptr = spectral_scales_.data();
        const T* mix_w = mixing_weights_.data(); // 3 rows: L, C, R
        const T* w_L = mix_w;
        const T* w_C = mix_w + in_channels_;
        const T* w_R = mix_w + 2 * in_channels_;

        T* g_scale = grad_spectral_scales_.data();
        T* g_mix = grad_mixing_weights_.data();
        T* gw_L = g_mix;
        T* gw_C = g_mix + in_channels_;
        T* gw_R = g_mix + 2 * in_channels_;
        T* g_bias = grad_bias_.data();

        // Temporary buffers for backward through Mixer
        // We process sample by sample to save memory or block by block

        // Backward Mixer (Generic)
        // Flow: Output -> ReLU -> Bias -> IFWHT -> Locally Connected Mix -> Scale -> FWHT -> Eyes

        // Precompute norm
        T norm = (use_ifwht_) ? (1.0f / static_cast<T>(out_channels_)) : 1.0f;

        // We need to accumulate gradients for weights.
        // Use thread-local accumulators for weights to avoid atomics

        #pragma omp parallel
        {
            std::vector<T> local_g_scale(in_channels_, 0);
            std::vector<T> local_gw_L(in_channels_, 0);
            std::vector<T> local_gw_C(in_channels_, 0);
            std::vector<T> local_gw_R(in_channels_, 0);
            std::vector<T> local_g_bias(out_channels_, 0);

            // Use default allocator to avoid potential aligned_alloc issues in threads
            std::vector<T> buf_grad(std::max(in_channels_, out_channels_));
            std::vector<T> buf_eyes(in_channels_);

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H_out; ++h) {
                    for(size_t w=0; w<W_out; ++w) {
                        // 1. Gradient of ReLU + Bias
                        size_t out_idx = ((n*H_out + h)*W_out + w)*out_channels_;

                        // We need the pre-activation value to compute ReLU gradient?
                        // Or use output? ReLU(x) = y. if y > 0, grad=1.
                        // But we don't have y cached easily (only eyes_out_cached).
                        // Recomputing forward pass for this pixel is cheap.

                        // Recompute Mixer Forward to get pre-relu
                        // Load eyes (Recompute if not cached)
                        if (eyes_ptr) {
                            size_t eyes_idx = ((n*H_out + h)*W_out + w)*in_channels_;
                            for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] = eyes_ptr[eyes_idx + c];
                        } else {
                            // Recompute Depthwise
                            int k_rad = kernel_size_ / 2;
                            const T* w_base = packed_weights_.data();
                            // std::cout << "Recomputing " << h << "," << w << std::endl;
                            std::fill(buf_eyes.begin(), buf_eyes.end(), 0);
                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                int ih = (int)h + ky;
                                if(ih < 0 || ih >= (int)H) continue;
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int iw = (int)w + kx;
                                    if(iw < 0 || iw >= (int)W) continue;

                                    const T* in_p = input_ptr + ((n*H + ih)*W + iw)*in_channels_;

                                    // Optimized Inner Loop for cache locality
                                    // w_base: [C, 1, K, K] (Planar)
                                    // Accessing c stride 1 is bad for cache if stride is large.
                                    // w_base[c*9 + ...] -> stride 9 floats. 36 bytes.
                                    // in_p[c] -> stride 1 float.

                                    // If C=4096.
                                    // c=0: w[0], in[0]
                                    // c=1: w[9], in[1]
                                    // This is fine?

                                    size_t w_offset = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                    size_t w_stride = kernel_size_ * kernel_size_;

                                    #pragma omp simd
                                    for(size_t c=0; c<in_channels_; ++c) {
                                        T w_val = w_base[c*w_stride + w_offset];
                                        buf_eyes[c] += in_p[c] * w_val;
                                    }
                                }
                            }
                        }

                        // FWHT
                        algo::WHT::fwht_1d(buf_eyes.data(), in_channels_);
                        // Scale
                        for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] *= scale_ptr[c];

                        // Mix
                        std::vector<T> mixed(out_channels_);
                        if (in_channels_ == out_channels_) {
                            for(size_t c=0; c<in_channels_; ++c) {
                                T prev = (c==0)?0:buf_eyes[c-1];
                                T next = (c==in_channels_-1)?0:buf_eyes[c+1];
                                mixed[c] = w_L[c]*prev + w_C[c]*buf_eyes[c] + w_R[c]*next;
                            }
                        } else {
                            // Fallback mix
                             size_t min_c = std::min(in_channels_, out_channels_);
                             std::fill(mixed.begin(), mixed.end(), 0);
                             for(size_t c=0; c<min_c; ++c) {
                                T prev = (c==0)?0:buf_eyes[c-1];
                                T next = (c==in_channels_-1)?0:buf_eyes[c+1];
                                mixed[c] = w_L[c]*prev + w_C[c]*buf_eyes[c] + w_R[c]*next;
                             }
                        }

                        // IFWHT
                        if(use_ifwht_) {
                            algo::WHT::fwht_1d(mixed.data(), out_channels_);
                            for(size_t c=0; c<out_channels_; ++c) mixed[c] *= norm;
                        }

                        // Now backprop ReLU/Bias
                        for(size_t c=0; c<out_channels_; ++c) {
                            T dy = go_ptr[out_idx + c];
                            T val = mixed[c] + bias_.data()[c]; // bias_ptr is not captured, use bias_.data()
                            if (val <= 0) dy = 0; // ReLU derivative

                            local_g_bias[c] += dy;
                            buf_grad[c] = dy;
                        }

                        // 2. Backward IFWHT
                        if(use_ifwht_) {
                             for(size_t c=0; c<out_channels_; ++c) buf_grad[c] *= norm; // Scale gradient
                             algo::WHT::fwht_1d(buf_grad.data(), out_channels_);
                        }

                        // 3. Backward Mixing (Locally Connected)
                        // y[c] = wL*p + wC*c + wR*n
                        // dL/dwL[c] = dL/dy[c] * p
                        // dL/dp += dL/dy[c] * wL[c]

                        // buf_grad holds dL/dy. buf_eyes holds inputs to mixing (scaled wht).
                        // We need to compute dL/d_buf_eyes (grad wrt scaled wht).

                        std::vector<T> d_eyes(in_channels_, 0);

                        if (in_channels_ == out_channels_) {
                            for(size_t c=0; c<in_channels_; ++c) {
                                T dy = buf_grad[c];
                                T prev = (c==0)?0:buf_eyes[c-1];
                                T curr = buf_eyes[c];
                                T next = (c==in_channels_-1)?0:buf_eyes[c+1];

                                local_gw_L[c] += dy * prev;
                                local_gw_C[c] += dy * curr;
                                local_gw_R[c] += dy * next;

                                // Propagate to inputs
                                // curr contributes to y[c] via wC
                                d_eyes[c] += dy * w_C[c];
                                // prev (index c-1) contributes to y[c] via wL[c]
                                if (c > 0) d_eyes[c-1] += dy * w_L[c];
                                // next (index c+1) contributes to y[c] via wR[c]
                                if (c < in_channels_-1) d_eyes[c+1] += dy * w_R[c];
                            }
                        } else {
                            // Fallback backward
                            size_t min_c = std::min(in_channels_, out_channels_);
                            for(size_t c=0; c<min_c; ++c) {
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
                        }

                        // 4. Backward Scale
                        // dL/dScale[c] = d_eyes[c] * fwht_out[c]
                        // We need fwht_out. buf_eyes currently holds scaled values.
                        // We need to recover unscaled FWHT output.

                        if (eyes_ptr) {
                            size_t eyes_idx = ((n*H_out + h)*W_out + w)*in_channels_;
                            for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] = eyes_ptr[eyes_idx + c];
                        } else {
                            // Recompute Depthwise again
                            int k_rad = kernel_size_ / 2;
                            const T* w_base = packed_weights_.data();
                            std::fill(buf_eyes.begin(), buf_eyes.end(), 0);
                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                int ih = (int)h + ky;
                                if(ih < 0 || ih >= (int)H) continue;
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int iw = (int)w + kx;
                                    if(iw < 0 || iw >= (int)W) continue;
                                    const T* in_p = input_ptr + ((n*H + ih)*W + iw)*in_channels_;
                                    for(size_t c=0; c<in_channels_; ++c) {
                                        T w_val = w_base[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                        buf_eyes[c] += in_p[c] * w_val;
                                    }
                                }
                            }
                        }
                        algo::WHT::fwht_1d(buf_eyes.data(), in_channels_); // Now buf_eyes is unscaled FWHT output

                        for(size_t c=0; c<in_channels_; ++c) {
                            local_g_scale[c] += d_eyes[c] * buf_eyes[c];
                            // Propagate to FWHT input: dL/dFWHT = d_eyes * scale
                            d_eyes[c] *= scale_ptr[c];
                        }

                        // 5. Backward FWHT
                        algo::WHT::fwht_1d(d_eyes.data(), in_channels_); // Symmetric

                        // d_eyes is now gradient wrt Eyes Output.
                        // We assume Eyes is Depthwise Conv.
                        // We need to backprop through Eyes to Input and Packed Weights.
                        // This is expensive to do per pixel if we iterate all weights.
                        // But we must.

                        // Eyes Backward:
                        // dL/dw[c, ky, kx] += d_eyes[c] * input[h+ky, w+kx, c]
                        // dL/dx += d_eyes * w (Spatial convolution transpose)

                        // We need atomic accumulation for weights if sharing threads?
                        // Or use thread local weight grads? Size C*K*K.
                        // C=64, K=3 -> 576 floats. Small enough.
                        // But input grads? (N, H, W, C). Shared memory. Need atomics.

                        // For simplicity in this implementation, we can use atomics for input grads.
                        // Or just don't compute input grads if not needed? (Layer usually needs to return them).
                        // We must compute input grads.

                        int k_rad = kernel_size_ / 2;
                        T* g_pack = grad_packed_weights_.data(); // Shared! Risk.
                        const T* w_pack = packed_weights_.data();

                        // If we want thread safety without massive locks, we can skip input grad or use atomics.
                        // For weight grad, we can use local buffer.
                        // But for `test_zenith_regression`, we assume batch size 1, maybe serial?
                        // The loop is parallel collapse(3).
                        // Let's use atomic add for input grad.

                        // Optimizing Eyes Backward is hard.
                        // Let's do a simple loop.

                        for(size_t c=0; c<in_channels_; ++c) {
                            T dy = d_eyes[c];
                            if (dy == 0) continue;

                            // For upscaling, we need to map out (h,w) to input coords.
                            // General Backward for Stride/Upscale
                            int ih_center = (upscale_ > 1) ? (int)h / (int)upscale_ : (int)h * (int)stride_;
                            int iw_center = (upscale_ > 1) ? (int)w / (int)upscale_ : (int)w * (int)stride_;

                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                int ih = ih_center + ky;
                                if(ih < 0 || ih >= (int)H) continue;
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int iw = iw_center + kx;
                                    if(iw < 0 || iw >= (int)W) continue;

                                    size_t in_idx = ((n*H + ih)*W + iw)*in_channels_ + c;
                                    T val = input_ptr[in_idx];

                                    // Weight Grad
                                    #pragma omp atomic
                                    g_pack[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)] += dy * val;

                                    // Input Grad
                                    T w_val = w_pack[c*kernel_size_*kernel_size_ + (ky+k_rad)*kernel_size_ + (kx+k_rad)];
                                    #pragma omp atomic
                                    gi_ptr[in_idx] += dy * w_val;
                                }
                            }
                        }
                    }
                }
            }

            // Reduce local accumulators
            #pragma omp critical
            {
                for(size_t i=0; i<local_g_bias.size(); ++i) g_bias[i] += local_g_bias[i];
                for(size_t i=0; i<local_g_scale.size(); ++i) g_scale[i] += local_g_scale[i];
                for(size_t i=0; i<local_gw_L.size(); ++i) gw_L[i] += local_gw_L[i];
                for(size_t i=0; i<local_gw_C.size(); ++i) gw_C[i] += local_gw_C[i];
                for(size_t i=0; i<local_gw_R.size(); ++i) gw_R[i] += local_gw_R[i];
            }
        }

        // Scale mixing gradients for stability (slower learning)
        // Factor 0.1 ensures the spectral topology remains stable while fine-tuning
        T mix_lr_scale = static_cast<T>(0.1);
        size_t mw_size = grad_mixing_weights_.size();
        T* g_mix_ptr = grad_mixing_weights_.data();
        #pragma omp parallel for
        for(size_t i=0; i<mw_size; ++i) {
            g_mix_ptr[i] *= mix_lr_scale;
        }

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&packed_weights_, &spectral_scales_, &mixing_weights_, &bias_, &oracle_projection_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_packed_weights_, &grad_spectral_scales_, &grad_mixing_weights_, &grad_bias_, &grad_oracle_projection_};
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

    void forward_avx2_c32_mixer(size_t N, size_t H, size_t W, float* out_ptr, const float* eyes_ptr, const float* scale_ptr, const float* mix_w, const float* bias_ptr, const std::vector<bool>& active_mask) {
        using namespace dreidel::hal::x86;
        // Pointers to mixing weights: Row 0 (L), Row 1 (C), Row 2 (R)
        const float* w_L = mix_w;
        const float* w_C = mix_w + 32;
        const float* w_R = mix_w + 64;

        __m256 norm = _mm256_set1_ps(1.0f/32.0f);
        __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    size_t out_idx = ((n*H + h)*W + w)*32;
                    if (!active_mask[n]) { for(int i=0; i<4; ++i) _mm256_storeu_ps(out_ptr + out_idx + i*8, zero); continue; }

                    __m256 r[4];
                    const float* ptr = eyes_ptr + out_idx;
                    for(int i=0; i<4; ++i) r[i] = _mm256_loadu_ps(ptr + i*8);

                    fwht8_avx2(r[0]); fwht8_avx2(r[1]); fwht8_avx2(r[2]); fwht8_avx2(r[3]);
                    Ops::butterfly(r[0], r[1]); Ops::butterfly(r[2], r[3]);
                    Ops::butterfly(r[0], r[2]); Ops::butterfly(r[1], r[3]);

                    // Scale
                    for(int i=0; i<4; ++i) {
                        __m256 s = _mm256_loadu_ps(scale_ptr + i*8);
                        r[i] = _mm256_mul_ps(r[i], s);
                    }

                    // Locally Connected Mixing
                    __m256 t[4]; for(int i=0; i<4; ++i) t[i] = r[i];

                    // Boundaries: r[0] prev is Zero. r[3] next is Zero.
                    auto mix_ch = [&](int i, __m256 curr, __m256 prev, __m256 next) {
                        __m256 wl = _mm256_load_ps(w_L + i*8);
                        __m256 wc = _mm256_load_ps(w_C + i*8);
                        __m256 wr = _mm256_load_ps(w_R + i*8);

                        __m256 p = shift_right_1(curr, prev);
                        __m256 n = shift_left_1(curr, next);
                        __m256 res = _mm256_mul_ps(curr, wc);
                        res = _mm256_fmadd_ps(p, wl, res);
                        res = _mm256_fmadd_ps(n, wr, res);
                        return res;
                    };

                    r[0] = mix_ch(0, t[0], zero, t[1]);
                    r[1] = mix_ch(1, t[1], t[0], t[2]);
                    r[2] = mix_ch(2, t[2], t[1], t[3]);
                    r[3] = mix_ch(3, t[3], t[2], zero);

                    fwht8_avx2(r[0]); fwht8_avx2(r[1]); fwht8_avx2(r[2]); fwht8_avx2(r[3]);
                    Ops::butterfly(r[0], r[1]); Ops::butterfly(r[2], r[3]);
                    Ops::butterfly(r[0], r[2]); Ops::butterfly(r[1], r[3]);

                    for(int i=0; i<4; ++i) r[i] = _mm256_mul_ps(r[i], norm);
                    for(int i=0; i<4; ++i) {
                        __m256 b = _mm256_loadu_ps(bias_ptr + i*8);
                        r[i] = _mm256_add_ps(r[i], b);
                        r[i] = _mm256_max_ps(r[i], zero);
                        _mm256_storeu_ps(out_ptr + out_idx + i*8, r[i]);
                    }
                }
            }
        }
    }

    void forward_avx2_c64_mixer(
        size_t N, size_t H_out, size_t W_out,
        float* out_ptr, const float* eyes_ptr,
        const float* scale_ptr, const float* mix_w, const float* bias_ptr,
        const std::vector<bool>& active_mask
    ) {
        using namespace dreidel::hal::x86;
        const float* w_L = mix_w;
        const float* w_C = mix_w + 64;
        const float* w_R = mix_w + 128;

        __m256 norm = _mm256_set1_ps(1.0f/64.0f);
        __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H_out; ++h) {
                for(size_t w=0; w<W_out; ++w) {
                    size_t out_idx = ((n*H_out + h)*W_out + w)*64;
                    if (!active_mask[n]) {
                        for(int i=0; i<8; ++i) _mm256_storeu_ps(out_ptr + out_idx + i*8, zero);
                        continue;
                    }
                    size_t eyes_idx = ((n*H_out + h)*W_out + w)*64;
                    const float* ptr = eyes_ptr + eyes_idx;

                    __m256 r[8];
                    for(int i=0; i<8; ++i) r[i] = _mm256_loadu_ps(ptr + i*8);

                    fwht8_avx2(r[0]); fwht8_avx2(r[1]); fwht8_avx2(r[2]); fwht8_avx2(r[3]);
                    fwht8_avx2(r[4]); fwht8_avx2(r[5]); fwht8_avx2(r[6]); fwht8_avx2(r[7]);
                    Ops::butterfly(r[0], r[1]); Ops::butterfly(r[2], r[3]); Ops::butterfly(r[4], r[5]); Ops::butterfly(r[6], r[7]);
                    Ops::butterfly(r[0], r[2]); Ops::butterfly(r[1], r[3]); Ops::butterfly(r[4], r[6]); Ops::butterfly(r[5], r[7]);
                    Ops::butterfly(r[0], r[4]); Ops::butterfly(r[1], r[5]); Ops::butterfly(r[2], r[6]); Ops::butterfly(r[3], r[7]);

                    for(int i=0; i<8; ++i) {
                         __m256 s = _mm256_loadu_ps(scale_ptr + i*8);
                         r[i] = _mm256_mul_ps(r[i], s);
                    }

                    __m256 t[8]; for(int i=0; i<8; ++i) t[i] = r[i];

                    auto mix_ch = [&](int i, __m256 curr, __m256 prev, __m256 next) {
                        __m256 wl = _mm256_load_ps(w_L + i*8);
                        __m256 wc = _mm256_load_ps(w_C + i*8);
                        __m256 wr = _mm256_load_ps(w_R + i*8);
                        __m256 p = shift_right_1(curr, prev);
                        __m256 n = shift_left_1(curr, next);
                        __m256 res = _mm256_mul_ps(curr, wc);
                        res = _mm256_fmadd_ps(p, wl, res);
                        res = _mm256_fmadd_ps(n, wr, res);
                        return res;
                    };

                    r[0] = mix_ch(0, t[0], zero, t[1]);
                    for(int i=1; i<7; ++i) r[i] = mix_ch(i, t[i], t[i-1], t[i+1]);
                    r[7] = mix_ch(7, t[7], t[6], zero);

                    fwht8_avx2(r[0]); fwht8_avx2(r[1]); fwht8_avx2(r[2]); fwht8_avx2(r[3]);
                    fwht8_avx2(r[4]); fwht8_avx2(r[5]); fwht8_avx2(r[6]); fwht8_avx2(r[7]);
                    Ops::butterfly(r[0], r[1]); Ops::butterfly(r[2], r[3]); Ops::butterfly(r[4], r[5]); Ops::butterfly(r[6], r[7]);
                    Ops::butterfly(r[0], r[2]); Ops::butterfly(r[1], r[3]); Ops::butterfly(r[4], r[6]); Ops::butterfly(r[5], r[7]);
                    Ops::butterfly(r[0], r[4]); Ops::butterfly(r[1], r[5]); Ops::butterfly(r[2], r[6]); Ops::butterfly(r[3], r[7]);

                    for(int i=0; i<8; ++i) {
                        __m256 b = _mm256_loadu_ps(bias_ptr + i*8);
                        r[i] = _mm256_mul_ps(r[i], norm);
                        r[i] = _mm256_add_ps(r[i], b);
                        r[i] = _mm256_max_ps(r[i], zero);
                        _mm256_storeu_ps(out_ptr + out_idx + i*8, r[i]);
                    }
                }
            }
        }
    }

    void forward_avx2_c128_mixer(size_t N, size_t H, size_t W, float* out_ptr, const float* eyes_ptr, const float* scale_ptr, const float* mix_w, const float* bias_ptr, const std::vector<bool>& active_mask) {
        using namespace dreidel::hal::x86;
        const float* w_L = mix_w;
        const float* w_C = mix_w + 128;
        const float* w_R = mix_w + 256;

        __m256 norm = _mm256_set1_ps(1.0f/128.0f);
        __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    size_t out_idx = ((n*H + h)*W + w)*128;
                    if (!active_mask[n]) { for(int i=0; i<16; ++i) _mm256_storeu_ps(out_ptr + out_idx + i*8, zero); continue; }

                    __m256 r[16];
                    const float* ptr = eyes_ptr + out_idx;
                    for(int i=0; i<16; ++i) r[i] = _mm256_loadu_ps(ptr + i*8);

                    alignas(32) float buf[128];
                    for(int i=0; i<16; ++i) _mm256_store_ps(buf + i*8, r[i]);
                    fwht128_avx2(buf);
                    for(int i=0; i<16; ++i) r[i] = _mm256_load_ps(buf + i*8);

                    // Scale
                    for(int i=0; i<16; ++i) {
                        __m256 s = _mm256_loadu_ps(scale_ptr + i*8);
                        r[i] = _mm256_mul_ps(r[i], s);
                    }

                    // Locally Connected Mixing
                    // We must save copy to avoid overwriting prev
                    alignas(32) float t_buf[128];
                    for(int i=0; i<16; ++i) _mm256_store_ps(t_buf + i*8, r[i]);

                    auto mix_ch = [&](int i, __m256 curr, __m256 prev, __m256 next) {
                        __m256 wl = _mm256_load_ps(w_L + i*8);
                        __m256 wc = _mm256_load_ps(w_C + i*8);
                        __m256 wr = _mm256_load_ps(w_R + i*8);
                        __m256 p = shift_right_1(curr, prev);
                        __m256 n = shift_left_1(curr, next);
                        __m256 res = _mm256_mul_ps(curr, wc);
                        res = _mm256_fmadd_ps(p, wl, res);
                        res = _mm256_fmadd_ps(n, wr, res);
                        return res;
                    };

                    for(int i=0; i<16; ++i) {
                         __m256 prev = (i==0) ? zero : _mm256_load_ps(t_buf + (i-1)*8);
                         __m256 next = (i==15) ? zero : _mm256_load_ps(t_buf + (i+1)*8);
                         __m256 curr = _mm256_load_ps(t_buf + i*8);
                         r[i] = mix_ch(i, curr, prev, next);
                    }

                    // IFWHT
                    for(int i=0; i<16; ++i) _mm256_store_ps(buf + i*8, r[i]);
                    fwht128_avx2(buf);
                    for(int i=0; i<16; ++i) r[i] = _mm256_load_ps(buf + i*8);

                    // Bias/ReLU
                    for(int i=0; i<16; ++i) {
                        __m256 b = _mm256_loadu_ps(bias_ptr + i*8);
                        r[i] = _mm256_mul_ps(r[i], norm);
                        r[i] = _mm256_add_ps(r[i], b);
                        r[i] = _mm256_max_ps(r[i], zero);
                        _mm256_storeu_ps(out_ptr + out_idx + i*8, r[i]);
                    }
                }
            }
        }
    }

    void forward_avx2_generic_large_mixer(size_t N, size_t H, size_t W, float* out_ptr, const float* eyes_ptr, const float* scale_ptr, const float* mix_w, const float* bias_ptr, const std::vector<bool>& active_mask, size_t C) {
        using namespace dreidel::hal::x86;
        const float* w_L = mix_w;
        const float* w_C = mix_w + C;
        const float* w_R = mix_w + 2*C;

        __m256 norm = _mm256_set1_ps(1.0f/C);
        __m256 zero = _mm256_setzero_ps();

        #pragma omp parallel
        {
            std::vector<float, core::AlignedAllocator<float>> buf(C);
            std::vector<float, core::AlignedAllocator<float>> t_buf(C);
            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H; ++h) {
                    for(size_t w=0; w<W; ++w) {
                        size_t idx = ((n*H + h)*W + w)*C;
                        if (!active_mask[n]) {
                            for(size_t i=0; i<C; i+=8) _mm256_storeu_ps(out_ptr + idx + i, zero);
                            continue;
                        }

                        // Load & FWHT
                        std::copy(eyes_ptr + idx, eyes_ptr + idx + C, buf.begin());
                        algo::WHT::fwht_1d(buf.data(), C);

                        // Scale
                        for(size_t i=0; i<C; i+=8) {
                            __m256 v = _mm256_load_ps(buf.data() + i);
                            __m256 s = _mm256_loadu_ps(scale_ptr + i);
                            _mm256_store_ps(buf.data() + i, _mm256_mul_ps(v, s));
                        }

                        // Copy to temp for mixing source
                        std::copy(buf.begin(), buf.end(), t_buf.begin());

                        // Vectorized Mixing Loop
                        for(size_t i=0; i<C; i+=8) {
                             __m256 curr = _mm256_load_ps(t_buf.data() + i);
                             __m256 prev, next;

                             if (i > 0) prev = _mm256_loadu_ps(t_buf.data() + i - 1); // Unaligned load of [i-1...i+6]
                             else {
                                 // Boundary 0. Prev is 0?
                                 // shift_right_1 needs prev reg.
                                 // We need specific unaligned logic or use scalar fallback for boundary?
                                 // Actually, simpler:
                                 // Construct 'prev' from Zero and curr?
                                 // shift_right_1(curr, zero) gives [0, c0, c1...c6]. Correct.
                                 prev = zero;
                             }

                             if (i + 8 < C) next = _mm256_loadu_ps(t_buf.data() + i + 1); // Unaligned [i+1...i+8]
                             else {
                                 // Boundary end. Next is 0?
                                 // shift_left_1(curr, zero) gives [c1...c7, 0]. Correct.
                                 next = zero;
                             }

                             // BUT wait. `shift_right_1` uses a register as prev.
                             // `_mm256_loadu_ps` loads a block of memory.
                             // If I load `t_buf + i - 1` (valid pointer), it contains `[c-1, c0... c6]`.
                             // This IS the shifted data directly! I don't need `shift_right_1`.
                             // `shift_right_1` is for when data is in registers and we can't unaligned load across them easily.
                             // Here we are in memory!
                             // So:
                             // val = w_C * curr + w_L * loadu(i-1) + w_R * loadu(i+1)
                             // This is much faster.

                             __m256 wl = _mm256_load_ps(w_L + i);
                             __m256 wc = _mm256_load_ps(w_C + i);
                             __m256 wr = _mm256_load_ps(w_R + i);

                             __m256 p_vec, n_vec;

                             // Handle Boundary i=0 for loadu(i-1) - unsafe pointer
                             if (i == 0) {
                                 // loadu at -1 is invalid.
                                 // Construct p_vec manually: [0, t[0], t[1]...t[6]]
                                 // shift_right_1(curr, zero) does exactly this.
                                 p_vec = shift_right_1(curr, zero);
                             } else {
                                 p_vec = _mm256_loadu_ps(t_buf.data() + i - 1);
                             }

                             // Handle Boundary i=C-8 for loadu(i+1) - unsafe pointer if C aligned?
                             // t_buf is size C. t_buf[C] is end. t_buf + i + 1 + 7 = t_buf + C - 8 + 1 + 7 = t_buf + C.
                             // So loadu at C-8+1 reads up to t_buf[C]. This is 1 float past end?
                             // No. Range is [i+1, i+8]. Last element is at i+8.
                             // If i = C-8. Indices are C-7 ... C.
                             // t_buf has indices 0...C-1.
                             // So accessing index C is out of bounds.
                             // So for last block, we cannot use loadu(i+1).
                             if (i + 8 >= C) {
                                 n_vec = shift_left_1(curr, zero);
                             } else {
                                 n_vec = _mm256_loadu_ps(t_buf.data() + i + 1);
                             }

                             __m256 res = _mm256_mul_ps(curr, wc);
                             res = _mm256_fmadd_ps(p_vec, wl, res);
                             res = _mm256_fmadd_ps(n_vec, wr, res);

                             _mm256_store_ps(buf.data() + i, res);
                        }

                        // IFWHT
                        algo::WHT::fwht_1d(buf.data(), C);

                        // Bias/ReLU/Store
                        for(size_t i=0; i<C; i+=8) {
                            __m256 v = _mm256_load_ps(buf.data() + i);
                            __m256 b = _mm256_loadu_ps(bias_ptr + i);
                            v = _mm256_mul_ps(v, norm);
                            v = _mm256_add_ps(v, b);
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
        float* out_ptr, const float* eyes_ptr,
        const float* scale_ptr, const float* mix_w, const float* bias_ptr,
        const std::vector<bool>& active_mask
    ) {
        using namespace dreidel::hal::x86;
        // Pointers for channel 0 weights (since output is 1 channel, we effectively select C=0 after mixing?
        // Wait, C64->1 implies we reduce.
        // Previous implementation: `val = sp_w[0] * buf63 + sp_w[1] * buf0 + sp_w[2] * buf1`
        // This calculated mixed value for channel 0.
        // With locally connected:
        // `val = w_L[0]*0 + w_C[0]*buf0 + w_R[0]*buf1`
        // So we only need w_L[0], w_C[0], w_R[0].

        float w_l = mix_w[0];
        float w_c = mix_w[64];
        float w_r = mix_w[128];

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
                    size_t out_idx = ((n*H_out + h)*W_out + w); // out_channels=1
                    if (!active_mask[n]) {
                        out_ptr[out_idx] = 0.0f;
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

                    // FWHT64
                    fwht8_avx2(r0); fwht8_avx2(r1); fwht8_avx2(r2); fwht8_avx2(r3);
                    fwht8_avx2(r4); fwht8_avx2(r5); fwht8_avx2(r6); fwht8_avx2(r7);
                    Ops::butterfly(r0, r1); Ops::butterfly(r2, r3); Ops::butterfly(r4, r5); Ops::butterfly(r6, r7);
                    Ops::butterfly(r0, r2); Ops::butterfly(r1, r3); Ops::butterfly(r4, r6); Ops::butterfly(r5, r7);
                    Ops::butterfly(r0, r4); Ops::butterfly(r1, r5); Ops::butterfly(r2, r6); Ops::butterfly(r3, r7);

                    // Scale
                    r0 = _mm256_mul_ps(r0, scale_r0);
                    r1 = _mm256_mul_ps(r1, scale_r1);
                    r2 = _mm256_mul_ps(r2, scale_r2);
                    r3 = _mm256_mul_ps(r3, scale_r3);
                    r4 = _mm256_mul_ps(r4, scale_r4);
                    r5 = _mm256_mul_ps(r5, scale_r5);
                    r6 = _mm256_mul_ps(r6, scale_r6);
                    r7 = _mm256_mul_ps(r7, scale_r7);

                    // Permute logic: c=0. prev=0 (Linear), next=buf1.
                    // buf[0] is r0[0]. buf[1] is r0[1].

                    // Extract needed values
                    float buf0 = _mm256_cvtss_f32(r0); // low element of r0
                    // Extract r0[1]
                    __m256 r0_shuf = _mm256_permute_ps(r0, 0x01); // rotate
                    float buf1 = _mm256_cvtss_f32(r0_shuf);

                    // Mix
                    float val = w_l * 0.0f + w_c * buf0 + w_r * buf1;

                    // IFWHT(1) is Identity

                    // Bias
                    val += bias_val;

                    // ReLU
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
    bool use_gating_;
    size_t stride_;
    size_t upscale_;

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

    // Cache for optimized weight layout
    std::vector<float, core::AlignedAllocator<float>> optimized_weights_cache_;

    // --- Fused 4096 Kernel ---
    void forward_avx2_fused_c4096(
        size_t N, size_t H, size_t W,
        float* out_ptr, const float* in_ptr,
        const float* scale_ptr, const float* mix_w, const float* bias_ptr,
        const std::vector<bool>& active_mask
    ) {
        using namespace dreidel::hal::x86;
        constexpr int C = 4096;

        // Pointers for mixing weights
        const float* w_L = mix_w;
        const float* w_C = mix_w + C;
        const float* w_R = mix_w + 2*C;

        __m256 norm = _mm256_set1_ps(1.0f/4096.0f);
        __m256 zero = _mm256_setzero_ps();

        const float* w_base = optimized_weights_cache_.data(); // Repacked Depthwise Weights

        #pragma omp parallel
        {
            // Scratchpad in L1 (Stack)
            // 4096 floats = 16KB. Fits in 32KB L1 D-Cache.
            alignas(64) float l1_buffer[C];

            // Temporary buffer for Mixing source (need copy of FWHT result)
            // Also L1? If we use 2 buffers, 32KB. Might spill to L2 if L1 is 32KB.
            // But L1 is often 32KB or 48KB.
            // If we spill, it's L2, which is still fast (12 cycles).
            // We need a temp buffer for Locally Connected Mixing (dependency on neighbors).
            alignas(64) float mix_temp[C];

            // Padding Buffer (L2 resident, per thread? No, per image)
            // We process one image at a time per thread?
            // "Pad the input map in L2 with zeros before starting."
            // Since H, W are variable, we allocate on heap (L2/L3).
            // (W+2)*(H+2)*C.
            // For 64x64: 66*66*4096*4 = ~70MB. Too big for L2.
            // Wait. The feature map is H*W*C.
            // 64*64*4096*4 = 64MB.
            // "Your entire feature map fits in L2 Cache." -> User said "The entire feature map fits in L2".
            // 30MB L3. L2 is 1-2MB.
            // 64x64x4096 is HUGE. 16M floats = 64MB.
            // Maybe H, W are small?
            // If H=7, W=7 (Bottleneck): 49*4096*4 = 800KB. Fits in L2.
            // If H=64, W=64: 64MB. Does not fit in L2.
            // The user said: "You have a feature map of size 7x7x4096."
            // Ah, MAP_W=7 in the snippet.
            // So this optimization is crucial for bottlenecks.

            // We will dynamically allocate padded buffer.
            std::vector<float, core::AlignedAllocator<float>> padded_input;

            #pragma omp for
            for(size_t n=0; n<N; ++n) {
                if (!active_mask[n]) {
                    // Zero output
                    size_t total = H * W * C;
                    std::memset(out_ptr + n*total, 0, total * sizeof(float));
                    continue;
                }

                // 1. Prepare Padded Input (L2)
                // We copy the current image to a padded buffer.
                // size: (H+2)*(W+2)*C
                size_t H_pad = H + 2;
                size_t W_pad = W + 2;
                padded_input.resize(H_pad * W_pad * C);

                // Zero init padding
                // Efficient zeroing?
                // Just zero the borders? Or memset all.
                // Memset is safe.
                std::memset(padded_input.data(), 0, padded_input.size() * sizeof(float));

                // Copy center
                const float* img_in = in_ptr + (n*H*W)*C;
                float* pad_ptr = padded_input.data();
                for(size_t h=0; h<H; ++h) {
                    const float* src_row = img_in + (h*W)*C;
                    float* dst_row = pad_ptr + ((h+1)*W_pad + 1)*C;
                    std::memcpy(dst_row, src_row, W*C*sizeof(float));
                }

                // 2. Iterate "Fat Pixels"
                for(size_t h=0; h<H; ++h) {
                    for(size_t w=0; w<W; ++w) {
                        // Coordinates in padded input
                        size_t ph = h + 1;
                        size_t pw = w + 1;

                        // --- STEP 1: DEPTHWISE CONV (Gather -> L1) ---
                        // Accumulate into l1_buffer
                        // Clear buffer
                        _mm256_store_ps(l1_buffer, zero); // First chunk clear logic inside loop
                        // Actually we can just set to 0.
                        // Better: Use first neighbor to overwrite, others accumulate.
                        // But weight might be 0?
                        // Safe: set to 0.
                        for(int i=0; i<C; i+=8) _mm256_store_ps(l1_buffer+i, zero);

                        // 3x3 Loop unrolled?
                        // ky, kx from -1 to 1.
                        // Input indices: (ph+ky, pw+kx)
                        for(int ky=-1; ky<=1; ++ky) {
                            for(int kx=-1; kx<=1; ++kx) {
                                // Padded buffer index
                                size_t ny = ph + ky;
                                size_t nx = pw + kx;
                                const float* neighbor_pixel = pad_ptr + (ny * W_pad + nx) * C;

                                // Weight index: repacked as [ky][kx][C]
                                // ky+1 -> 0..2, kx+1 -> 0..2
                                // Flat index: ((ky+1)*3 + (kx+1)) * C
                                const float* w_ptr_k = w_base + ((ky+1)*3 + (kx+1)) * C;

                                for(int i=0; i<C; i+=8) {
                                    __m256 v_acc = _mm256_load_ps(l1_buffer + i);
                                    __m256 v_in  = _mm256_load_ps(neighbor_pixel + i); // Aligned? Yes, padded buffer is aligned.
                                    __m256 v_w   = _mm256_load_ps(w_ptr_k + i);
                                    v_acc = _mm256_fmadd_ps(v_in, v_w, v_acc);
                                    _mm256_store_ps(l1_buffer + i, v_acc);
                                }
                            }
                        }

                        // --- STEP 2: SPECTRAL TRANSFORM (In-Place L1) ---
                        algo::WHT::fwht_1d(l1_buffer, C);

                        // --- STEP 3: SPECTRAL MIXING (In-Place L1) ---
                        // Scale
                         for(int i=0; i<C; i+=8) {
                             __m256 v = _mm256_load_ps(l1_buffer + i);
                             __m256 s = _mm256_loadu_ps(scale_ptr + i);
                             _mm256_store_ps(l1_buffer + i, _mm256_mul_ps(v, s));
                         }

                         // Locally Connected Mixing
                         // Need copy to mix_temp to read neighbors safely
                         // Or copy l1_buffer to mix_temp, read mix_temp, write l1_buffer.
                         std::memcpy(mix_temp, l1_buffer, C*sizeof(float));

                         for(int i=0; i<C; i+=8) {
                             __m256 curr = _mm256_load_ps(mix_temp + i);
                             __m256 prev, next;

                             // Helper for boundaries
                             // i=0: prev is 0. i=C-8: next is 0.
                             if (i == 0) {
                                 prev = shift_right_1(curr, zero);
                             } else {
                                 prev = _mm256_loadu_ps(mix_temp + i - 1);
                             }

                             if (i + 8 >= C) {
                                 next = shift_left_1(curr, zero);
                             } else {
                                 next = _mm256_loadu_ps(mix_temp + i + 1);
                             }

                             __m256 wl = _mm256_load_ps(w_L + i);
                             __m256 wc = _mm256_load_ps(w_C + i);
                             __m256 wr = _mm256_load_ps(w_R + i);

                             __m256 res = _mm256_mul_ps(curr, wc);
                             res = _mm256_fmadd_ps(prev, wl, res);
                             res = _mm256_fmadd_ps(next, wr, res);

                             _mm256_store_ps(l1_buffer + i, res);
                         }

                        // --- STEP 4: INVERSE TRANSFORM (In-Place L1) ---
                        // IFWHT
                        algo::WHT::fwht_1d(l1_buffer, C);

                        // --- STEP 5: STORE (L1 -> L2 Output) ---
                        // Apply Bias, Norm, ReLU
                        float* dst_pixel = out_ptr + ((n*H + h)*W + w)*C;
                        for(int i=0; i<C; i+=8) {
                            __m256 v = _mm256_load_ps(l1_buffer + i);
                            __m256 b = _mm256_loadu_ps(bias_ptr + i);
                            v = _mm256_mul_ps(v, norm); // IFWHT Norm
                            v = _mm256_add_ps(v, b);
                            v = _mm256_max_ps(v, zero); // ReLU
                            _mm256_storeu_ps(dst_pixel + i, v);
                        }
                    }
                }
            }
        }
    }

}; // Close class ZenithBlock

} // namespace layers
} // namespace dreidel
