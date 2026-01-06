#pragma once

#include "Layer.hpp"
#include "GroupNorm.hpp"
#include "../core/Memory.hpp"
#include "../core/Allocator.hpp"
#include "../hal/ops.hpp"
#include "../hal/x86.hpp"
#include "../algo/WHT.hpp"
#include "../algo/Sequency.hpp"
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
    static constexpr size_t rig_iterations_ = 3;

    ZenithBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = false, bool use_gating = false, size_t stride = 1, size_t upscale = 1,
                const std::string& init_scheme = "he", bool use_slm = false, bool use_sequency = false)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_gating_(use_gating), stride_(stride), upscale_(upscale), use_slm_(use_slm), use_sequency_(use_sequency),
          packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          spectral_scales_({1, in_channels}),
          mixing_weights_({3, in_channels}),
          oracle_projection_({1, in_channels}),
          slm_weights1_({5, in_channels}),
          slm_bias1_({1, in_channels}),
          slm_weights2_({5, in_channels}),
          slm_bias2_({1, in_channels}),

          grad_packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          grad_spectral_scales_({1, in_channels}),
          grad_mixing_weights_({3, in_channels}),
          grad_oracle_projection_({1, in_channels}),
          grad_slm_weights1_({5, in_channels}),
          grad_slm_bias1_({1, in_channels}),
          grad_slm_weights2_({5, in_channels}),
          grad_slm_bias2_({1, in_channels}),
          rng_(std::random_device{}())
    {
        if ((in_channels_ & (in_channels_ - 1)) != 0) {
            throw std::invalid_argument("ZenithBlock in_channels must be a power of 2 for Spectral Mixing.");
        }

        initialize(init_scheme);

        // Gating
        oracle_projection_.random(-1.0, 1.0);

        // SLM Init (DSG: Denoising Signal Gate)
        if (use_slm_) {
            // Layer 1: Energy Integration & Thresholding
            // Kaiming Normal for ReLU
            T stddev = std::sqrt(2.0f / (5.0f)); // Fan-in is 5 (kernel size) * 1 (depthwise)
            slm_weights1_.random(0, stddev);
            slm_bias1_.fill(-1.0f); // Learnable Noise Floor (-sigma)

            // Layer 2: Gate Smoothing
            // Xavier for Sigmoid (or Kaiming if we consider it linear until sigmoid)
            slm_weights2_.random(0, stddev);
            slm_bias2_.fill(0.0f);
        }

        // Zero Grads
        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_mixing_weights_.fill(0);
        grad_oracle_projection_.fill(0);
        grad_slm_weights1_.fill(0);
        grad_slm_bias1_.fill(0);
        grad_slm_weights2_.fill(0);
        grad_slm_bias2_.fill(0);

        // 2. GroupNorm (Hardcoded Standard)
        size_t groups = 32;
        if (out_channels_ % groups != 0) groups = 1;
        group_norm_ = std::make_unique<GroupNorm<T>>(groups, out_channels_);

        // Sequency Map
        if (use_sequency_) {
            auto s_map = algo::Sequency::compute_to_natural_map(in_channels_);
            sequency_map_.assign(s_map.begin(), s_map.end());

            auto n_map = algo::Sequency::compute_to_sequency_map(in_channels_);
            natural_map_.assign(n_map.begin(), n_map.end());
        }
    }

    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = false, bool use_gating = false, bool use_slm = false, bool use_sequency = false)
        : ZenithBlock(channels, channels, kernel_size, spectral_dim, use_ifwht, use_dilated, use_gating, 1, 1, "he", use_slm, use_sequency) {}

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

    void set_sequency_ordering(bool use) {
        use_sequency_ = use;
        if (use_sequency_ && sequency_map_.empty()) {
            auto s_map = algo::Sequency::compute_to_natural_map(in_channels_);
            sequency_map_.assign(s_map.begin(), s_map.end());

            auto n_map = algo::Sequency::compute_to_sequency_map(in_channels_);
            natural_map_.assign(n_map.begin(), n_map.end());
        }
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

        const T* slm_w1_ptr = use_slm_ ? slm_weights1_.data() : nullptr;
        const T* slm_b1_ptr = use_slm_ ? slm_bias1_.data() : nullptr;
        const T* slm_w2_ptr = use_slm_ ? slm_weights2_.data() : nullptr;
        const T* slm_b2_ptr = use_slm_ ? slm_bias2_.data() : nullptr;

        // Inverted Dropout Scale
        T dropout_scale = (apply_dropout) ? (1.0f / (1.0f - spectral_dropout_rate_)) : 1.0f;

        // Fused Step 1 (Eyes/Depthwise) & Step 2 (Mixer) & ReLU
        #pragma omp parallel
        {
            std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_), buf_out(out_channels_);
            std::vector<T, core::AlignedAllocator<T>> buf_mag; // Alloc only if needed
            std::vector<T, core::AlignedAllocator<T>> buf_hidden; // For DSG

            if (use_slm_ || use_sequency_) buf_mag.resize(in_channels_);
            if (use_slm_) buf_hidden.resize(in_channels_);

            // Buffers for RIG
            std::vector<T, core::AlignedAllocator<T>> buf_est;
            std::vector<T, core::AlignedAllocator<T>> buf_gate;

            if (use_slm_) {
                buf_est.resize((rig_iterations_ + 1) * in_channels_);
                buf_hidden.resize(rig_iterations_ * in_channels_);
                buf_gate.resize(rig_iterations_ * in_channels_);
            }

            // Scratch for permutation
            std::vector<T, core::AlignedAllocator<T>> buf_temp;
            if (use_sequency_) buf_temp.resize(in_channels_);

            const int32_t* seq_map_ptr = use_sequency_ ? sequency_map_.data() : nullptr;
            const int32_t* nat_map_ptr = use_sequency_ ? natural_map_.data() : nullptr;

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

                        // Permute Natural -> Sequency
                        if (use_sequency_) {
                            if constexpr (std::is_same_v<T, float>) {
                                #ifdef DREIDEL_ARCH_AVX2
                                    permute_avx2(buf_temp.data(), buf_in.data(), seq_map_ptr, in_channels_);
                                #else
                                    for(size_t k=0; k<in_channels_; ++k) buf_temp[k] = buf_in[seq_map_ptr[k]];
                                #endif
                            } else {
                                for(size_t k=0; k<in_channels_; ++k) buf_temp[k] = buf_in[seq_map_ptr[k]];
                            }
                            std::memcpy(buf_in.data(), buf_temp.data(), in_channels_ * sizeof(T));
                        }

                        // Zenith-RIG (Recurrent Iterative Gating)
                        if (use_slm_) {
                            // 1. Initialize Est[0] = |Input|
                            T* est_0 = buf_est.data();
                            for(size_t c=0; c<in_channels_; ++c) est_0[c] = std::abs(buf_in[c]);

                            // 2. Unrolled Loop
                            for(size_t k=0; k<rig_iterations_; ++k) {
                                const T* curr_est = buf_est.data() + k * in_channels_;
                                T* next_est = buf_est.data() + (k+1) * in_channels_;
                                T* curr_hidden = buf_hidden.data() + k * in_channels_;
                                T* curr_gate = buf_gate.data() + k * in_channels_;

                                // Layer 1: Conv(Est[k]) -> Hidden
                                for(size_t c=0; c<in_channels_; ++c) {
                                    T val = slm_b1_ptr[c];
                                    const T* w = slm_w1_ptr + c * 5;
                                    if (c >= 2) val += w[0] * curr_est[c-2];
                                    if (c >= 1) val += w[1] * curr_est[c-1];
                                    val += w[2] * curr_est[c];
                                    if (c + 1 < in_channels_) val += w[3] * curr_est[c+1];
                                    if (c + 2 < in_channels_) val += w[4] * curr_est[c+2];
                                    curr_hidden[c] = (val > 0) ? val : 0; // ReLU
                                }

                                // Layer 2: Conv(Hidden) -> Gate
                                for(size_t c=0; c<in_channels_; ++c) {
                                    T val = slm_b2_ptr[c];
                                    const T* w = slm_w2_ptr + c * 5;
                                    if (c >= 2) val += w[0] * curr_hidden[c-2];
                                    if (c >= 1) val += w[1] * curr_hidden[c-1];
                                    val += w[2] * curr_hidden[c];
                                    if (c + 1 < in_channels_) val += w[3] * curr_hidden[c+1];
                                    if (c + 2 < in_channels_) val += w[4] * curr_hidden[c+2];
                                    curr_gate[c] = 1.0f / (1.0f + std::exp(-val)); // Sigmoid
                                }

                                // Update State: Est[k+1] = Est[0] * Gate[k] (Refinement)
                                for(size_t c=0; c<in_channels_; ++c) {
                                    next_est[c] = est_0[c] * curr_gate[c];
                                }
                            }

                            // 3. Apply Final Gate to Complex Input
                            const T* final_gate = buf_gate.data() + (rig_iterations_ - 1) * in_channels_;
                            for(size_t c=0; c<in_channels_; ++c) {
                                buf_in[c] *= final_gate[c];
                            }
                        }

                        // Permute Sequency -> Natural
                        if (use_sequency_) {
                            if constexpr (std::is_same_v<T, float>) {
                                #ifdef DREIDEL_ARCH_AVX2
                                    permute_avx2(buf_temp.data(), buf_in.data(), nat_map_ptr, in_channels_);
                                #else
                                    for(size_t i=0; i<in_channels_; ++i) buf_temp[i] = buf_in[nat_map_ptr[i]];
                                #endif
                            } else {
                                for(size_t i=0; i<in_channels_; ++i) buf_temp[i] = buf_in[nat_map_ptr[i]];
                            }
                            std::memcpy(buf_in.data(), buf_temp.data(), in_channels_ * sizeof(T));
                        }

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

        T* g_slm_w1 = use_slm_ ? grad_slm_weights1_.data() : nullptr;
        T* g_slm_b1 = use_slm_ ? grad_slm_bias1_.data() : nullptr;
        T* g_slm_w2 = use_slm_ ? grad_slm_weights2_.data() : nullptr;
        T* g_slm_b2 = use_slm_ ? grad_slm_bias2_.data() : nullptr;

        const T* slm_w1_ptr = use_slm_ ? slm_weights1_.data() : nullptr;
        const T* slm_b1_ptr = use_slm_ ? slm_bias1_.data() : nullptr;
        const T* slm_w2_ptr = use_slm_ ? slm_weights2_.data() : nullptr;
        const T* slm_b2_ptr = use_slm_ ? slm_bias2_.data() : nullptr;

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

            // SLM Local Grads
            std::vector<T> local_g_slm_w1, local_g_slm_b1;
            std::vector<T> local_g_slm_w2, local_g_slm_b2;
            if (use_slm_) {
                local_g_slm_w1.resize(in_channels_ * 5, 0.0);
                local_g_slm_b1.resize(in_channels_, 0.0);
                local_g_slm_w2.resize(in_channels_ * 5, 0.0);
                local_g_slm_b2.resize(in_channels_, 0.0);
            }

            std::vector<T, core::AlignedAllocator<T>> buf_grad(std::max(in_channels_, out_channels_));
            std::vector<T, core::AlignedAllocator<T>> buf_eyes(in_channels_);
            std::vector<T, core::AlignedAllocator<T>> d_eyes(in_channels_);

            // SLM temp buffers (RIG sized)
            std::vector<T, core::AlignedAllocator<T>> buf_est;
            std::vector<T, core::AlignedAllocator<T>> buf_hidden;
            std::vector<T, core::AlignedAllocator<T>> buf_gate;

            // Gradients buffers
            std::vector<T, core::AlignedAllocator<T>> d_est;
            std::vector<T, core::AlignedAllocator<T>> d_hidden;
            std::vector<T, core::AlignedAllocator<T>> d_gate;

            std::vector<T, core::AlignedAllocator<T>> buf_temp; // Scratch for permute

            if (use_slm_ || use_sequency_) buf_temp.resize(in_channels_);

            if (use_slm_) {
                buf_est.resize((rig_iterations_ + 1) * in_channels_);
                buf_hidden.resize(rig_iterations_ * in_channels_);
                buf_gate.resize(rig_iterations_ * in_channels_);

                d_est.resize((rig_iterations_ + 1) * in_channels_);
                d_hidden.resize(in_channels_); // Can reuse layer-wise
                d_gate.resize(in_channels_);
            }

            const int32_t* seq_map_ptr = use_sequency_ ? sequency_map_.data() : nullptr;
            const int32_t* nat_map_ptr = use_sequency_ ? natural_map_.data() : nullptr;

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

                        // Permute Natural -> Sequency (Forward Logic)
                        if (use_sequency_) {
                            if constexpr (std::is_same_v<T, float>) {
                                #ifdef DREIDEL_ARCH_AVX2
                                    permute_avx2(buf_temp.data(), buf_eyes.data(), seq_map_ptr, in_channels_);
                                #else
                                    for(size_t k=0; k<in_channels_; ++k) buf_temp[k] = buf_eyes[seq_map_ptr[k]];
                                #endif
                            } else {
                                for(size_t k=0; k<in_channels_; ++k) buf_temp[k] = buf_eyes[seq_map_ptr[k]];
                            }
                            std::memcpy(buf_eyes.data(), buf_temp.data(), in_channels_ * sizeof(T));
                        }

                        // Zenith-RIG Forward Recompute (for Backward)
                        if (use_slm_) {
                            // 1. Initialize Est[0]
                            T* est_0 = buf_est.data();
                            for(size_t c=0; c<in_channels_; ++c) est_0[c] = std::abs(buf_eyes[c]);

                            // 2. Unrolled Loop (Replay)
                            for(size_t k=0; k<rig_iterations_; ++k) {
                                const T* curr_est = buf_est.data() + k * in_channels_;
                                T* next_est = buf_est.data() + (k+1) * in_channels_;
                                T* curr_hidden = buf_hidden.data() + k * in_channels_;
                                T* curr_gate = buf_gate.data() + k * in_channels_;

                                for(size_t c=0; c<in_channels_; ++c) {
                                    T val = slm_b1_ptr[c];
                                    const T* w = slm_w1_ptr + c * 5;
                                    if (c >= 2) val += w[0] * curr_est[c-2];
                                    if (c >= 1) val += w[1] * curr_est[c-1];
                                    val += w[2] * curr_est[c];
                                    if (c + 1 < in_channels_) val += w[3] * curr_est[c+1];
                                    if (c + 2 < in_channels_) val += w[4] * curr_est[c+2];
                                    curr_hidden[c] = (val > 0) ? val : 0;
                                }

                                for(size_t c=0; c<in_channels_; ++c) {
                                    T val = slm_b2_ptr[c];
                                    const T* w = slm_w2_ptr + c * 5;
                                    if (c >= 2) val += w[0] * curr_hidden[c-2];
                                    if (c >= 1) val += w[1] * curr_hidden[c-1];
                                    val += w[2] * curr_hidden[c];
                                    if (c + 1 < in_channels_) val += w[3] * curr_hidden[c+1];
                                    if (c + 2 < in_channels_) val += w[4] * curr_hidden[c+2];
                                    curr_gate[c] = 1.0f / (1.0f + std::exp(-val));
                                }

                                for(size_t c=0; c<in_channels_; ++c) {
                                    next_est[c] = est_0[c] * curr_gate[c];
                                }
                            }

                            // buf_eyes holds u_seq (pre-gated). We DO NOT apply final gate here.
                            // We need u_seq for gradient calculation (dL/dGate = dL/dY * u_seq).
                        }

                        // NOTE: buf_eyes is now in Sequency order (if use_sequency).

                        // dL/dScale & Backprop through Scale & Dropout
                        // buf_eyes holds `u`. buf_gate holds `Gate`.
                        const T* final_gate = buf_gate.data() + (rig_iterations_ - 1) * in_channels_;

                        for(size_t c=0; c<in_channels_; ++c) {
                            T u = buf_eyes[c]; // u_seq
                            T g = final_gate[c];
                            T val = u * g; // u * Gate

                            if (training_ && spectral_dropout_rate_ > 0.0f) {
                                val *= (dropout_mask_[c] * dropout_scale);
                            }

                            // dL/dScale = dL/dY * val
                            local_g_scale[c] += d_eyes[c] * val;

                            // dL/dVal = dL/dY * Scale
                            d_eyes[c] *= scale_ptr[c];

                            // Backprop Dropout
                            if (training_ && spectral_dropout_rate_ > 0.0f) {
                                d_eyes[c] *= (dropout_mask_[c] * dropout_scale);
                            }
                        }

                        // Permute d_eyes Natural -> Sequency (Before SLM Backward)
                        // Input d_eyes is dL/d(Natural). We need dL/d(Sequency) for SLM.
                        if (use_sequency_) {
                            if constexpr (std::is_same_v<T, float>) {
                                #ifdef DREIDEL_ARCH_AVX2
                                    permute_avx2(buf_temp.data(), d_eyes.data(), seq_map_ptr, in_channels_);
                                #else
                                    for(size_t k=0; k<in_channels_; ++k) buf_temp[k] = d_eyes[seq_map_ptr[k]];
                                #endif
                            } else {
                                for(size_t k=0; k<in_channels_; ++k) buf_temp[k] = d_eyes[seq_map_ptr[k]];
                            }
                            std::memcpy(d_eyes.data(), buf_temp.data(), in_channels_ * sizeof(T));
                        }

                        // Zenith-RIG Backward
                        if (use_slm_) {
                            // d_eyes is dL/d(u * Gate).
                            // buf_eyes is `u`.
                            // final_gate is `Gate`.

                            // Initialize d_est buffer to 0
                            std::fill(d_est.begin(), d_est.end(), 0.0f);

                            for(size_t c=0; c<in_channels_; ++c) {
                                T g = final_gate[c];
                                T u = buf_eyes[c];

                                // 1. Gradient to Final Gate
                                // dL/dGate = dL/d(u*g) * u
                                d_gate[c] = d_eyes[c] * u;

                                // 2. Update d_eyes to hold dL/du partial (dL/d(u*g) * g)
                                d_eyes[c] = d_eyes[c] * g;
                            }

                            // BPTT Loop (k = iterations-1 down to 0)
                            for(int k = (int)rig_iterations_ - 1; k >= 0; --k) {
                                const T* curr_gate = buf_gate.data() + k * in_channels_;
                                const T* curr_hidden = buf_hidden.data() + k * in_channels_;
                                const T* curr_est = buf_est.data() + k * in_channels_;
                                const T* est_0 = buf_est.data(); // Always Input Mag

                                // dL/dGate[k] is in d_gate.
                                // Note: For k < last, d_gate comes from next_est.
                                // Est[k+1] = Est[0] * Gate[k].
                                // dL/dGate[k] += dL/dEst[k+1] * Est[0].
                                if (k < (int)rig_iterations_ - 1) {
                                    T* d_next_est = d_est.data() + (k+1) * in_channels_;
                                    for(size_t c=0; c<in_channels_; ++c) {
                                        d_gate[c] += d_next_est[c] * est_0[c];
                                    }
                                }

                                // Backprop Gate -> Logits -> Layer 2 -> Hidden
                                std::fill(d_hidden.begin(), d_hidden.end(), 0.0f);
                                for(size_t c=0; c<in_channels_; ++c) {
                                    T g = curr_gate[c];
                                    T d_g = d_gate[c];
                                    T d_z2 = d_g * g * (1.0f - g); // Sigmoid deriv

                                    local_g_slm_b2[c] += d_z2;

                                    T* w_g = local_g_slm_w2.data() + c * 5;
                                    const T* w_fwd = slm_w2_ptr + c * 5;

                                    if (c >= 2) { w_g[0] += d_z2 * curr_hidden[c-2]; d_hidden[c-2] += d_z2 * w_fwd[0]; }
                                    if (c >= 1) { w_g[1] += d_z2 * curr_hidden[c-1]; d_hidden[c-1] += d_z2 * w_fwd[1]; }
                                    w_g[2] += d_z2 * curr_hidden[c]; d_hidden[c] += d_z2 * w_fwd[2];
                                    if (c + 1 < in_channels_) { w_g[3] += d_z2 * curr_hidden[c+1]; d_hidden[c+1] += d_z2 * w_fwd[3]; }
                                    if (c + 2 < in_channels_) { w_g[4] += d_z2 * curr_hidden[c+2]; d_hidden[c+2] += d_z2 * w_fwd[4]; }
                                }

                                // Backprop Hidden -> Layer 1 -> Input(Est[k])
                                T* d_curr_est = d_est.data() + k * in_channels_;
                                for(size_t c=0; c<in_channels_; ++c) {
                                    T d_h = d_hidden[c];
                                    if (curr_hidden[c] <= 0) d_h = 0; // ReLU

                                    local_g_slm_b1[c] += d_h;

                                    T* w_g = local_g_slm_w1.data() + c * 5;
                                    const T* w_fwd = slm_w1_ptr + c * 5;

                                    if (c >= 2) { w_g[0] += d_h * curr_est[c-2]; d_curr_est[c-2] += d_h * w_fwd[0]; }
                                    if (c >= 1) { w_g[1] += d_h * curr_est[c-1]; d_curr_est[c-1] += d_h * w_fwd[1]; }
                                    w_g[2] += d_h * curr_est[c]; d_curr_est[c] += d_h * w_fwd[2];
                                    if (c + 1 < in_channels_) { w_g[3] += d_h * curr_est[c+1]; d_curr_est[c+1] += d_h * w_fwd[3]; }
                                    if (c + 2 < in_channels_) { w_g[4] += d_h * curr_est[c+2]; d_curr_est[c+2] += d_h * w_fwd[4]; }
                                }

                                // Reset d_gate for next iteration
                                std::fill(d_gate.begin(), d_gate.end(), 0.0f);
                            }

                            // Propagate dL/dEst[k] to dL/dEst[0] (Input Mag)
                            T* d_est_0 = d_est.data();
                            for(size_t k=0; k<rig_iterations_; ++k) {
                                const T* gate_k = buf_gate.data() + k * in_channels_;
                                const T* d_est_k_plus_1 = d_est.data() + (k+1) * in_channels_;

                                for(size_t c=0; c<in_channels_; ++c) {
                                    d_est_0[c] += d_est_k_plus_1[c] * gate_k[c];
                                }
                            }

                            // Finally: d_eyes += dL/dMag * sgn(u)
                            // dL/dMag is in d_est[0].
                            for(size_t c=0; c<in_channels_; ++c) {
                                T u = buf_eyes[c];
                                T sgn = (u > 0) ? 1.0f : ((u < 0) ? -1.0f : 0.0f);
                                d_eyes[c] += d_est_0[c] * sgn;
                            }
                        }

                        // Permute d_eyes Sequency -> Natural (for IFWHT)
                        if (use_sequency_) {
                            if constexpr (std::is_same_v<T, float>) {
                                #ifdef DREIDEL_ARCH_AVX2
                                    permute_avx2(buf_temp.data(), d_eyes.data(), nat_map_ptr, in_channels_);
                                #else
                                    for(size_t i=0; i<in_channels_; ++i) buf_temp[i] = d_eyes[nat_map_ptr[i]];
                                #endif
                            } else {
                                for(size_t i=0; i<in_channels_; ++i) buf_temp[i] = d_eyes[nat_map_ptr[i]];
                            }
                            std::memcpy(d_eyes.data(), buf_temp.data(), in_channels_ * sizeof(T));
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
                if (use_slm_) {
                    for(size_t i=0; i<local_g_slm_w1.size(); ++i) g_slm_w1[i] += local_g_slm_w1[i];
                    for(size_t i=0; i<local_g_slm_b1.size(); ++i) g_slm_b1[i] += local_g_slm_b1[i];
                    for(size_t i=0; i<local_g_slm_w2.size(); ++i) g_slm_w2[i] += local_g_slm_w2[i];
                    for(size_t i=0; i<local_g_slm_b2.size(); ++i) g_slm_b2[i] += local_g_slm_b2[i];
                }
            }
        }

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params = {&packed_weights_, &spectral_scales_, &mixing_weights_};
        if (use_gating_) params.push_back(&oracle_projection_);
        if (use_slm_) {
            params.push_back(&slm_weights1_);
            params.push_back(&slm_bias1_);
            params.push_back(&slm_weights2_);
            params.push_back(&slm_bias2_);
        }
        auto p = group_norm_->parameters();
        params.insert(params.end(), p.begin(), p.end());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads = {&grad_packed_weights_, &grad_spectral_scales_, &grad_mixing_weights_};
        if (use_gating_) grads.push_back(&grad_oracle_projection_);
        if (use_slm_) {
            grads.push_back(&grad_slm_weights1_);
            grads.push_back(&grad_slm_bias1_);
            grads.push_back(&grad_slm_weights2_);
            grads.push_back(&grad_slm_bias2_);
        }
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
    bool use_slm_;
    bool use_sequency_ = false;
    float spectral_dropout_rate_ = 0.1f;
    bool training_ = true;

    std::vector<int32_t, core::AlignedAllocator<int32_t>> sequency_map_;
    std::vector<int32_t, core::AlignedAllocator<int32_t>> natural_map_;

    Tensor<T> packed_weights_;

    static inline void permute_avx2(float* out, const float* in, const int32_t* indices, size_t N) {
        #ifdef DREIDEL_ARCH_AVX2
        size_t i = 0;
        // Unroll by 4 (8 floats * 4 = 32 floats per loop)
        for (; i + 32 <= N; i += 32) {
             // Gather 4 blocks of 8
             __m256i idx0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i));
             __m256i idx1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i + 8));
             __m256i idx2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i + 16));
             __m256i idx3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i + 24));

             __m256 r0 = _mm256_i32gather_ps(in, idx0, 4);
             __m256 r1 = _mm256_i32gather_ps(in, idx1, 4);
             __m256 r2 = _mm256_i32gather_ps(in, idx2, 4);
             __m256 r3 = _mm256_i32gather_ps(in, idx3, 4);

             _mm256_store_ps(out + i, r0);
             _mm256_store_ps(out + i + 8, r1);
             _mm256_store_ps(out + i + 16, r2);
             _mm256_store_ps(out + i + 24, r3);
        }
        // Fallback for remaining (still AVX for multiples of 8)
        for (; i + 8 <= N; i += 8) {
            __m256i idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i));
            __m256 r = _mm256_i32gather_ps(in, idx, 4);
            _mm256_store_ps(out + i, r);
        }
        // Scalar tail (likely N is power of 2 so 0, but good to have)
        for (; i < N; ++i) {
            out[i] = in[indices[i]];
        }
        #endif
    }
    Tensor<T> spectral_scales_;
    Tensor<T> mixing_weights_;
    Tensor<T> oracle_projection_;
    Tensor<T> slm_weights1_;
    Tensor<T> slm_bias1_;
    Tensor<T> slm_weights2_;
    Tensor<T> slm_bias2_;

    Tensor<T> grad_packed_weights_;
    Tensor<T> grad_spectral_scales_;
    Tensor<T> grad_mixing_weights_;
    Tensor<T> grad_oracle_projection_;
    Tensor<T> grad_slm_weights1_;
    Tensor<T> grad_slm_bias1_;
    Tensor<T> grad_slm_weights2_;
    Tensor<T> grad_slm_bias2_;

    Tensor<T> input_cached_;
    Tensor<T> eyes_out_cached_;

    std::mt19937 rng_;
    std::vector<float> dropout_mask_;

    std::unique_ptr<GroupNorm<T>> group_norm_;
};

} // namespace layers
} // namespace dreidel
