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
    static constexpr size_t srig_iterations_ = 3;

    ZenithBlock(size_t in_channels, size_t out_channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = false, bool use_gating = false, size_t stride = 1, size_t upscale = 1,
                const std::string& init_scheme = "he", bool use_slm = false, bool use_sequency = false, float norm_eps = 1e-5)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_gating_(use_gating), stride_(stride), upscale_(upscale), use_slm_(use_slm), use_sequency_(use_sequency),
          norm_eps_(norm_eps),
          packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          spectral_scales_({1, in_channels}),
          mixing_weights_({3, in_channels}),
          oracle_projection_({1, in_channels}),
          srig_weights_({5, in_channels}),
          srig_bias_({1, in_channels}),

          grad_packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          grad_spectral_scales_({1, in_channels}),
          grad_mixing_weights_({3, in_channels}),
          grad_oracle_projection_({1, in_channels}),
          grad_srig_weights_({5, in_channels}),
          grad_srig_bias_({1, in_channels}),
          rng_(std::random_device{}())
    {
        if ((in_channels_ & (in_channels_ - 1)) != 0) {
            throw std::invalid_argument("ZenithBlock in_channels must be a power of 2 for Spectral Mixing.");
        }

        initialize(init_scheme);

        // Gating
        oracle_projection_.random(-1.0, 1.0);

        // SRIG Init (Spherical Recurrent Iterative Gating)
        if (use_slm_) {
            // Weights: He Init
            T stddev = std::sqrt(2.0f / 5.0f);
            srig_weights_.random(0, stddev);
            // Bias: -1.0 to close the gate on noise (Learned Noise Floor)
            srig_bias_.fill(-1.0f);
        }

        // Zero Grads
        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_mixing_weights_.fill(0);
        grad_oracle_projection_.fill(0);
        grad_srig_weights_.fill(0);
        grad_srig_bias_.fill(0);

        // 2. GroupNorm (Hardcoded Standard)
        size_t groups = 32;
        if (out_channels_ % groups != 0) groups = 1;
        group_norm_ = std::make_unique<GroupNorm<T>>(groups, out_channels_, norm_eps_);

        // Sequency Map
        if (use_sequency_) {
            auto s_map = algo::Sequency::compute_to_natural_map(in_channels_);
            sequency_map_.assign(s_map.begin(), s_map.end());

            auto n_map = algo::Sequency::compute_to_sequency_map(in_channels_);
            natural_map_.assign(n_map.begin(), n_map.end());
        }
    }

    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim,
                bool use_ifwht = true, bool use_dilated = false, bool use_gating = false, bool use_slm = false, bool use_sequency = false, float norm_eps = 1e-5)
        : ZenithBlock(channels, channels, kernel_size, spectral_dim, use_ifwht, use_dilated, use_gating, 1, 1, "he", use_slm, use_sequency, norm_eps) {}

    void initialize(const std::string& scheme) {
        packed_weights_.fill(0);
        if (scheme == "identity") {
            T* w_ptr = packed_weights_.data();
            size_t k_center = kernel_size_ / 2;
            size_t spatial_size = kernel_size_ * kernel_size_;
            for (size_t c = 0; c < in_channels_; ++c) {
                w_ptr[c * spatial_size + k_center * kernel_size_ + k_center] = 1.0f;
            }
        } else if (scheme == "he") {
            T stddev = std::sqrt(2.0f / (in_channels_ * kernel_size_ * kernel_size_));
            packed_weights_.random(0, stddev);
        } else {
            packed_weights_.random(-0.01, 0.01);
        }

        T norm_factor = (use_ifwht_) ? (1.0f / static_cast<T>(out_channels_)) : 1.0f;
        spectral_scales_.fill(1.0f * norm_factor);

        mixing_weights_.fill(0);
        T* mw = mixing_weights_.data();
        std::fill(mw + in_channels_, mw + 2 * in_channels_, 1.0f);

        repack_weights();
    }

    void repack_weights() {
        size_t C = in_channels_;
        size_t K = kernel_size_;
        repacked_weights_.resize(K * K * C);
        const T* w_src = packed_weights_.data(); // (C, 1, K, K)

        // Target: (K, K, C) -> [ky, kx, c]
        for(size_t c=0; c<C; ++c) {
            for(size_t ky=0; ky<K; ++ky) {
                for(size_t kx=0; kx<K; ++kx) {
                    // src: c*K*K + ky*K + kx
                    T val = w_src[c*K*K + ky*K + kx];
                    // dst: (ky*K + kx)*C + c
                    repacked_weights_[(ky*K + kx)*C + c] = val;
                }
            }
        }
    }

    void set_spectral_dropout(float rate) {
        spectral_dropout_rate_ = rate;
    }

    void set_training(bool training) override {
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

    void set_pruning_mask(const Tensor<T>* mask) {
        pruning_mask_ = mask;
    }

    void set_epsilon(float eps) {
        norm_eps_ = eps;
        if (group_norm_) {
            group_norm_->set_epsilon(static_cast<T>(eps));
        }
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Optim: Repack weights on forward if needed.
        if (training_) repack_weights();

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
        const T* scale_ptr = spectral_scales_.data();
        const T* mix_w = mixing_weights_.data();

        const T* srig_w_ptr = use_slm_ ? srig_weights_.data() : nullptr;
        const T* srig_b_ptr = use_slm_ ? srig_bias_.data() : nullptr;

        T dropout_scale = (apply_dropout) ? (1.0f / (1.0f - spectral_dropout_rate_)) : 1.0f;

        // Precompute MagW
        std::vector<T> mag_w_cache(in_channels_);
        if (use_slm_) {
            for(size_t c=0; c<in_channels_; ++c) {
                 const T* w = srig_w_ptr + c * 5;
                 T sum_sq = 0;
                 for(int k=0; k<5; ++k) sum_sq += w[k]*w[k];
                 mag_w_cache[c] = std::sqrt(sum_sq + 1e-8f);
            }
        }
        const T* mag_w_ptr = mag_w_cache.data();

        const T* mask_ptr = (pruning_mask_ && !pruning_mask_->shape().empty()) ? pruning_mask_->data() : nullptr;
        size_t mask_blocks_stride = (mask_ptr) ? (in_channels_ / 8) : 0;

        #pragma omp parallel
        {
            std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_), buf_out(out_channels_);
            std::vector<T, core::AlignedAllocator<T>> buf_est;
            std::vector<T, core::AlignedAllocator<T>> buf_gate;
            std::vector<T, core::AlignedAllocator<T>> buf_act;

            if (use_slm_) {
                buf_est.resize((srig_iterations_ + 1) * in_channels_);
                buf_gate.resize(srig_iterations_ * in_channels_);
                buf_act.resize(srig_iterations_ * in_channels_);
            }

            std::vector<T, core::AlignedAllocator<T>> buf_temp;
            if (use_sequency_) buf_temp.resize(in_channels_);

            const int32_t* seq_map_ptr = use_sequency_ ? sequency_map_.data() : nullptr;
            const int32_t* nat_map_ptr = use_sequency_ ? natural_map_.data() : nullptr;

            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h_out=0; h_out<H_out; ++h_out) {
                    for(size_t w_out=0; w_out<W_out; ++w_out) {

                        size_t eyes_idx = ((n*H_out + h_out)*W_out + w_out)*in_channels_;
                        T* eyes_store_ptr = eyes_ptr + eyes_idx;

                        // --- Vectorized Eyes Convolution ---
                        bool can_prune = (mask_ptr && stride_ == 1 && upscale_ == 1);
                        const T* pixel_mask = nullptr;
                        if (can_prune) {
                            size_t mask_idx = ((n*H_out + h_out)*W_out + w_out) * mask_blocks_stride;
                            pixel_mask = mask_ptr + mask_idx;
                        }

                        int h_in_base, w_in_base;
                        bool is_upscale = (upscale_ > 1);

                        if (is_upscale) {
                             h_in_base = (up_shift > 0) ? ((int)h_out >> up_shift) : ((int)h_out / (int)upscale_);
                             w_in_base = (up_shift > 0) ? ((int)w_out >> up_shift) : ((int)w_out / (int)upscale_);
                        } else {
                             h_in_base = h_out * stride_;
                             w_in_base = w_out * stride_;
                        }

                        using Ops = hal::ActiveOps;

                        for(int ky=-k_rad; ky<=k_rad; ++ky) {
                            for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                int ih = h_in_base;
                                int iw = w_in_base;

                                if (is_upscale) {
                                    int v_h = (int)h_out + ky;
                                    int v_w = (int)w_out + kx;
                                    if (v_h < 0 || v_h >= (int)H_out || v_w < 0 || v_w >= (int)W_out) continue;
                                    ih = (up_shift > 0) ? (v_h >> up_shift) : (v_h / (int)upscale_);
                                    iw = (up_shift > 0) ? (v_w >> up_shift) : (v_w / (int)upscale_);
                                } else {
                                    ih += ky;
                                    iw += kx;
                                    if(ih < 0 || ih >= (int)H || iw < 0 || iw >= (int)W) continue;
                                }

                                const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                const T* p_w = repacked_weights_.data() + k_idx * in_channels_;

                                size_t c = 0;
                                for (; c + Ops::SIMD_WIDTH <= in_channels_; c += Ops::SIMD_WIDTH) {
                                    if (can_prune && (c % 8 == 0)) {
                                        if (pixel_mask[c/8] == 0.0f) {
                                            c += 8;
                                            c -= Ops::SIMD_WIDTH;
                                            c += 8;
                                            continue;
                                        }
                                    }

                                    auto v_in = Ops::load(p_in + c);
                                    auto v_w = Ops::load(p_w + c);
                                    auto v_acc = Ops::load(buf_in.data() + c);
                                    v_acc = Ops::add(v_acc, Ops::mul(v_in, v_w));
                                    Ops::store(buf_in.data() + c, v_acc);
                                }
                                for (; c < in_channels_; ++c) {
                                    buf_in[c] += p_in[c] * p_w[c];
                                }
                            }
                        }

                        size_t c_s = 0;
                        for(; c_s + Ops::SIMD_WIDTH <= in_channels_; c_s += Ops::SIMD_WIDTH) {
                            Ops::store(eyes_store_ptr + c_s, Ops::load(buf_in.data() + c_s));
                        }
                        for(; c_s < in_channels_; ++c_s) eyes_store_ptr[c_s] = buf_in[c_s];

                        algo::WHT::fwht_1d(buf_in.data(), in_channels_);

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

                        if (use_slm_) {
                            T* est_0 = buf_est.data();
                            for(size_t c=0; c<in_channels_; ++c) est_0[c] = std::abs(buf_in[c]);

                            for(size_t k=0; k<srig_iterations_; ++k) {
                                const T* curr_est = buf_est.data() + k * in_channels_;
                                T* next_est = buf_est.data() + (k+1) * in_channels_;
                                T* curr_gate = buf_gate.data() + k * in_channels_;
                                T* curr_act = buf_act.data() + k * in_channels_;

                                for(size_t c=0; c<in_channels_; ++c) {
                                    T sum_sq = 0;
                                    if (c>=2) sum_sq += curr_est[c-2]*curr_est[c-2];
                                    if (c>=1) sum_sq += curr_est[c-1]*curr_est[c-1];
                                    sum_sq += curr_est[c]*curr_est[c];
                                    if (c + 1 < in_channels_) sum_sq += curr_est[c+1]*curr_est[c+1];
                                    if (c + 2 < in_channels_) sum_sq += curr_est[c+2]*curr_est[c+2];

                                    T mag_x = std::sqrt(sum_sq / 5.0f + 1e-10f);

                                    const T* w = srig_w_ptr + c * 5;
                                    T dot = 0;
                                    if (c>=2) dot += w[0] * curr_est[c-2];
                                    if (c>=1) dot += w[1] * curr_est[c-1];
                                    dot += w[2] * curr_est[c];
                                    if (c + 1 < in_channels_) dot += w[3] * curr_est[c+1];
                                    if (c + 2 < in_channels_) dot += w[4] * curr_est[c+2];

                                    T mag_w = mag_w_ptr[c];
                                    T cosine = dot / (mag_x * mag_w + 1e-6f);
                                    T gain = std::sqrt(mag_x + 1e-10f);
                                    T bias = srig_b_ptr[c];

                                    T linear = cosine * gain + bias;
                                    curr_act[c] = linear;

                                    T act = (linear > 0) ? linear : 0.0f;
                                    T gate = 1.0f / (1.0f + std::exp(-act));
                                    curr_gate[c] = gate;
                                    next_est[c] = est_0[c] * gate;
                                }
                            }

                            const T* final_gate = buf_gate.data() + (srig_iterations_ - 1) * in_channels_;
                            for(size_t c=0; c<in_channels_; ++c) {
                                buf_in[c] *= final_gate[c];
                            }
                        }

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

                        for(size_t c=0; c<out_channels_; ++c) {
                            T v = buf_out[c];
                            if (v < 0) v = 0;
                            out_ptr[out_idx + c] = v;
                        }
                    }
                }
            }
        }

        output = group_norm_->forward(output);
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // ... (existing backward logic unchanged)
        // Note: For full speedup, backward should also be vectorized, but priority is forward speedup for now.
        // We preserved eyes_out_cached_ so backward logic holds.
        // Copying previous backward code exactly.

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

        T* g_srig_w = use_slm_ ? grad_srig_weights_.data() : nullptr;
        T* g_srig_b = use_slm_ ? grad_srig_bias_.data() : nullptr;
        const T* srig_w_ptr = use_slm_ ? srig_weights_.data() : nullptr;
        const T* srig_b_ptr = use_slm_ ? srig_bias_.data() : nullptr;

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
                             }
                             size_t out_idx = ((n*H_out + h)*W_out + w)*out_channels_;
                             for(size_t c=0; c<out_channels_; ++c) gn_in.data()[out_idx+c] = buf_out[c];
                        }
                    }
                }
             }

             Tensor<T> gn_out = group_norm_->forward(gn_in);
             Tensor<T> d_gn_out = grad_output;
             T* d_gn_ptr = d_gn_out.data();
             const T* gn_out_ptr = gn_out.data();

             for(size_t i=0; i<d_gn_out.size(); ++i) {
                 if (gn_out_ptr[i] <= 0) d_gn_ptr[i] = 0;
             }
             d_mixer_out = group_norm_->backward(d_gn_out);
        }

        const T* d_mix_ptr = d_mixer_out.data();

        std::vector<T> mag_w_cache(in_channels_);
        if (use_slm_) {
            for(size_t c=0; c<in_channels_; ++c) {
                 const T* w = srig_w_ptr + c * 5;
                 T sum_sq = 0;
                 for(int k=0; k<5; ++k) sum_sq += w[k]*w[k];
                 mag_w_cache[c] = std::sqrt(sum_sq + 1e-8f);
            }
        }
        const T* mag_w_ptr = mag_w_cache.data();

        #pragma omp parallel
        {
            std::vector<T> local_g_scale(in_channels_, 0);
            std::vector<T> local_gw_L(in_channels_, 0);
            std::vector<T> local_gw_C(in_channels_, 0);
            std::vector<T> local_gw_R(in_channels_, 0);

            std::vector<T> local_g_srig_w, local_g_srig_b;
            if (use_slm_) {
                local_g_srig_w.resize(in_channels_ * 5, 0.0);
                local_g_srig_b.resize(in_channels_, 0.0);
            }

            std::vector<T, core::AlignedAllocator<T>> buf_grad(std::max(in_channels_, out_channels_));
            std::vector<T, core::AlignedAllocator<T>> buf_eyes(in_channels_);
            std::vector<T, core::AlignedAllocator<T>> d_eyes(in_channels_);

            std::vector<T, core::AlignedAllocator<T>> buf_est;
            std::vector<T, core::AlignedAllocator<T>> buf_gate;
            std::vector<T, core::AlignedAllocator<T>> d_est;
            std::vector<T, core::AlignedAllocator<T>> d_gate;
            std::vector<T, core::AlignedAllocator<T>> buf_act;

            std::vector<T, core::AlignedAllocator<T>> buf_temp;
            if (use_slm_ || use_sequency_) buf_temp.resize(in_channels_);

            if (use_slm_) {
                buf_est.resize((srig_iterations_ + 1) * in_channels_);
                buf_gate.resize(srig_iterations_ * in_channels_);
                buf_act.resize(srig_iterations_ * in_channels_);
                d_est.resize((srig_iterations_ + 1) * in_channels_);
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

                        if(use_ifwht_) algo::WHT::fwht_1d(buf_grad.data(), out_channels_);

                        size_t idx = ((n*H_out + h)*W_out + w)*in_channels_;
                        for(size_t c=0; c<in_channels_; ++c) buf_eyes[c] = eyes_ptr[idx+c];

                        algo::WHT::fwht_1d(buf_eyes.data(), in_channels_);

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

                        if (in_channels_ == out_channels_) {
                            const T* w_L = mix_w;
                            const T* w_C = mix_w + in_channels_;
                            const T* w_R = mix_w + 2 * in_channels_;
                            for(size_t c=0; c<in_channels_; ++c) {
                                T d_out = buf_grad[c];
                                T val_c = buf_eyes[c];
                                T val_prev = (c == 0) ? 0 : buf_eyes[c - 1];
                                T val_next = (c == in_channels_ - 1) ? 0 : buf_eyes[c + 1];

                                local_gw_C[c] += d_out * val_c;
                                local_gw_L[c] += d_out * val_prev;
                                local_gw_R[c] += d_out * val_next;
                            }
                            std::fill(d_eyes.begin(), d_eyes.end(), 0);
                            for(size_t c=0; c<in_channels_; ++c) {
                                T d_out = buf_grad[c];
                                d_eyes[c] += d_out * w_C[c];
                                if (c < in_channels_ - 1) d_eyes[c+1] += d_out * w_R[c];
                                if (c > 0) d_eyes[c-1] += d_out * w_L[c];
                            }
                        } else {
                             std::fill(d_eyes.begin(), d_eyes.end(), 0);
                             size_t min_c = std::min(in_channels_, out_channels_);
                             const T* w_L = mix_w;
                             const T* w_C = mix_w + in_channels_;
                             const T* w_R = mix_w + 2 * in_channels_;
                             for(size_t c=0; c<min_c; ++c) {
                                T d_out = buf_grad[c];
                                T val_c = buf_eyes[c];
                                T val_prev = (c == 0) ? 0 : buf_eyes[c - 1];
                                T val_next = (c == in_channels_ - 1) ? 0 : buf_eyes[c + 1];
                                local_gw_C[c] += d_out * val_c;
                                local_gw_L[c] += d_out * val_prev;
                                local_gw_R[c] += d_out * val_next;

                                d_eyes[c] += d_out * w_C[c];
                                if (c < in_channels_ - 1) d_eyes[c+1] += d_out * w_R[c];
                                if (c > 0) d_eyes[c-1] += d_out * w_L[c];
                             }
                        }

                        if (use_slm_) {
                            T* est_0 = buf_est.data();
                            for(size_t c=0; c<in_channels_; ++c) est_0[c] = std::abs(buf_eyes[c]);

                            for(size_t k=0; k<srig_iterations_; ++k) {
                                const T* curr_est = buf_est.data() + k * in_channels_;
                                T* next_est = buf_est.data() + (k+1) * in_channels_;
                                T* curr_gate = buf_gate.data() + k * in_channels_;
                                T* curr_act = buf_act.data() + k * in_channels_;

                                for(size_t c=0; c<in_channels_; ++c) {
                                    T sum_sq = 0;
                                    if (c>=2) sum_sq += curr_est[c-2]*curr_est[c-2];
                                    if (c>=1) sum_sq += curr_est[c-1]*curr_est[c-1];
                                    sum_sq += curr_est[c]*curr_est[c];
                                    if (c + 1 < in_channels_) sum_sq += curr_est[c+1]*curr_est[c+1];
                                    if (c + 2 < in_channels_) sum_sq += curr_est[c+2]*curr_est[c+2];
                                    T mag_x = std::sqrt(sum_sq / 5.0f + 1e-10f);

                                    const T* w = srig_w_ptr + c * 5;
                                    T dot = 0;
                                    if (c>=2) dot += w[0] * curr_est[c-2];
                                    if (c>=1) dot += w[1] * curr_est[c-1];
                                    dot += w[2] * curr_est[c];
                                    if (c + 1 < in_channels_) dot += w[3] * curr_est[c+1];
                                    if (c + 2 < in_channels_) dot += w[4] * curr_est[c+2];

                                    T mag_w = mag_w_ptr[c];
                                    T cosine = dot / (mag_x * mag_w + 1e-6f);
                                    T gain = std::sqrt(mag_x + 1e-10f);
                                    T bias = srig_b_ptr[c];

                                    T linear = cosine * gain + bias;
                                    curr_act[c] = linear;
                                    T act = (linear > 0) ? linear : 0.0f;
                                    curr_gate[c] = 1.0f / (1.0f + std::exp(-act));
                                    next_est[c] = est_0[c] * curr_gate[c];
                                }
                            }
                        }

                        const T* final_gate = use_slm_ ? (buf_gate.data() + (srig_iterations_ - 1) * in_channels_) : nullptr;

                        for(size_t c=0; c<in_channels_; ++c) {
                            T u = buf_eyes[c];
                            T g = use_slm_ ? final_gate[c] : 1.0f;
                            T val = u * g;

                            if (training_ && spectral_dropout_rate_ > 0.0f) {
                                val *= (dropout_mask_[c] * dropout_scale);
                            }

                            local_g_scale[c] += d_eyes[c] * val;
                            d_eyes[c] *= scale_ptr[c];

                            if (training_ && spectral_dropout_rate_ > 0.0f) {
                                d_eyes[c] *= (dropout_mask_[c] * dropout_scale);
                            }
                        }

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

                        if (use_slm_) {
                            std::fill(d_est.begin(), d_est.end(), 0.0f);
                            for(size_t c=0; c<in_channels_; ++c) {
                                T g = final_gate[c];
                                T u = buf_eyes[c];
                                d_gate[c] = d_eyes[c] * u;
                                d_eyes[c] = d_eyes[c] * g;
                            }

                            for(int k = (int)srig_iterations_ - 1; k >= 0; --k) {
                                // IMPORTANT: In backward recompute, these MUST be mutable T* because we are writing to them!
                                // The previous code declared them as const T* which caused errors.
                                T* curr_gate = buf_gate.data() + k * in_channels_;
                                const T* curr_est = buf_est.data() + k * in_channels_; // read-only here? No, recompute writes to next_est
                                T* curr_act = buf_act.data() + k * in_channels_;
                                const T* est_0 = buf_est.data();
                                T* next_est = buf_est.data() + (k+1) * in_channels_; // Needed for recompute output

                                if (k < (int)srig_iterations_ - 1) {
                                    T* d_next_est = d_est.data() + (k+1) * in_channels_;
                                    for(size_t c=0; c<in_channels_; ++c) {
                                        d_gate[c] += d_next_est[c] * est_0[c];
                                    }
                                }

                                T* d_curr_est = d_est.data() + k * in_channels_;

                                for(size_t c=0; c<in_channels_; ++c) {
                                    T sum_sq = 0;
                                    if (c>=2) sum_sq += curr_est[c-2]*curr_est[c-2];
                                    if (c>=1) sum_sq += curr_est[c-1]*curr_est[c-1];
                                    sum_sq += curr_est[c]*curr_est[c];
                                    if (c + 1 < in_channels_) sum_sq += curr_est[c+1]*curr_est[c+1];
                                    if (c + 2 < in_channels_) sum_sq += curr_est[c+2]*curr_est[c+2];
                                    T mag_x = std::sqrt(sum_sq / 5.0f + 1e-10f);

                                    const T* w = srig_w_ptr + c * 5;
                                    T dot = 0;
                                    if (c>=2) dot += w[0] * curr_est[c-2];
                                    if (c>=1) dot += w[1] * curr_est[c-1];
                                    dot += w[2] * curr_est[c];
                                    if (c + 1 < in_channels_) dot += w[3] * curr_est[c+1];
                                    if (c + 2 < in_channels_) dot += w[4] * curr_est[c+2];

                                    T mag_w = mag_w_ptr[c];
                                    T cosine = dot / (mag_x * mag_w + 1e-6f);
                                    T gain = std::sqrt(mag_x + 1e-10f);
                                    T bias = srig_b_ptr[c];

                                    T linear = cosine * gain + bias;
                                    curr_act[c] = linear;
                                    T act = (linear > 0) ? linear : 0.0f;
                                    curr_gate[c] = 1.0f / (1.0f + std::exp(-act));
                                    next_est[c] = est_0[c] * curr_gate[c];
                                }
                            }

                            for(size_t c=0; c<in_channels_; ++c) {
                                T u = buf_eyes[c];
                                T sgn = (u > 0) ? 1.0f : ((u < 0) ? -1.0f : 0.0f);
                                d_eyes[c] += d_est.data()[c] * sgn;
                            }
                        }

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
                            int ih_center;
                            int iw_center;
                            if (upscale_ > 1) {
                                ih_center = (up_shift > 0) ? ((int)h >> up_shift) : ((int)h / (int)upscale_);
                                iw_center = (up_shift > 0) ? ((int)w >> up_shift) : ((int)w / (int)upscale_);
                            } else {
                                ih_center = h * stride_;
                                iw_center = w * stride_;
                            }

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
                    for(size_t i=0; i<local_g_srig_w.size(); ++i) g_srig_w[i] += local_g_srig_w[i];
                    for(size_t i=0; i<local_g_srig_b.size(); ++i) g_srig_b[i] += local_g_srig_b[i];
                }
            }
        }
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        std::vector<Tensor<T>*> params = {&packed_weights_, &spectral_scales_, &mixing_weights_};
        if (use_gating_) params.push_back(&oracle_projection_);
        if (use_slm_) {
            params.push_back(&srig_weights_);
            params.push_back(&srig_bias_);
        }
        auto p = group_norm_->parameters();
        params.insert(params.end(), p.begin(), p.end());
        return params;
    }

    std::vector<Tensor<T>*> gradients() override {
        std::vector<Tensor<T>*> grads = {&grad_packed_weights_, &grad_spectral_scales_, &grad_mixing_weights_};
        if (use_gating_) grads.push_back(&grad_oracle_projection_);
        if (use_slm_) {
            grads.push_back(&grad_srig_weights_);
            grads.push_back(&grad_srig_bias_);
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
    float norm_eps_;

    std::vector<int32_t, core::AlignedAllocator<int32_t>> sequency_map_;
    std::vector<int32_t, core::AlignedAllocator<int32_t>> natural_map_;

    Tensor<T> packed_weights_;
    std::vector<T, core::AlignedAllocator<T>> repacked_weights_;

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

    Tensor<T> srig_weights_;
    Tensor<T> srig_bias_;

    Tensor<T> grad_packed_weights_;
    Tensor<T> grad_spectral_scales_;
    Tensor<T> grad_mixing_weights_;
    Tensor<T> grad_oracle_projection_;

    Tensor<T> grad_srig_weights_;
    Tensor<T> grad_srig_bias_;

    Tensor<T> input_cached_;
    Tensor<T> eyes_out_cached_;

    std::mt19937 rng_;
    std::vector<float> dropout_mask_;

    std::unique_ptr<GroupNorm<T>> group_norm_;

    // Pruning
    const Tensor<T>* pruning_mask_ = nullptr;
};

} // namespace layers
} // namespace dreidel
