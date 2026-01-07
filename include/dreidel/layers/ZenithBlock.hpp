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
                const std::string& init_scheme = "he", bool use_slm = false, bool use_sequency = false)
        : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          use_ifwht_(use_ifwht), use_gating_(use_gating), stride_(stride), upscale_(upscale), use_slm_(use_slm), use_sequency_(use_sequency),
          packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          spectral_scales_({1, in_channels}),
          mixing_weights_({3, in_channels}),
          oracle_projection_({1, in_channels}),
          srig_weights_({5, in_channels}),
          srig_bias_({1, in_channels}),
          srig_gamma_({1, in_channels}),

          grad_packed_weights_({in_channels, 1, kernel_size, kernel_size}),
          grad_spectral_scales_({1, in_channels}),
          grad_mixing_weights_({3, in_channels}),
          grad_oracle_projection_({1, in_channels}),
          grad_srig_weights_({5, in_channels}),
          grad_srig_bias_({1, in_channels}),
          grad_srig_gamma_({1, in_channels}),
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
            // Bias: 0.5 (Barrier)
            srig_bias_.fill(0.5f);
            // Gamma: 0.1 (Sensitivity)
            srig_gamma_.fill(0.1f);
        }

        // Zero Grads
        grad_packed_weights_.fill(0);
        grad_spectral_scales_.fill(0);
        grad_mixing_weights_.fill(0);
        grad_oracle_projection_.fill(0);
        grad_srig_weights_.fill(0);
        grad_srig_bias_.fill(0);
        grad_srig_gamma_.fill(0);

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
    }

    void set_spectral_dropout(float rate) {
        spectral_dropout_rate_ = rate;
    }

    void set_training(bool training) {
        training_ = training;
    }

    void set_monitor_sparsity(bool monitor) {
        monitor_sparsity_ = monitor;
    }

    float get_last_sparsity() const {
        return last_sparsity_;
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

        const T* srig_w_ptr = use_slm_ ? srig_weights_.data() : nullptr;
        const T* srig_b_ptr = use_slm_ ? srig_bias_.data() : nullptr;
        const T* srig_g_ptr = use_slm_ ? srig_gamma_.data() : nullptr;

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

        if (monitor_sparsity_) {
            debug_zeros_ = 0;
            debug_total_ = 0;
        }

        #pragma omp parallel
        {
            std::vector<T, core::AlignedAllocator<T>> buf_in(in_channels_), buf_out(out_channels_);
            std::vector<T, core::AlignedAllocator<T>> buf_est;
            std::vector<T, core::AlignedAllocator<T>> buf_gate;
            std::vector<T, core::AlignedAllocator<T>> buf_act;

            size_t local_zeros = 0;
            size_t local_total = 0;

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
                                    // T gain = std::sqrt(mag_x + 1e-10f); // Unused

                                    T bias_val = srig_b_ptr[c];
                                    T gamma_val = srig_g_ptr[c];
                                    T est_mag = std::abs(curr_est[c]);

                                    T bias_reduction = bias_val - (gamma_val * est_mag);
                                    T effective_bias = (bias_reduction > 0) ? bias_reduction : 0.0f;
                                    T activation = cosine - effective_bias;

                                    T gate = (activation > 0) ? activation : 0.0f;

                                    if (monitor_sparsity_ && n==0 && h_out==0 && w_out==0 && c==0) {
                                        std::cout << "DEBUG k=" << k
                                                  << " est=" << curr_est[c]
                                                  << " bias=" << bias_val
                                                  << " gamma=" << gamma_val
                                                  << " eff_bias=" << effective_bias
                                                  << " act=" << activation
                                                  << " gate=" << gate << std::endl;
                                    }

                                    curr_act[c] = activation;
                                    curr_gate[c] = gate;
                                    next_est[c] = est_0[c] * gate;
                                }
                            }

                            const T* final_gate = buf_gate.data() + (srig_iterations_ - 1) * in_channels_;
                            for(size_t c=0; c<in_channels_; ++c) {
                                buf_in[c] *= final_gate[c];
                            }

                            if (monitor_sparsity_) {
                                for(size_t c=0; c<in_channels_; ++c) {
                                    if (final_gate[c] <= 1e-3f) local_zeros++;
                                }
                                local_total += in_channels_;
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
            if (monitor_sparsity_) {
                #pragma omp atomic
                debug_zeros_ += local_zeros;
                #pragma omp atomic
                debug_total_ += local_total;
            }
        }

        if (monitor_sparsity_ && debug_total_ > 0) {
            last_sparsity_ = (float)debug_zeros_ / (float)debug_total_;
        }

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

        T* g_srig_w = use_slm_ ? grad_srig_weights_.data() : nullptr;
        T* g_srig_b = use_slm_ ? grad_srig_bias_.data() : nullptr;
        T* g_srig_g = use_slm_ ? grad_srig_gamma_.data() : nullptr;
        const T* srig_w_ptr = use_slm_ ? srig_weights_.data() : nullptr;
        const T* srig_b_ptr = use_slm_ ? srig_bias_.data() : nullptr;
        const T* srig_g_ptr = use_slm_ ? srig_gamma_.data() : nullptr;

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

            std::vector<T> local_g_srig_w, local_g_srig_b, local_g_srig_g;
            if (use_slm_) {
                local_g_srig_w.resize(in_channels_ * 5, 0.0);
                local_g_srig_b.resize(in_channels_, 0.0);
                local_g_srig_g.resize(in_channels_, 0.0);
            }

            std::vector<T, core::AlignedAllocator<T>> buf_grad(std::max(in_channels_, out_channels_));
            std::vector<T, core::AlignedAllocator<T>> buf_eyes(in_channels_);
            std::vector<T, core::AlignedAllocator<T>> d_eyes(in_channels_);

            std::vector<T, core::AlignedAllocator<T>> buf_est;
            std::vector<T, core::AlignedAllocator<T>> buf_gate;
            std::vector<T, core::AlignedAllocator<T>> d_est;
            std::vector<T, core::AlignedAllocator<T>> d_gate;

            // Zenith-SRIG specific
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

                        // Mixer Backward: buf_grad -> d_eyes
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
                            // Compute d_eyes (dL/dInput)
                            std::fill(d_eyes.begin(), d_eyes.end(), 0);
                            for(size_t c=0; c<in_channels_; ++c) {
                                T d_out = buf_grad[c];
                                d_eyes[c] += d_out * w_C[c];
                                if (c < in_channels_ - 1) d_eyes[c+1] += d_out * w_R[c]; // w_R multiplies next
                                if (c > 0) d_eyes[c-1] += d_out * w_L[c]; // w_L multiplies prev
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

                        // SRIG Forward Recompute
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
                                    // T gain = std::sqrt(mag_x + 1e-10f); // Unused

                                    T bias_val = srig_b_ptr[c];
                                    T gamma_val = srig_g_ptr[c];
                                    T est_mag = std::abs(curr_est[c]);

                                    T bias_reduction = bias_val - (gamma_val * est_mag);
                                    T effective_bias = (bias_reduction > 0) ? bias_reduction : 0.0f;
                                    T activation = cosine - effective_bias;

                                    curr_act[c] = activation;
                                    curr_gate[c] = (activation > 0) ? activation : 0.0f;
                                    next_est[c] = est_0[c] * curr_gate[c];
                                }
                            }
                        }

                        // Scale Backward
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

                        // Permute d_eyes Nat -> Seq (for SRIG Bwd)
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

                        // SRIG Backward
                        if (use_slm_) {
                            std::fill(d_est.begin(), d_est.end(), 0.0f);

                            for(size_t c=0; c<in_channels_; ++c) {
                                T g = final_gate[c];
                                T u = buf_eyes[c]; // buf_eyes is Natural here if use_sequency=false.
                                // Wait, if use_sequency_, buf_eyes was permuted to Natural before Mixer!
                                // But SRIG ran in Sequency domain (if enabled).
                                // So we need u in Sequency domain?
                                // Let's check Forward.
                                // Eyes -> WHT -> Permute(Nat->Seq) -> SRIG -> Permute(Seq->Nat) -> Scale.
                                // In Backward:
                                // buf_eyes loaded. WHT.
                                // Permute(Nat->Seq).
                                // SRIG Forward Recompute (Sequency Domain).
                                // buf_eyes IS in Sequency Domain (because we did not reverse permute).

                                // WAIT! In forward:
                                // Permute(Nat->Seq). SRIG. Permute(Seq->Nat). Scale.
                                // Scale input is Nat.
                                // buf_eyes (in backward so far) was WHT'd.
                                // Then we check Permute(Nat->Seq) BEFORE SRIG Recompute.
                                // So buf_eyes is in Seq domain here.

                                // BUT! We used buf_eyes for Mixer Backward!
                                // Mixer works in Natural domain (usually).
                                // In Forward: Scale -> Mixer.
                                // Scale input was Natural.
                                // So buf_eyes MUST be Natural for Mixer Backward and Scale Backward.

                                // In Backward recompute:
                                // Eyes -> WHT -> Permute(Nat->Seq) -> SRIG -> Permute(Seq->Nat).
                                // We missed the re-permutation to Natural before Mixer Bwd!
                                // In my previous code:
                                // ... WHT(buf_eyes) ...
                                // Permute(Nat->Seq).
                                // Mixer Bwd using buf_eyes.
                                // Scale Bwd using buf_eyes.

                                // THIS IS WRONG if use_sequency_ is true.
                                // Mixer and Scale operate on Natural domain.
                                // But buf_eyes is Seq.

                                // FIX:
                                // 1. WHT(buf_eyes).
                                // 2. If use_sequency_: Permute(Nat->Seq).
                                // 3. SRIG Forward (Seq domain).
                                // 4. If use_sequency_: Permute(Seq->Nat).
                                // 5. Mixer Backward (Nat).
                                // 6. Scale Backward (Nat).
                                // 7. If use_sequency_: Permute d_eyes (Nat->Seq).
                                // 8. SRIG Backward (Seq).
                                // 9. If use_sequency_: Permute d_eyes (Seq->Nat).

                                // My current implementation of Backward Recompute:
                                // 1. WHT(buf_eyes).
                                // 2. If use_sequency_: Permute(Nat->Seq).
                                // 3. Mixer Bwd (using buf_eyes). <-- WRONG if Seq.
                                // 4. SRIG Fwd.

                                // Also, SRIG Fwd modifies buf_eyes.
                                // But in Bwd, we are recomputing.

                                // Let's correct this flow.
                                // Also d_gate needs `u` which is input to Gate (Seq domain).

                                d_gate[c] = d_eyes[c] * u;
                                d_eyes[c] = d_eyes[c] * g;
                            }

                            for(int k = (int)srig_iterations_ - 1; k >= 0; --k) {
                                const T* curr_gate = buf_gate.data() + k * in_channels_;
                                const T* curr_est = buf_est.data() + k * in_channels_;
                                const T* curr_act = buf_act.data() + k * in_channels_;
                                const T* est_0 = buf_est.data();

                                if (k < (int)srig_iterations_ - 1) {
                                    T* d_next_est = d_est.data() + (k+1) * in_channels_;
                                    for(size_t c=0; c<in_channels_; ++c) {
                                        d_gate[c] += d_next_est[c] * est_0[c];
                                    }
                                }

                                T* d_curr_est = d_est.data() + k * in_channels_;

                                for(size_t c=0; c<in_channels_; ++c) {
                                    // Recompute forward stats (local)
                                    // Use curr_est
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
                                    T eps = 1e-6f;
                                    T denom = mag_x * mag_w + eps;
                                    T cosine = dot / denom;

                                    T act = curr_act[c];

                                    T d_g = d_gate[c];
                                    T d_act = (act > 0) ? d_g : 0.0f;

                                    T bias_val = srig_b_ptr[c];
                                    T gamma_val = srig_g_ptr[c];
                                    T est_mag = std::abs(curr_est[c]);
                                    T bias_reduction = bias_val - (gamma_val * est_mag);

                                    T d_cosine = d_act;
                                    T d_eff_bias = -d_act;

                                    T d_bias_red = (bias_reduction > 0) ? d_eff_bias : 0.0f;

                                    local_g_srig_b[c] += d_bias_red;
                                    local_g_srig_g[c] += d_bias_red * (-est_mag);

                                    T d_est_mag = d_bias_red * (-gamma_val);
                                    T sgn_est = (curr_est[c] > 0) ? 1.0f : ((curr_est[c] < 0) ? -1.0f : 0.0f);
                                    d_curr_est[c] += d_est_mag * sgn_est;

                                    T d_mag_x = d_cosine * (-dot * mag_w) / (denom * denom);

                                    T d_mag_w = d_cosine * (-dot * mag_x) / (denom * denom);
                                    T d_dot = d_cosine / denom;

                                    T* gw = local_g_srig_w.data() + c * 5;
                                    T scale_w = d_mag_w / mag_w;
                                    for(int kw=0; kw<5; ++kw) gw[kw] += scale_w * w[kw];

                                    if (c>=2) gw[0] += d_dot * curr_est[c-2];
                                    if (c>=1) gw[1] += d_dot * curr_est[c-1];
                                    gw[2] += d_dot * curr_est[c];
                                    if (c + 1 < in_channels_) gw[3] += d_dot * curr_est[c+1];
                                    if (c + 2 < in_channels_) gw[4] += d_dot * curr_est[c+2];

                                    if (c>=2) d_curr_est[c-2] += d_dot * w[0];
                                    if (c>=1) d_curr_est[c-1] += d_dot * w[1];
                                    d_curr_est[c] += d_dot * w[2];
                                    if (c + 1 < in_channels_) d_curr_est[c+1] += d_dot * w[3];
                                    if (c + 2 < in_channels_) d_curr_est[c+2] += d_dot * w[4];

                                    T scale_x = d_mag_x / (mag_x * 5.0f + 1e-10f);

                                    if (c>=2) d_curr_est[c-2] += scale_x * curr_est[c-2];
                                    if (c>=1) d_curr_est[c-1] += scale_x * curr_est[c-1];
                                    d_curr_est[c] += scale_x * curr_est[c];
                                    if (c + 1 < in_channels_) d_curr_est[c+1] += scale_x * curr_est[c+1];
                                    if (c + 2 < in_channels_) d_curr_est[c+2] += scale_x * curr_est[c+2];
                                }

                                std::fill(d_gate.begin(), d_gate.end(), 0.0f);
                            }

                            T* d_est_0 = d_est.data();
                            for(size_t k=0; k<srig_iterations_; ++k) {
                                const T* gate_k = buf_gate.data() + k * in_channels_;
                                const T* d_est_k_plus_1 = d_est.data() + (k+1) * in_channels_;
                                for(size_t c=0; c<in_channels_; ++c) {
                                    d_est_0[c] += d_est_k_plus_1[c] * gate_k[c];
                                }
                            }

                            for(size_t c=0; c<in_channels_; ++c) {
                                T u = buf_eyes[c];
                                T sgn = (u > 0) ? 1.0f : ((u < 0) ? -1.0f : 0.0f);
                                d_eyes[c] += d_est_0[c] * sgn;
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
                    for(size_t i=0; i<local_g_srig_w.size(); ++i) g_srig_w[i] += local_g_srig_w[i];
                    for(size_t i=0; i<local_g_srig_b.size(); ++i) g_srig_b[i] += local_g_srig_b[i];
                    for(size_t i=0; i<local_g_srig_g.size(); ++i) g_srig_g[i] += local_g_srig_g[i];
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
            params.push_back(&srig_gamma_);
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
            grads.push_back(&grad_srig_gamma_);
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
    bool monitor_sparsity_ = false;
    mutable float last_sparsity_ = 0.0f;
    mutable size_t debug_zeros_ = 0;
    mutable size_t debug_total_ = 0;

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

    Tensor<T> srig_weights_;
    Tensor<T> srig_bias_;
    Tensor<T> srig_gamma_;

    Tensor<T> grad_packed_weights_;
    Tensor<T> grad_spectral_scales_;
    Tensor<T> grad_mixing_weights_;
    Tensor<T> grad_oracle_projection_;

    Tensor<T> grad_srig_weights_;
    Tensor<T> grad_srig_bias_;
    Tensor<T> grad_srig_gamma_;

    Tensor<T> input_cached_;
    Tensor<T> eyes_out_cached_;

    std::mt19937 rng_;
    std::vector<float> dropout_mask_;

    std::unique_ptr<GroupNorm<T>> group_norm_;
};

} // namespace layers
} // namespace dreidel
