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

        if (upscale_ > 1) {
            H_out = H * upscale_;
            W_out = W * upscale_;
        }

        output_cached_pre_relu_ = Tensor<T>({N, H_out, W_out, out_channels_});
        Tensor<T> output({N, H_out, W_out, out_channels_});
        T* out_ptr = output.data();
        T* pre_relu_ptr = output_cached_pre_relu_.data();

        const T* in_ptr = input.data();

        eyes_out_cached_ = Tensor<T>({N, H_out, W_out, in_channels_});
        eyes_out_cached_.fill(0);
        T* eyes_ptr = eyes_out_cached_.data();
        const T* w_ptr = packed_weights_.data();
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

        const T* scale_ptr = spectral_scales_.data();
        const T* bias_ptr = bias_.data();

        #pragma omp parallel
        {
            std::vector<T> buf(in_channels_);
            std::vector<T> buf_out(out_channels_);
            #pragma omp for collapse(3)
            for(size_t n=0; n<N; ++n) {
                for(size_t h=0; h<H_out; ++h) {
                    for(size_t w=0; w<W_out; ++w) {
                        size_t idx = ((n*H_out + h)*W_out + w);
                        const T* e_p = eyes_ptr + idx * in_channels_;
                        T* o_p = out_ptr + idx * out_channels_;
                        T* pr_p = pre_relu_ptr + idx * out_channels_;

                        for(size_t c=0; c<in_channels_; ++c) buf[c] = e_p[c];
                        algo::WHT::fwht_1d(buf.data(), in_channels_);

                        for(size_t c=0; c<in_channels_; ++c) buf[c] *= scale_ptr[c];

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

        T* g_bias_ptr = grad_bias_.data();
        T* g_scale_ptr = grad_spectral_scales_.data();
        T* g_weights_ptr = grad_packed_weights_.data();

        Tensor<T> grad_eyes({N, H_out, W_out, in_channels_});
        T* ge_ptr = grad_eyes.data();

        static bool printed_debug = false;

        #pragma omp parallel
        {
            std::vector<T> buf(out_channels_);
            std::vector<T> buf_in(in_channels_);

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
        return {&packed_weights_, &spectral_scales_, &bias_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_packed_weights_, &grad_spectral_scales_, &grad_bias_};
    }

    std::string name() const override { return "ZenithBlock"; }

private:
#ifdef __AVX2__
    // AVX methods omitted for clarity/size constraints
    // This assumes they are not called in the above implementation
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
