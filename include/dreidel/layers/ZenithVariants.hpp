#pragma once

#include "Layer.hpp"
#include "Conv2D.hpp"
#include "Dense.hpp"
#include "ZenithBlock.hpp"
#include "../algo/WHT.hpp"
#include <vector>

namespace dreidel {
namespace layers {

/**
 * @brief Zenith Variant 1: Dense Mixer (No FWHT)
 * Replaces FWHT mixing with a standard Dense (1x1 Conv) mixing.
 * Keeps Eyes (Spatial APoT).
 */
template <typename T>
class ZenithDenseMixer : public Layer<T> {
public:
    ZenithDenseMixer(size_t channels, size_t kernel_size)
        : eyes_(channels, kernel_size, channels), // Reuse Zenith logic but we will hack it or reimplement?
          // Reimplementing simplified version to swap mixer is cleaner.
          channels_(channels), kernel_size_(kernel_size),
          spatial_weights_({channels, 1, kernel_size, kernel_size}),
          mixer_weights_({channels, channels}), // Dense mixing
          grad_spatial_({channels, 1, kernel_size, kernel_size}),
          grad_mixer_({channels, channels})
    {
        // Init
        spatial_weights_.random(0.0, std::sqrt(2.0 / (kernel_size * kernel_size * channels)));
        mixer_weights_.random(0.0, std::sqrt(2.0 / channels)); // Xavier

        // APoT Quantize Spatial
        T* sw_ptr = spatial_weights_.data();
        for(size_t i=0; i<spatial_weights_.size(); ++i) sw_ptr[i] = quantize_apot(sw_ptr[i]);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        input_ = input;
        auto shape = input.shape();
        size_t batch = shape[0], H = shape[1], W = shape[2], C = shape[3];
        Tensor<T> output(shape);
        output.fill(0);

        const T* in_ptr = input.data();
        T* out_ptr = output.data();
        const T* sw_ptr = spatial_weights_.data();
        const T* mw_ptr = mixer_weights_.data();
        int k_rad = kernel_size_ / 2;

        std::vector<T> pixel_buf(C);
        std::vector<T> mix_buf(C);

        for(size_t n=0; n<batch; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    // Eyes
                    std::fill(pixel_buf.begin(), pixel_buf.end(), 0);
                    for(int ky=-k_rad; ky<=k_rad; ++ky) {
                        for(int kx=-k_rad; kx<=k_rad; ++kx) {
                            int ih = h + ky; int iw = w + kx;
                            if(ih>=0 && ih<H && iw>=0 && iw<W) {
                                const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                const T* p_w = sw_ptr + k_idx * channels_;
                                for(size_t c=0; c<C; ++c) pixel_buf[c] += lut_mul(p_in[c], p_w[c]);
                            }
                        }
                    }

                    // Dense Mixer: y = x * W
                    std::fill(mix_buf.begin(), mix_buf.end(), 0);
                    for(size_t c_out=0; c_out<C; ++c_out) {
                        T acc = 0;
                        for(size_t c_in=0; c_in<C; ++c_in) {
                            acc += pixel_buf[c_in] * mw_ptr[c_in*C + c_out];
                        }
                        mix_buf[c_out] = acc;
                    }

                    // Write
                    T* p_out = out_ptr + ((n*H + h)*W + w)*C;
                    for(size_t c=0; c<C; ++c) p_out[c] = mix_buf[c];
                }
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Simplified backward
        auto shape = input_.shape();
        size_t batch = shape[0], H = shape[1], W = shape[2], C = shape[3];
        Tensor<T> grad_input(shape); grad_input.fill(0);
        grad_spatial_.fill(0); grad_mixer_.fill(0);

        const T* go_ptr = grad_output.data();
        const T* in_ptr = input_.data();
        T* gi_ptr = grad_input.data();
        T* gsw_ptr = grad_spatial_.data();
        T* gmw_ptr = grad_mixer_.data();
        const T* sw_ptr = spatial_weights_.data();
        const T* mw_ptr = mixer_weights_.data();
        int k_rad = kernel_size_ / 2;

        std::vector<T> d_vec(C);
        std::vector<T> d_eyes(C);
        std::vector<T> eyes_out(C); // Need recompute

        for(size_t n=0; n<batch; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    const T* p_go = go_ptr + ((n*H + h)*W + w)*C;
                    for(size_t c=0; c<C; ++c) d_vec[c] = p_go[c];

                    // Recompute Eyes Output for Mixer Grad
                    std::fill(eyes_out.begin(), eyes_out.end(), 0);
                    for(int ky=-k_rad; ky<=k_rad; ++ky) {
                        for(int kx=-k_rad; kx<=k_rad; ++kx) {
                            int ih = h + ky; int iw = w + kx;
                            if(ih>=0 && ih<H && iw>=0 && iw<W) {
                                const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                const T* p_w = sw_ptr + k_idx * channels_;
                                for(size_t c=0; c<C; ++c) eyes_out[c] += lut_mul(p_in[c], p_w[c]);
                            }
                        }
                    }

                    // Backward Mixer
                    // dL/dW = x^T * dy
                    // dL/dx = dy * W^T
                    std::fill(d_eyes.begin(), d_eyes.end(), 0);
                    for(size_t c_out=0; c_out<C; ++c_out) {
                        T dy = d_vec[c_out];
                        for(size_t c_in=0; c_in<C; ++c_in) {
                            gmw_ptr[c_in*C + c_out] += eyes_out[c_in] * dy;
                            d_eyes[c_in] += dy * mw_ptr[c_in*C + c_out];
                        }
                    }

                    // Backward Eyes
                    for(int ky=-k_rad; ky<=k_rad; ++ky) {
                        for(int kx=-k_rad; kx<=k_rad; ++kx) {
                            int ih = h + ky; int iw = w + kx;
                            if(ih>=0 && ih<H && iw>=0 && iw<W) {
                                int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                T* p_gs = gsw_ptr + k_idx * channels_;
                                const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                const T* p_w = sw_ptr + k_idx * channels_;
                                T* p_gi = gi_ptr + ((n*H + ih)*W + iw)*C;

                                for(size_t c=0; c<C; ++c) {
                                    T dy = d_eyes[c];
                                    p_gs[c] += dy * p_in[c];
                                    p_gi[c] += dy * p_w[c];
                                }
                            }
                        }
                    }
                }
            }
        }
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {&spatial_weights_, &mixer_weights_}; }
    std::vector<Tensor<T>*> gradients() override { return {&grad_spatial_, &grad_mixer_}; }
    std::string name() const override { return "ZenithDenseMixer"; }

private:
    ZenithBlock<T> eyes_; // unused wrapper
    size_t channels_, kernel_size_;
    Tensor<T> spatial_weights_, mixer_weights_;
    Tensor<T> grad_spatial_, grad_mixer_;
    Tensor<T> input_;
};

/**
 * @brief Zenith Variant 2: Float Eyes (Standard Depthwise)
 * Replaces APoT/LUT Eyes with standard float Depthwise Conv.
 * Keeps Mixer (FWHT).
 */
template <typename T>
class ZenithFloatEyes : public Layer<T> {
public:
    ZenithFloatEyes(size_t channels, size_t kernel_size, size_t spectral_dim)
        : channels_(channels), kernel_size_(kernel_size),
          spatial_weights_({channels, 1, kernel_size, kernel_size}),
          spectral_scales_({1, channels}),
          perm_indices_(channels),
          grad_spatial_({channels, 1, kernel_size, kernel_size}),
          grad_scales_({1, channels})
    {
        // Standard init (no quantization)
        spatial_weights_.random(0.0, std::sqrt(2.0 / (kernel_size * kernel_size * channels)));

        T scale_init = 1.0 / std::sqrt(static_cast<T>(channels));
        spectral_scales_.fill(scale_init);

        std::iota(perm_indices_.begin(), perm_indices_.end(), 0);
        std::random_device rd; std::mt19937 g(rd());
        std::shuffle(perm_indices_.begin(), perm_indices_.end(), g);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        input_ = input;
        auto shape = input.shape();
        size_t batch = shape[0], H = shape[1], W = shape[2], C = shape[3];
        Tensor<T> output(shape); output.fill(0);

        const T* in_ptr = input.data();
        T* out_ptr = output.data();
        const T* sw_ptr = spatial_weights_.data();
        const T* scale_ptr = spectral_scales_.data();
        int k_rad = kernel_size_ / 2;

        std::vector<T> pixel_buf(C);
        std::vector<T> temp_buf(C);

        for(size_t n=0; n<batch; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    // Standard Depthwise (Float)
                    std::fill(pixel_buf.begin(), pixel_buf.end(), 0);
                    for(int ky=-k_rad; ky<=k_rad; ++ky) {
                        for(int kx=-k_rad; kx<=k_rad; ++kx) {
                            int ih = h + ky; int iw = w + kx;
                            if(ih>=0 && ih<H && iw>=0 && iw<W) {
                                const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                const T* p_w = sw_ptr + k_idx * channels_;
                                for(size_t c=0; c<C; ++c) pixel_buf[c] += p_in[c] * p_w[c]; // FLOAT MUL
                            }
                        }
                    }

                    // Mixer (FWHT)
                    for(size_t i=0; i<C; ++i) temp_buf[i] = pixel_buf[i];
                    for(size_t i=0; i<C; ++i) pixel_buf[i] = temp_buf[perm_indices_[i]];

                    algo::WHT::fwht_1d(pixel_buf.data(), C);

                    for(size_t i=0; i<C; ++i) pixel_buf[i] *= scale_ptr[i]; // FLOAT SCALE

                    T* p_out = out_ptr + ((n*H + h)*W + w)*C;
                    for(size_t c=0; c<C; ++c) p_out[c] = pixel_buf[c];
                }
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // ... (Similar logic, omitted for brevity, assumes essentially same as ZenithBlock but float mul)
        // Implementing basic backward to allow training
        auto shape = input_.shape();
        size_t batch = shape[0], H = shape[1], W = shape[2], C = shape[3];
        Tensor<T> grad_input(shape); grad_input.fill(0);
        grad_spatial_.fill(0); grad_scales_.fill(0);

        T* gs_ptr = grad_spatial_.data();
        T* gscale_ptr = grad_scales_.data();
        const T* sw_ptr = spatial_weights_.data();
        const T* scale_ptr = spectral_scales_.data();
        int k_rad = kernel_size_ / 2;

        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();
        const T* in_ptr = input_.data();

        std::vector<T> d_vec(C), eyes_out(C), mixer_in(C), d_unperm(C);

        for(size_t n=0; n<batch; ++n) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    const T* p_go = go_ptr + ((n*H + h)*W + w)*C;
                    for(size_t c=0; c<C; ++c) d_vec[c] = p_go[c];

                    // Recompute forward eyes (Float)
                    std::fill(eyes_out.begin(), eyes_out.end(), 0);
                    for(int ky=-k_rad; ky<=k_rad; ++ky) {
                        for(int kx=-k_rad; kx<=k_rad; ++kx) {
                            int ih = h + ky; int iw = w + kx;
                            if(ih>=0 && ih<H && iw>=0 && iw<W) {
                                const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                const T* p_w = sw_ptr + k_idx * channels_;
                                for(size_t c=0; c<C; ++c) eyes_out[c] += p_in[c] * p_w[c];
                            }
                        }
                    }

                    for(size_t i=0; i<C; ++i) mixer_in[i] = eyes_out[perm_indices_[i]];
                    algo::WHT::fwht_1d(mixer_in.data(), C);

                    for(size_t c=0; c<C; ++c) {
                        gscale_ptr[c] += d_vec[c] * mixer_in[c];
                        d_vec[c] *= scale_ptr[c];
                    }

                    algo::WHT::fwht_1d(d_vec.data(), C);

                    for(size_t i=0; i<C; ++i) d_unperm[perm_indices_[i]] = d_vec[i];

                    for(int ky=-k_rad; ky<=k_rad; ++ky) {
                        for(int kx=-k_rad; kx<=k_rad; ++kx) {
                            int ih = h + ky; int iw = w + kx;
                            if(ih>=0 && ih<H && iw>=0 && iw<W) {
                                int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                T* p_gs = gs_ptr + k_idx * channels_;
                                const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                const T* p_w = sw_ptr + k_idx * channels_;
                                T* p_gi = gi_ptr + ((n*H + ih)*W + iw)*C;

                                for(size_t c=0; c<C; ++c) {
                                    T dy = d_unperm[c];
                                    p_gs[c] += dy * p_in[c];
                                    p_gi[c] += dy * p_w[c];
                                }
                            }
                        }
                    }
                }
            }
        }
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {&spatial_weights_, &spectral_scales_}; }
    std::vector<Tensor<T>*> gradients() override { return {&grad_spatial_, &grad_scales_}; }
    std::string name() const override { return "ZenithFloatEyes"; }

private:
    size_t channels_, kernel_size_;
    Tensor<T> spatial_weights_, spectral_scales_;
    std::vector<int> perm_indices_;
    Tensor<T> grad_spatial_, grad_scales_;
    Tensor<T> input_;
};

} // namespace layers
} // namespace dreidel
