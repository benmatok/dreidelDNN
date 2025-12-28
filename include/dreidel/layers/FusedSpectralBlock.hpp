#pragma once

#include "Layer.hpp"
#include "../algo/WHT.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <cmath>
#include <iostream>

namespace dreidel {
namespace layers {

/**
 * @brief Fused Spectral Block (Depthwise-Spectral Separable Block).
 *
 * Designed for CPU inference optimization.
 * Pipeline:
 * 1. Depthwise Convolution (Spatial)
 * 2. FWHT (Spectral Projection)
 * 3. Soft Permutation (1D Circular Convolution Stencil)
 * 4. Gating (Spectral Weights)
 * 5. Inverse FWHT
 */
template <typename T>
class FusedSpectralBlock : public Layer<T> {
public:
    FusedSpectralBlock(size_t channels, size_t kernel_size, size_t spectral_dim)
        : channels_(channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          depthwise_weights_({channels, 1, kernel_size, kernel_size}),
          spectral_weights_({1, spectral_dim}), // Diagonal Gating
          soft_perm_weights_({1, 3}), // 3-tap filter
          grad_dw_({channels, 1, kernel_size, kernel_size}),
          grad_spectral_({1, spectral_dim}),
          grad_soft_perm_({1, 3})
    {
        if (spectral_dim != channels) {
            // For now, assume spectral dim = channels for simplicity of fused loop
            // Otherwise we need slicing/padding.
            // prompt says "Project channels to sequency domain", usually C -> C.
        }

        // Initialize Weights
        depthwise_weights_.random(0.0, std::sqrt(2.0 / (kernel_size * kernel_size * channels)));
        spectral_weights_.fill(1.0); // Start as identity-ish

        // Soft perm: [0, 1, 0] identity
        T* sp_ptr = soft_perm_weights_.data();
        sp_ptr[0] = 0.0; sp_ptr[1] = 1.0; sp_ptr[2] = 0.0;

        grad_dw_.fill(0);
        grad_spectral_.fill(0);
        grad_soft_perm_.fill(0);

        repack_weights();
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Input: (Batch, C, H, W) or (Batch, H, W, C)?
        // Assuming NHWC layout for the block input/output as it enables contiguous FWHT.

        input_shape_ = input.shape();
        input_ = input; // Store for backward

        // Assume 4D: [N, H, W, C]
        size_t batch = input_shape_[0];
        size_t height = input_shape_[1];
        size_t width = input_shape_[2];
        size_t channels = input_shape_[3]; // Should match channels_

        Tensor<T> output(input_shape_); // Output same shape

        const T* in_ptr = input.data();
        const T* dw_w_ptr = depthwise_weights_.data();
        const T* sp_w_ptr = soft_perm_weights_.data();
        const T* g_w_ptr = spectral_weights_.data();
        T* out_ptr = output.data();

        int k_rad = kernel_size_ / 2;

        // Buffer for one pixel's channel vector (to perform FWHT in-place)
        // We can parallelize over Batch and Spatial dimensions.
        // Thread-local storage for buffer.

        #pragma omp parallel
        {
            // Use AlignedAllocator to ensure SIMD compatibility
            std::vector<T, core::AlignedAllocator<T>> buffer(channels);
            std::vector<T, core::AlignedAllocator<T>> buffer_temp(channels);

            #pragma omp for collapse(3)
            for (size_t b = 0; b < batch; ++b) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {

                        // 1. Depthwise Convolution Step
                        std::fill(buffer.begin(), buffer.end(), 0);

                        // Naive spatial loop
                        for (int ky = -k_rad; ky <= k_rad; ++ky) {
                            for (int kx = -k_rad; kx <= k_rad; ++kx) {
                                int ih = h + ky;
                                int iw = w + kx;

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    size_t in_idx_base = ((b * height + ih) * width + iw) * channels;

                                    using Ops = hal::ActiveOps;
                                    size_t c = 0;
                                    // Vectorized loop over channels
                                    for (; c + Ops::SIMD_WIDTH <= channels; c += Ops::SIMD_WIDTH) {
                                        auto v_in = Ops::load(in_ptr + in_idx_base + c);
                                        auto v_acc = Ops::load(buffer.data() + c);

                                        int k_idx = (ky + k_rad) * kernel_size_ + (kx + k_rad);
                                        auto v_w = Ops::load(repacked_dw_weights_.data() + k_idx * channels + c);

                                        // v_acc += v_in * v_w
                                        auto prod = Ops::mul(v_in, v_w);
                                        v_acc = Ops::add(v_acc, prod);

                                        Ops::store(buffer.data() + c, v_acc);
                                    }

                                    // Scalar tail
                                    for (; c < channels; ++c) {
                                        int k_idx = (ky + k_rad) * kernel_size_ + (kx + k_rad);
                                        T w_val = repacked_dw_weights_[k_idx * channels + c];
                                        buffer[c] += in_ptr[in_idx_base + c] * w_val;
                                    }
                                }
                            }
                        }

                        // 2. FWHT (Spectral Step)
                        algo::WHT::fwht_1d(buffer.data(), channels);

                        // 3. Soft Permutation (1D Circular Conv)
                        T w0 = sp_w_ptr[0];
                        T w1 = sp_w_ptr[1];
                        T w2 = sp_w_ptr[2];

                        std::copy(buffer.begin(), buffer.end(), buffer_temp.begin());

                        for (size_t i = 0; i < channels; ++i) {
                            size_t prev = (i == 0) ? channels - 1 : i - 1;
                            size_t next = (i == channels - 1) ? 0 : i + 1;
                            buffer[i] = w0 * buffer_temp[prev] + w1 * buffer_temp[i] + w2 * buffer_temp[next];
                        }

                        // 4. Gating (Element-wise mul)
                        using Ops = hal::ActiveOps;
                        size_t c = 0;
                        for (; c + Ops::SIMD_WIDTH <= channels; c += Ops::SIMD_WIDTH) {
                            auto v_b = Ops::load(buffer.data() + c);
                            auto v_g = Ops::load(g_w_ptr + c);
                            auto v_res = Ops::mul(v_b, v_g);
                            Ops::store(buffer.data() + c, v_res);
                        }
                        for (; c < channels; ++c) {
                            buffer[c] *= g_w_ptr[c];
                        }

                        // 5. Inverse FWHT
                        algo::WHT::fwht_1d(buffer.data(), channels);
                        // Scale
                        T scale = 1.0f / channels;
                        for (size_t i = 0; i < channels; ++i) buffer[i] *= scale;

                        // 6. Store to Output
                        size_t out_idx = ((b * height + h) * width + w) * channels;
                        c = 0;
                        for (; c + Ops::SIMD_WIDTH <= channels; c += Ops::SIMD_WIDTH) {
                            auto v = Ops::load(buffer.data() + c);
                            Ops::store(out_ptr + out_idx + c, v);
                        }
                        for (; c < channels; ++c) {
                            out_ptr[out_idx + c] = buffer[c];
                        }
                    }
                }
            }
        }

        return output;
    }

    // Explicitly expose backward for training
    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Assume grad_output: (N, H, W, C)
        Tensor<T> grad_input(input_shape_);
        grad_input.fill(0);

        const T* go_ptr = grad_output.data();
        size_t batch = input_shape_[0];
        size_t height = input_shape_[1];
        size_t width = input_shape_[2];
        size_t channels = channels_;

        T* gi_ptr = grad_input.data();

        // Weight gradients reset
        grad_dw_.fill(0);
        grad_spectral_.fill(0);
        grad_soft_perm_.fill(0);

        // Buffer reuse
        #pragma omp parallel
        {
            std::vector<T, core::AlignedAllocator<T>> buffer(channels);
            std::vector<T, core::AlignedAllocator<T>> buffer_temp(channels);

            // Thread-local grad accumulators
            std::vector<T> t_grad_dw(channels_ * kernel_size_ * kernel_size_, 0);
            std::vector<T> t_grad_spectral(spectral_dim_, 0);
            std::vector<T> t_grad_sp(3, 0);

            #pragma omp for collapse(3)
            for (size_t b = 0; b < batch; ++b) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        size_t idx = ((b * height + h) * width + w) * channels;

                        // Load dL/dy
                        for(size_t c=0; c<channels; ++c) buffer[c] = go_ptr[idx+c];

                        // 1. dL/dz = IFWHT(dL/dy)
                        algo::WHT::fwht_1d(buffer.data(), channels);
                        T scale = 1.0f / channels;
                        for(size_t c=0; c<channels; ++c) buffer[c] *= scale;

                        // 2. dL/dw = dL/dz * g
                        // Recompute Forward partial
                        std::vector<T, core::AlignedAllocator<T>> fwd_buffer(channels);
                        std::fill(fwd_buffer.begin(), fwd_buffer.end(), 0);

                        // Recompute Depthwise (Need input access)
                        const T* in_ptr = input_.data(); // Need stored input
                        int k_rad = kernel_size_ / 2;

                        for (int ky = -k_rad; ky <= k_rad; ++ky) {
                            for (int kx = -k_rad; kx <= k_rad; ++kx) {
                                int ih = h + ky;
                                int iw = w + kx;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    size_t in_idx_base = ((b * height + ih) * width + iw) * channels;
                                    int k_idx = (ky + k_rad) * kernel_size_ + (kx + k_rad);
                                    for(size_t c=0; c<channels; ++c) {
                                        fwd_buffer[c] += in_ptr[in_idx_base+c] * repacked_dw_weights_[k_idx*channels+c];
                                    }
                                }
                            }
                        }

                        algo::WHT::fwht_1d(fwd_buffer.data(), channels);

                        // Soft Perm Forward to get 'w'
                        std::vector<T, core::AlignedAllocator<T>> fwd_temp(channels);
                        std::copy(fwd_buffer.begin(), fwd_buffer.end(), fwd_temp.begin());
                        T w0 = soft_perm_weights_[0], w1 = soft_perm_weights_[1], w2 = soft_perm_weights_[2];
                        for(size_t i=0; i<channels; ++i) {
                            size_t prev = (i == 0) ? channels - 1 : i - 1;
                            size_t next = (i == channels - 1) ? 0 : i + 1;
                            fwd_buffer[i] = w0 * fwd_temp[prev] + w1 * fwd_temp[i] + w2 * fwd_temp[next];
                        }

                        // Gradients for Gating
                        const T* g_ptr = spectral_weights_.data();
                        for(size_t c=0; c<channels; ++c) {
                            T dz = buffer[c]; // dL/dz
                            t_grad_spectral[c] += dz * fwd_buffer[c];
                            buffer[c] = dz * g_ptr[c]; // buffer is now dL/dw
                        }

                        // 3. dL/dv = SoftPerm_Backward(dL/dw)
                        std::copy(buffer.begin(), buffer.end(), buffer_temp.begin()); // buffer_temp is dL/dw

                        for(size_t i=0; i<channels; ++i) {
                            size_t prev = (i == 0) ? channels - 1 : i - 1;
                            size_t next = (i == channels - 1) ? 0 : i + 1;

                            // Grad Weights
                            t_grad_sp[0] += buffer[i] * fwd_temp[prev];
                            t_grad_sp[1] += buffer[i] * fwd_temp[i];
                            t_grad_sp[2] += buffer[i] * fwd_temp[next];

                            // Grad Input (dL/dv)
                            buffer[i] = w2 * buffer_temp[prev] + w1 * buffer_temp[i] + w0 * buffer_temp[next];
                        }
                        // buffer is now dL/dv

                        // 4. dL/du = FWHT(dL/dv)
                        // Symmetric
                        algo::WHT::fwht_1d(buffer.data(), channels);
                        // buffer is now dL/du

                        // 5. dL/dx = Depthwise_Backward(dL/du)
                        for (int ky = -k_rad; ky <= k_rad; ++ky) {
                            for (int kx = -k_rad; kx <= k_rad; ++kx) {
                                int ih = h + ky;
                                int iw = w + kx;
                                int k_idx = (ky + k_rad) * kernel_size_ + (kx + k_rad);

                                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                    size_t in_idx_base = ((b * height + ih) * width + iw) * channels;

                                    // Grad Weights
                                    for(size_t c=0; c<channels; ++c) {
                                        t_grad_dw[k_idx * channels + c] += buffer[c] * in_ptr[in_idx_base + c];
                                    }

                                    // Grad Input (Atomic add required if writing to global)
                                    for(size_t c=0; c<channels; ++c) {
                                        #pragma omp atomic
                                        gi_ptr[in_idx_base + c] += buffer[c] * repacked_dw_weights_[k_idx*channels + c];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Reduce thread local weight grads
            #pragma omp critical
            {
                size_t sz_dw = grad_dw_.size();
                T* gdw = grad_dw_.data();
                for(size_t i=0; i<sz_dw; ++i) gdw[i] += t_grad_dw[i];

                size_t sz_spec = grad_spectral_.size();
                T* gspec = grad_spectral_.data();
                for(size_t i=0; i<sz_spec; ++i) gspec[i] += t_grad_spectral[i];

                T* gsp = grad_soft_perm_.data();
                gsp[0] += t_grad_sp[0];
                gsp[1] += t_grad_sp[1];
                gsp[2] += t_grad_sp[2];
            }
        }

        return grad_input;
    }

    // Repack weights helper
    void repack_weights() {
        size_t C = channels_;
        size_t K = kernel_size_;
        repacked_dw_weights_.resize(K * K * C);
        const T* w_src = depthwise_weights_.data(); // (C, 1, K, K)

        // Target: (K, K, C) -> [ky, kx, c]
        for(size_t c=0; c<C; ++c) {
            for(size_t ky=0; ky<K; ++ky) {
                for(size_t kx=0; kx<K; ++kx) {
                    // src: c*K*K + ky*K + kx
                    T val = w_src[c*K*K + ky*K + kx];
                    // dst: (ky*K + kx)*C + c
                    repacked_dw_weights_[(ky*K + kx)*C + c] = val;
                }
            }
        }
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&depthwise_weights_, &spectral_weights_, &soft_perm_weights_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_dw_, &grad_spectral_, &grad_soft_perm_};
    }

    std::string name() const override { return "FusedSpectralBlock"; }

private:
    size_t channels_;
    size_t kernel_size_;
    size_t spectral_dim_;

    Tensor<T> depthwise_weights_;
    Tensor<T> spectral_weights_;
    Tensor<T> soft_perm_weights_;

    Tensor<T> grad_dw_;
    Tensor<T> grad_spectral_;
    Tensor<T> grad_soft_perm_;

    // Optimization buffers
    std::vector<T, core::AlignedAllocator<T>> repacked_dw_weights_;
    std::vector<size_t> input_shape_;

    mutable Tensor<T> input_;
};

} // namespace layers
} // namespace dreidel
