#pragma once

#include "Layer.hpp"
#include "../core/Tensor.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

namespace dreidel {
namespace layers {

// Optimized Conv2D using AVX2 and weight repacking
// Optimized layout: [K, K, In, Out]
// Vectorizes over Output Channels.
template <typename T>
class OptimizedConv2D : public Layer<T> {
public:
    OptimizedConv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0, size_t groups = 1)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding), groups_(groups),
          packed_weights_({kernel_size, kernel_size, in_channels, out_channels}), // Transposed layout
          bias_({1, out_channels})
    {
        if (groups != 1) {
            // TODO: Support groups
            // For now fallback or throw? ZenithTAESD uses groups=1 for Stem/Down/Up.
            // ZenithLite uses GroupConv1x1 (handled by helper).
            // So groups=1 is the main target for Conv2D layers in ZenithTAESD.
            // But wait, ZenithTAESD uses standard Conv2D for Stem (grp1) and Down/Up (grp1).
            // It does NOT use groups > 1 for those layers.
            // Only ZenithLiteBlock uses groups=4, but it calls `group_conv_1x1_avx2` directly.
            // So groups=1 support is sufficient for ZenithTAESD's Conv2D layers.
            if (groups != 1) throw std::invalid_argument("OptimizedConv2D currently only supports groups=1");
        }

        // Initialize weights (standard layout then repack)
        // He Init
        std::vector<T> temp_weights(out_channels * in_channels * kernel_size * kernel_size);
        T stddev = std::sqrt(2.0 / (in_channels * kernel_size * kernel_size));
        // Simple random gen
        for(auto& x : temp_weights) x = (float(rand())/RAND_MAX - 0.5f) * 2 * stddev; // Mock

        bias_.fill(0);

        // Repack: [Out, In, K, K] -> [K, K, In, Out]
        T* packed = packed_weights_.data();
        for(size_t k_idx = 0; k_idx < kernel_size * kernel_size; ++k_idx) {
            for(size_t i = 0; i < in_channels; ++i) {
                for(size_t o = 0; o < out_channels; ++o) {
                    // Src index: o * (In*KK) + i * (KK) + k_idx
                    // Wait. Src layout [Out, In, K, K].
                    // Flattened: o * (In*K*K) + i * (K*K) + k_idx.
                    // But in Conv2D.hpp: ((c_out * C_in_group + c_in_sub) * kernel_size_ + ky) * kernel_size_ + kx
                    // = c_out*(In*KK) + c_in*(KK) + k_idx. Correct.

                    size_t src_idx = o * (in_channels * kernel_size * kernel_size) + i * (kernel_size * kernel_size) + k_idx;

                    // Dst index: k_idx * (In*Out) + i * Out + o
                    size_t dst_idx = k_idx * (in_channels * out_channels) + i * out_channels + o;

                    packed[dst_idx] = temp_weights[src_idx];
                }
            }
        }
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        auto shape = input.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];

        size_t H_out = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
        size_t W_out = (W + 2 * padding_ - kernel_size_) / stride_ + 1;

        Tensor<T> output({N, H_out, W_out, out_channels_});
        forward(input, output);
        return output;
    }

    // Store input for backward pass
    Tensor<T> saved_input_;
    Tensor<T> grad_weights_;
    Tensor<T> grad_bias_;

    void forward(const Tensor<T>& input, Tensor<T>& output) override {
        // Save input for backward (Deep copy usually needed if input changes,
        // but here we might get away with shallow if input persists,
        // but for safety in training loop we deep copy)
        // However, standard backprop keeps reference.
        // Let's assume input lifetime is managed by caller or we copy.
        // For simple training loop, copy is safer.
        saved_input_ = input; // Deep copy

        auto shape = input.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];

        size_t H_out = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
        size_t W_out = (W + 2 * padding_ - kernel_size_) / stride_ + 1;

        T* out_ptr = output.data();
        const T* in_ptr = input.data();
        const T* w_ptr = packed_weights_.data();
        const T* b_ptr = bias_.data();

        // Optimized 1x1 Path
        if (kernel_size_ == 1 && stride_ == 1 && padding_ == 0 && groups_ == 1) {
            size_t total_pixels = N * H * W;
            const size_t BLOCK = 32;

            #pragma omp parallel for
            for(size_t idx = 0; idx < total_pixels; ++idx) {
                T* pixel_out = out_ptr + idx * out_channels_;
                const T* pixel_in = in_ptr + idx * in_channels_;
                const T* w_base = w_ptr;

                size_t ob = 0;

                // Main Block Loop (32 channels at a time)
                for(; ob + BLOCK <= out_channels_; ob += BLOCK) {
#ifdef __AVX2__
                    // Init with Bias
                    __m256 acc0 = _mm256_loadu_ps(b_ptr + ob + 0);
                    __m256 acc1 = _mm256_loadu_ps(b_ptr + ob + 8);
                    __m256 acc2 = _mm256_loadu_ps(b_ptr + ob + 16);
                    __m256 acc3 = _mm256_loadu_ps(b_ptr + ob + 24);

                    for(size_t i=0; i<in_channels_; ++i) {
                        T val = pixel_in[i];
                        __m256 v_val = _mm256_set1_ps(val);
                        const T* w_i = w_base + i * out_channels_ + ob;

                        acc0 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_i+0), acc0);
                        acc1 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_i+8), acc1);
                        acc2 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_i+16), acc2);
                        acc3 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_i+24), acc3);
                    }

                    _mm256_storeu_ps(pixel_out + ob + 0, acc0);
                    _mm256_storeu_ps(pixel_out + ob + 8, acc1);
                    _mm256_storeu_ps(pixel_out + ob + 16, acc2);
                    _mm256_storeu_ps(pixel_out + ob + 24, acc3);
#else
                    // Fallback Scalar for Block
                    for(size_t k=0; k<BLOCK; ++k) pixel_out[ob+k] = b_ptr[ob+k];
                    for(size_t i=0; i<in_channels_; ++i) {
                        T val = pixel_in[i];
                        const T* w_i = w_base + i * out_channels_ + ob;
                        for(size_t k=0; k<BLOCK; ++k) pixel_out[ob+k] += val * w_i[k];
                    }
#endif
                }

                // Tail Loop
                if (ob < out_channels_) {
                    for(size_t o = ob; o < out_channels_; ++o) pixel_out[o] = b_ptr[o];
                    for(size_t i=0; i<in_channels_; ++i) {
                        T val = pixel_in[i];
                        const T* w_i = w_base + i * out_channels_;
                        for(size_t o = ob; o < out_channels_; ++o) {
                            pixel_out[o] += val * w_i[o];
                        }
                    }
                }
            }
            return;
        }

        // Optimized Loop (Generic)
        // Vectorize over Out Channels

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {

                    long h_in_start = static_cast<long>(h_out * stride_) - static_cast<long>(padding_);
                    long w_in_start = static_cast<long>(w_out * stride_) - static_cast<long>(padding_);

                    T* pixel_out = out_ptr + ((n*H_out + h_out)*W_out + w_out) * out_channels_;

                    // Initialize accumulators with bias
                    for(size_t o=0; o<out_channels_; ++o) {
                        pixel_out[o] = b_ptr[o];
                    }

                    for(size_t ky=0; ky<kernel_size_; ++ky) {
                        for(size_t kx=0; kx<kernel_size_; ++kx) {
                            long h_in = h_in_start + ky;
                            long w_in = w_in_start + kx;

                            if (h_in >= 0 && h_in < (long)H && w_in >= 0 && w_in < (long)W) {
                                const T* pixel_in = in_ptr + ((n*H + h_in)*W + w_in) * in_channels_;

                                // Weight pointer for this kernel position
                                // Layout: [K*K, In, Out]
                                size_t k_idx = ky * kernel_size_ + kx;
                                const T* w_k = w_ptr + k_idx * (in_channels_ * out_channels_);

                                for(size_t i=0; i<in_channels_; ++i) {
                                    T val = pixel_in[i];
                                    const T* w_ki = w_k + i * out_channels_;

                                    // Vectorize over Output Channels
                                    size_t o = 0;
#ifdef __AVX2__
                                    __m256 v_val = _mm256_set1_ps(val);
                                    for(; o + 8 <= out_channels_; o += 8) {
                                        __m256 v_out = _mm256_loadu_ps(pixel_out + o);
                                        __m256 v_w = _mm256_loadu_ps(w_ki + o);
                                        v_out = _mm256_fmadd_ps(v_val, v_w, v_out);
                                        _mm256_storeu_ps(pixel_out + o, v_out);
                                    }
#endif
                                    // Scalar tail
                                    for(; o < out_channels_; ++o) {
                                        pixel_out[o] += val * w_ki[o];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Only 1x1 stride 1 supported for now (ZenithNano)
        if (kernel_size_ != 1 || stride_ != 1 || padding_ != 0) {
            std::cerr << "OptimizedConv2D::backward only supports 1x1 stride 1 for now." << std::endl;
            return Tensor<T>();
        }

        // 1. Compute grad_weights: input.T * grad_output
        // Input: [N, H, W, Cin], GradOut: [N, H, W, Cout]
        // Flatten: [Pixels, Cin], [Pixels, Cout]
        // GradW: [Cin, Cout] = Input.T * GradOut

        size_t N = saved_input_.shape()[0];
        size_t H = saved_input_.shape()[1];
        size_t W = saved_input_.shape()[2];
        size_t Cin = in_channels_;
        size_t Cout = out_channels_;
        size_t Pixels = N * H * W;

        // Initialize grads
        if (grad_weights_.size() == 0) grad_weights_ = Tensor<T>(packed_weights_.shape());
        if (grad_bias_.size() == 0) grad_bias_ = Tensor<T>(bias_.shape());
        grad_weights_.fill(0);
        grad_bias_.fill(0);

        const T* in_ptr = saved_input_.data();
        const T* go_ptr = grad_output.data();
        T* gw_ptr = grad_weights_.data(); // [1, 1, Cin, Cout] packed as [1, 1, Cin, Cout] for 1x1
        T* gb_ptr = grad_bias_.data();

        // Compute Gradients
        // Parallel over pixels, reduce to GW/GB
        // Since atomic add floats is tricky/slow, we can tile or reduce.
        // For simplicity, simple reduction or serial accumulation.
        // Better: Tiled GEMM.
        // Loop over Pixels: GW += in * go

        // Let's do simple parallel over Cin, Cout? No, pixels dominate.
        // To allow OMP, we need private reduction buffers or atomic.
        // Given complexity, let's do a naive OMP parallel over Cin (outer loop of weights).

        #pragma omp parallel for
        for(size_t i=0; i<Cin; ++i) {
             for(size_t o=0; o<Cout; ++o) {
                 float sum = 0.0f;
                 // Vectorize pixel loop
                 size_t p=0;
                 #ifdef __AVX2__
                 __m256 v_sum = _mm256_setzero_ps();
                 for(; p+8<=Pixels; p+=8) {
                     __m256 v_in = _mm256_loadu_ps(in_ptr + p*Cin + i * 1); // Stride Cin? No.
                     // Access: in_ptr[p*Cin + i]. Stride is Cin. Not contiguous.
                     // Gather needed.
                     __m256i v_idx_in = _mm256_set_epi32((p+7)*Cin+i, (p+6)*Cin+i, (p+5)*Cin+i, (p+4)*Cin+i,
                                                        (p+3)*Cin+i, (p+2)*Cin+i, (p+1)*Cin+i, (p+0)*Cin+i);
                     __m256 v_val_in = _mm256_i32gather_ps(in_ptr, v_idx_in, 4);

                     __m256i v_idx_go = _mm256_set_epi32((p+7)*Cout+o, (p+6)*Cout+o, (p+5)*Cout+o, (p+4)*Cout+o,
                                                        (p+3)*Cout+o, (p+2)*Cout+o, (p+1)*Cout+o, (p+0)*Cout+o);
                     __m256 v_val_go = _mm256_i32gather_ps(go_ptr, v_idx_go, 4);

                     v_sum = _mm256_fmadd_ps(v_val_in, v_val_go, v_sum);
                 }
                 // Horizontal sum v_sum
                 float tmp[8]; _mm256_storeu_ps(tmp, v_sum);
                 sum += tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                 #endif

                 for(; p<Pixels; ++p) {
                     sum += in_ptr[p*Cin + i] * go_ptr[p*Cout + o];
                 }
                 // Store weight grad
                 // packed_weights layout for 1x1 is [1, 1, Cin, Cout] -> [Cin, Cout]
                 gw_ptr[i*Cout + o] = sum;
             }
        }

        // Bias Grad: Sum over N,H,W
        #pragma omp parallel for
        for(size_t o=0; o<Cout; ++o) {
            float sum = 0.0f;
            for(size_t p=0; p<Pixels; ++p) {
                sum += go_ptr[p*Cout + o];
            }
            gb_ptr[o] = sum;
        }

        // 2. Compute grad_input: grad_output * weights
        // [Pixels, Cout] * [Cout, Cin] -> [Pixels, Cin]
        // This is like a forward pass with transposed weights.
        Tensor<T> grad_input(saved_input_.shape());
        T* gi_ptr = grad_input.data();
        const T* w_ptr = packed_weights_.data(); // [Cin, Cout]

        #pragma omp parallel for
        for(size_t p=0; p<Pixels; ++p) {
            for(size_t i=0; i<Cin; ++i) {
                float sum = 0.0f;
                size_t o=0;
                #ifdef __AVX2__
                __m256 v_sum = _mm256_setzero_ps();
                for(; o+8<=Cout; o+=8) {
                    __m256 v_go = _mm256_loadu_ps(go_ptr + p*Cout + o);
                    __m256 v_w = _mm256_loadu_ps(w_ptr + i*Cout + o); // packed is [Cin, Cout]
                    v_sum = _mm256_fmadd_ps(v_go, v_w, v_sum);
                }
                float tmp[8]; _mm256_storeu_ps(tmp, v_sum);
                sum += tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
                #endif
                for(; o<Cout; ++o) {
                    sum += go_ptr[p*Cout + o] * w_ptr[i*Cout + o];
                }
                gi_ptr[p*Cin + i] = sum;
            }
        }

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {&packed_weights_, &bias_}; }
    std::vector<Tensor<T>*> gradients() override { return {&grad_weights_, &grad_bias_}; }

    std::string name() const override { return "OptimizedConv2D"; }

    // Accessors for Fused Kernels
    const Tensor<T>& get_packed_weights() const { return packed_weights_; }
    const Tensor<T>& get_bias() const { return bias_; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t groups_;

    Tensor<T> packed_weights_; // [K, K, In, Out]
    Tensor<T> bias_;
};

} // namespace layers
} // namespace dreidel
