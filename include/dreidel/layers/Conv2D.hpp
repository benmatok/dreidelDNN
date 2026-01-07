#pragma once

#include "Layer.hpp"
#include <vector>
#include <cmath>
#include <random>

namespace dreidel {
namespace layers {

template <typename T>
class Conv2D : public Layer<T> {
public:
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding),
          weights_({out_channels, in_channels, kernel_size, kernel_size}),
          bias_({1, out_channels}),
          grad_weights_({out_channels, in_channels, kernel_size, kernel_size}),
          grad_bias_({1, out_channels})
    {
        // He Initialization (Kaiming)
        T stddev = std::sqrt(2.0 / (in_channels * kernel_size * kernel_size));
        weights_.random(0, stddev);
        bias_.fill(0);
        grad_weights_.fill(0);
        grad_bias_.fill(0);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Input: (N, H, W, C_in) or (N, C, H, W)?
        // ZenithBlock uses (N, H, W, C). Let's stick to that for consistency,
        // although standard C++ implementations often prefer NCHW for SIMD.
        // dreidel::Tensor doesn't strictly enforce, but earlier code used NHWC loops.
        // Let's assume NHWC.

        input_ = input;
        auto shape = input.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C_in = shape[3]; // Should match in_channels_

        size_t H_out = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
        size_t W_out = (W + 2 * padding_ - kernel_size_) / stride_ + 1;
        size_t C_out = out_channels_;

        Tensor<T> output({N, H_out, W_out, C_out});
        output.fill(0);

        const T* in_ptr = input.data();
        const T* w_ptr = weights_.data(); // (C_out, C_in, K, K)
        const T* b_ptr = bias_.data();
        T* out_ptr = output.data();

        int k_rad = kernel_size_ / 2; // Only valid for odd kernels if we think in radius, but here we use 0..K-1

        // Naive 6-loop implementation (Slow baseline)
        // Optimized slightly by hoisting

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {

                    // Use long to prevent underflow during subtraction of padding
                    long h_in_start = static_cast<long>(h_out * stride_) - static_cast<long>(padding_);
                    long w_in_start = static_cast<long>(w_out * stride_) - static_cast<long>(padding_);

                    for(size_t c_out=0; c_out<C_out; ++c_out) {
                        T acc = b_ptr[c_out];

                        for(size_t ky=0; ky<kernel_size_; ++ky) {
                            for(size_t kx=0; kx<kernel_size_; ++kx) {
                                long h_in = h_in_start + ky;
                                long w_in = w_in_start + kx;

                                if (h_in >= 0 && h_in < static_cast<long>(H) && w_in >= 0 && w_in < static_cast<long>(W)) {
                                    for(size_t c_in=0; c_in<C_in; ++c_in) {
                                        // w index: c_out * (C_in*K*K) + c_in * (K*K) + ky*K + kx
                                        // in index: ((n*H + h_in)*W + w_in)*C_in + c_in
                                        T val = in_ptr[((n*H + h_in)*W + w_in)*C_in + c_in];
                                        T w = w_ptr[((c_out * in_channels_ + c_in) * kernel_size_ + ky) * kernel_size_ + kx];
                                        acc += val * w;
                                    }
                                }
                            }
                        }
                        out_ptr[((n*H_out + h_out)*W_out + w_out)*C_out + c_out] = acc;
                    }
                }
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // grad_output: (N, H_out, W_out, C_out)
        auto g_shape = grad_output.shape();
        size_t N = g_shape[0];
        size_t H_out = g_shape[1];
        size_t W_out = g_shape[2];

        auto i_shape = input_.shape();
        size_t H = i_shape[1];
        size_t W = i_shape[2];
        // Use actual input channels to prevent OOB if mismatch occurred
        size_t C_in = i_shape.size() > 3 ? i_shape[3] : in_channels_;

        Tensor<T> grad_input(i_shape);
        grad_input.fill(0);

        grad_weights_.fill(0);
        grad_bias_.fill(0); // Actually accumulates

        const T* go_ptr = grad_output.data();
        const T* in_ptr = input_.data();
        const T* w_ptr = weights_.data();

        T* gi_ptr = grad_input.data();
        T* gw_ptr = grad_weights_.data();
        T* gb_ptr = grad_bias_.data();

        // Accumulate Bias gradients
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H_out; ++h) {
                for(size_t w=0; w<W_out; ++w) {
                    for(size_t c=0; c<out_channels_; ++c) {
                        gb_ptr[c] += go_ptr[((n*H_out+h)*W_out+w)*out_channels_ + c];
                    }
                }
            }
        }

        // Gradients for Weights and Input
        // This is very slow naively.
        // We need to implement it correctly for validation.

        // Since we are writing to shared grad_weights and grad_input, parallelization needs care (atomics or reduction).
        // For simplicity in baseline, we might serialise or use atomics.

        // Let's iterate over output pixels and backpropagate
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {

                    long h_in_start = static_cast<long>(h_out * stride_) - static_cast<long>(padding_);
                    long w_in_start = static_cast<long>(w_out * stride_) - static_cast<long>(padding_);

                    for(size_t c_out=0; c_out<out_channels_; ++c_out) {
                        T dy = go_ptr[((n*H_out+h_out)*W_out+w_out)*out_channels_ + c_out];

                        for(size_t ky=0; ky<kernel_size_; ++ky) {
                            for(size_t kx=0; kx<kernel_size_; ++kx) {
                                long h_in = h_in_start + ky;
                                long w_in = w_in_start + kx;

                                if (h_in >= 0 && h_in < static_cast<long>(H) && w_in >= 0 && w_in < static_cast<long>(W)) {
                                    for(size_t c_in=0; c_in<C_in; ++c_in) {
                                        size_t w_idx = ((c_out * in_channels_ + c_in) * kernel_size_ + ky) * kernel_size_ + kx;
                                        size_t in_idx = ((n*H + h_in)*W + w_in)*C_in + c_in;

                                        // dL/dW += dy * x
                                        gw_ptr[w_idx] += dy * in_ptr[in_idx];

                                        // dL/dx += dy * w
                                        gi_ptr[in_idx] += dy * w_ptr[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&weights_, &bias_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_weights_, &grad_bias_};
    }

    std::string name() const override { return "Conv2D"; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;

    Tensor<T> weights_; // (Out, In, K, K)
    Tensor<T> bias_;

    Tensor<T> grad_weights_;
    Tensor<T> grad_bias_;

    Tensor<T> input_;
};

} // namespace layers
} // namespace dreidel
