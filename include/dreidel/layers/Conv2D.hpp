#pragma once

#include "Layer.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace dreidel {
namespace layers {

template <typename T>
class Conv2D : public Layer<T> {
public:
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0, size_t groups = 1)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding), groups_(groups),
          weights_({out_channels, in_channels / groups, kernel_size, kernel_size}),
          bias_({1, out_channels}),
          grad_weights_({out_channels, in_channels / groups, kernel_size, kernel_size}),
          grad_bias_({1, out_channels})
    {
        if (in_channels % groups != 0 || out_channels % groups != 0) {
            throw std::invalid_argument("Conv2D: Channels must be divisible by groups");
        }

        // He Initialization (Kaiming)
        // Fan-in: in_channels_per_group * K * K
        T stddev = std::sqrt(2.0 / ((in_channels / groups) * kernel_size * kernel_size));
        weights_.random(0, stddev);
        bias_.fill(0);
        grad_weights_.fill(0);
        grad_bias_.fill(0);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Assume NHWC input
        input_ = input;
        auto shape = input.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C_in = shape[3];

        if (C_in != in_channels_) throw std::runtime_error("Conv2D channel mismatch");

        size_t H_out = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
        size_t W_out = (W + 2 * padding_ - kernel_size_) / stride_ + 1;
        size_t C_out = out_channels_;

        Tensor<T> output({N, H_out, W_out, C_out});
        output.fill(0);

        const T* in_ptr = input.data();
        const T* w_ptr = weights_.data();
        const T* b_ptr = bias_.data();
        T* out_ptr = output.data();

        size_t C_in_group = C_in / groups_;
        size_t C_out_group = C_out / groups_;

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {

                    long h_in_start = static_cast<long>(h_out * stride_) - static_cast<long>(padding_);
                    long w_in_start = static_cast<long>(w_out * stride_) - static_cast<long>(padding_);

                    for(size_t g=0; g<groups_; ++g) {
                        for(size_t c_out_sub=0; c_out_sub<C_out_group; ++c_out_sub) {
                            size_t c_out = g * C_out_group + c_out_sub;
                            T acc = b_ptr[c_out];

                            for(size_t ky=0; ky<kernel_size_; ++ky) {
                                for(size_t kx=0; kx<kernel_size_; ++kx) {
                                    long h_in = h_in_start + ky;
                                    long w_in = w_in_start + kx;

                                    if (h_in >= 0 && h_in < static_cast<long>(H) && w_in >= 0 && w_in < static_cast<long>(W)) {
                                        for(size_t c_in_sub=0; c_in_sub<C_in_group; ++c_in_sub) {
                                            size_t c_in = g * C_in_group + c_in_sub;

                                            // Weights shape: (Out, In/G, K, K)
                                            // w index: c_out * (C_in_group*K*K) + c_in_sub * (K*K) + ...
                                            T val = in_ptr[((n*H + h_in)*W + w_in)*C_in + c_in];

                                            // w_ptr is flattened.
                                            // Indexing: [c_out][c_in_sub][ky][kx]
                                            size_t w_idx = ((c_out * C_in_group + c_in_sub) * kernel_size_ + ky) * kernel_size_ + kx;
                                            T w = w_ptr[w_idx];
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
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Assume correct shapes from forward
        auto g_shape = grad_output.shape();
        size_t N = g_shape[0];
        size_t H_out = g_shape[1];
        size_t W_out = g_shape[2];

        auto i_shape = input_.shape();
        size_t H = i_shape[1];
        size_t W = i_shape[2];
        size_t C_in = in_channels_;

        Tensor<T> grad_input(i_shape);
        grad_input.fill(0);

        grad_weights_.fill(0);
        grad_bias_.fill(0);

        const T* go_ptr = grad_output.data();
        const T* in_ptr = input_.data();
        const T* w_ptr = weights_.data();

        T* gi_ptr = grad_input.data();
        T* gw_ptr = grad_weights_.data();
        T* gb_ptr = grad_bias_.data();

        size_t C_in_group = C_in / groups_;
        size_t C_out_group = out_channels_ / groups_;

        // Bias Grads
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H_out; ++h) {
                for(size_t w=0; w<W_out; ++w) {
                    for(size_t c=0; c<out_channels_; ++c) {
                        gb_ptr[c] += go_ptr[((n*H_out+h)*W_out+w)*out_channels_ + c];
                    }
                }
            }
        }

        // Weight/Input Grads (Serial for safety/baseline)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {

                    long h_in_start = static_cast<long>(h_out * stride_) - static_cast<long>(padding_);
                    long w_in_start = static_cast<long>(w_out * stride_) - static_cast<long>(padding_);

                    for(size_t g=0; g<groups_; ++g) {
                        for(size_t c_out_sub=0; c_out_sub<C_out_group; ++c_out_sub) {
                            size_t c_out = g * C_out_group + c_out_sub;
                            T dy = go_ptr[((n*H_out+h_out)*W_out+w_out)*out_channels_ + c_out];

                            for(size_t ky=0; ky<kernel_size_; ++ky) {
                                for(size_t kx=0; kx<kernel_size_; ++kx) {
                                    long h_in = h_in_start + ky;
                                    long w_in = w_in_start + kx;

                                    if (h_in >= 0 && h_in < static_cast<long>(H) && w_in >= 0 && w_in < static_cast<long>(W)) {
                                        for(size_t c_in_sub=0; c_in_sub<C_in_group; ++c_in_sub) {
                                            size_t c_in = g * C_in_group + c_in_sub;

                                            size_t w_idx = ((c_out * C_in_group + c_in_sub) * kernel_size_ + ky) * kernel_size_ + kx;
                                            size_t in_idx = ((n*H + h_in)*W + w_in)*C_in + c_in;

                                            gw_ptr[w_idx] += dy * in_ptr[in_idx];
                                            gi_ptr[in_idx] += dy * w_ptr[w_idx];
                                        }
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
    size_t groups_;

    Tensor<T> weights_;
    Tensor<T> bias_;

    Tensor<T> grad_weights_;
    Tensor<T> grad_bias_;

    Tensor<T> input_;
};

} // namespace layers
} // namespace dreidel
