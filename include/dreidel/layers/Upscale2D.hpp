#pragma once

#include "Layer.hpp"
#include <vector>
#include <omp.h>

namespace dreidel {
namespace layers {

template <typename T>
class Upscale2D : public Layer<T> {
public:
    Upscale2D(size_t scale) : scale_(scale) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        input_ = input;
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
        size_t H_out = H * scale_; size_t W_out = W * scale_;
        Tensor<T> output({N, H_out, W_out, C});
        T* out_ptr = output.data();
        const T* in_ptr = input.data();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {
                    size_t h_in = h_out / scale_;
                    size_t w_in = w_out / scale_;
                    for(size_t c=0; c<C; ++c) {
                        out_ptr[((n*H_out + h_out)*W_out + w_out)*C + c] = in_ptr[((n*H + h_in)*W + w_in)*C + c];
                    }
                }
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Nearest Neighbor Backward: Sum gradients
        auto shape = input_.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
        Tensor<T> grad_input(shape);
        grad_input.fill(0);

        auto g_shape = grad_output.shape();
        size_t H_out = g_shape[1]; size_t W_out = g_shape[2];

        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {
                    size_t h_in = h_out / scale_;
                    size_t w_in = w_out / scale_;
                    for(size_t c=0; c<C; ++c) {
                        T val = go_ptr[((n*H_out + h_out)*W_out + w_out)*C + c];
                        #pragma omp atomic
                        gi_ptr[((n*H + h_in)*W + w_in)*C + c] += val;
                    }
                }
            }
        }
        return grad_input;
    }

    std::string name() const override { return "Upscale2D"; }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }

private:
    size_t scale_;
    Tensor<T> input_;
};

} // namespace layers
} // namespace dreidel
