#ifndef DREIDEL_LAYERS_QUANTIZED_STANDARD_HPP
#define DREIDEL_LAYERS_QUANTIZED_STANDARD_HPP

#include "Layer.hpp"
#include "../core/Tensor.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace dreidel {
namespace layers {

// Generic Helpers for Flatten/Reshape on Int8
// We can use standard Layer<int8_t> for these if we had them,
// but for now we implement them here for the benchmark chain.

class FlattenInt8 : public Layer<int8_t> {
public:
    Tensor<int8_t> forward(const Tensor<int8_t>& input) override {
        auto shape = input.shape();
        size_t batch = shape[0];
        size_t dim = 1;
        for(size_t i=1; i<shape.size(); ++i) dim *= shape[i];

        Tensor<int8_t> output({batch, dim});
        const int8_t* in_ptr = input.data();
        int8_t* out_ptr = output.data();
        std::copy(in_ptr, in_ptr + output.size(), out_ptr);
        return output;
    }
    Tensor<int8_t> backward(const Tensor<int8_t>&) override { return Tensor<int8_t>(); }
    std::string name() const override { return "FlattenInt8"; }
};

class ReshapeInt8 : public Layer<int8_t> {
public:
    ReshapeInt8(std::vector<size_t> suffix) : suffix_(suffix) {}
    Tensor<int8_t> forward(const Tensor<int8_t>& input) override {
        auto shape = input.shape();
        size_t batch = shape[0];
        std::vector<size_t> new_shape = {batch};
        new_shape.insert(new_shape.end(), suffix_.begin(), suffix_.end());

        Tensor<int8_t> output(new_shape);
        const int8_t* in_ptr = input.data();
        int8_t* out_ptr = output.data();
        std::copy(in_ptr, in_ptr + output.size(), out_ptr);
        return output;
    }
    Tensor<int8_t> backward(const Tensor<int8_t>&) override { return Tensor<int8_t>(); }
    std::string name() const override { return "ReshapeInt8"; }
private:
    std::vector<size_t> suffix_;
};

/**
 * @brief Quantized Average Pooling 2D.
 * Input: Tensor<int8_t> (APoT codes)
 * Output: Tensor<int8_t> (APoT codes)
 * Logic: Unpack Window -> Average -> Pack
 */
class QuantizedAvgPool2D : public Layer<int8_t> {
public:
    QuantizedAvgPool2D(size_t stride) : stride_(stride) {}

    Tensor<int8_t> forward(const Tensor<int8_t>& input) override {
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
        size_t H_out = H / stride_;
        size_t W_out = W / stride_;

        Tensor<int8_t> output({N, H_out, W_out, C});
        const int8_t* in_ptr = input.data();
        int8_t* out_ptr = output.data();

        // Parallelize over batch/height
        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H_out; ++h) {
                for(size_t w=0; w<W_out; ++w) {
                    for(size_t c=0; c<C; ++c) {
                        float sum = 0.0f;
                        for(size_t dy=0; dy<stride_; ++dy) {
                            for(size_t dx=0; dx<stride_; ++dx) {
                                int8_t code = in_ptr[((n*H + h*stride_+dy)*W + w*stride_+dx)*C + c];
                                sum += hal::AlienOps::unpack_apot(code);
                            }
                        }
                        float avg = sum / (stride_ * stride_);
                        out_ptr[((n*H_out + h)*W_out + w)*C + c] = hal::AlienOps::pack_apot(avg);
                    }
                }
            }
        }
        return output;
    }
    Tensor<int8_t> backward(const Tensor<int8_t>&) override { return Tensor<int8_t>(); }
    std::string name() const override { return "QuantizedAvgPool2D"; }
private:
    size_t stride_;
};

/**
 * @brief Quantized Upscale 2D (Nearest Neighbor).
 * Input: Tensor<int8_t>
 * Output: Tensor<int8_t>
 * Logic: Direct code copy (Nearest Neighbor preserves value, so preserves code).
 */
class QuantizedUpscale2D : public Layer<int8_t> {
public:
    QuantizedUpscale2D(size_t scale) : scale_(scale) {}

    Tensor<int8_t> forward(const Tensor<int8_t>& input) override {
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
        size_t H_out = H * scale_;
        size_t W_out = W * scale_;

        Tensor<int8_t> output({N, H_out, W_out, C});
        const int8_t* in_ptr = input.data();
        int8_t* out_ptr = output.data();

        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {
                    size_t h_in = h_out / scale_;
                    size_t w_in = w_out / scale_;

                    const int8_t* p_src = in_ptr + ((n*H + h_in)*W + w_in)*C;
                    int8_t* p_dst = out_ptr + ((n*H_out + h_out)*W_out + w_out)*C;

                    // Vectorized copy if C is large
                    // Since C is usually multiple of 8/16/32 in benchmark
                    size_t c = 0;
                    // Use memcpy or loop
                    std::copy(p_src, p_src + C, p_dst);
                }
            }
        }
        return output;
    }
    Tensor<int8_t> backward(const Tensor<int8_t>&) override { return Tensor<int8_t>(); }
    std::string name() const override { return "QuantizedUpscale2D"; }
private:
    size_t scale_;
};

/**
 * @brief Quantized Dense Layer.
 * Input: Tensor<int8_t>
 * Output: Tensor<int8_t>
 * Logic: Unpack Input -> Float MatMul (Weights are Float) -> Pack Output.
 * Optimizes bottleneck by keeping dense weights precise while fitting in quantized chain.
 */
class QuantizedDense : public Layer<int8_t> {
public:
    QuantizedDense(size_t input_dim, size_t output_dim)
        : input_dim_(input_dim), output_dim_(output_dim),
          weights_({input_dim, output_dim}), bias_({1, output_dim})
    {
        // Init weights (Float)
        float stddev = std::sqrt(2.0f / (input_dim + output_dim));
        weights_.random(0, stddev);
        bias_.fill(0);
    }

    Tensor<int8_t> forward(const Tensor<int8_t>& input) override {
        // Input: (B, In)
        auto shape = input.shape();
        size_t batch = shape[0];
        // Assert shape[1] == input_dim_ (if flattened)

        // 1. Unpack Input to Float
        Tensor<float> input_f({batch, input_dim_});
        const int8_t* in_ptr = input.data();
        float* f_ptr = input_f.data();

        // Parallel unpack
        #pragma omp parallel for
        for(size_t i=0; i<batch * input_dim_; ++i) {
            f_ptr[i] = hal::AlienOps::unpack_apot(in_ptr[i]);
        }

        // 2. MatMul (Float)
        // Output = InputF * Weights + Bias
        Tensor<float> out_f = input_f.matmul(weights_);
        out_f = out_f + bias_;

        // 3. Pack Output to Int8
        Tensor<int8_t> output({batch, output_dim_});
        const float* of_ptr = out_f.data();
        int8_t* out_ptr = output.data();

        #pragma omp parallel for
        for(size_t i=0; i<batch * output_dim_; ++i) {
            out_ptr[i] = hal::AlienOps::pack_apot(of_ptr[i]);
        }

        return output;
    }

    Tensor<int8_t> backward(const Tensor<int8_t>&) override { return Tensor<int8_t>(); }
    std::string name() const override { return "QuantizedDense"; }

private:
    size_t input_dim_, output_dim_;
    Tensor<float> weights_;
    Tensor<float> bias_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_QUANTIZED_STANDARD_HPP
