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
 * Logic: Accumulate using LNS add, divide by N using LNS mul (add exponent of 1/N).
 */
class QuantizedAvgPool2D : public Layer<int8_t> {
public:
    QuantizedAvgPool2D(size_t stride) : stride_(stride) {
        // Precompute code for 1.0 / (stride * stride)
        float inv_n = 1.0f / (stride * stride);
        inv_scale_code_ = hal::AlienOps::pack_apot(inv_n);
    }

    Tensor<int8_t> forward(const Tensor<int8_t>& input) override {
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
        size_t H_out = H / stride_;
        size_t W_out = W / stride_;

        Tensor<int8_t> output({N, H_out, W_out, C});
        const int8_t* in_ptr = input.data();
        int8_t* out_ptr = output.data();

        // Broadcast inverse scale for AVX
#if defined(DREIDEL_ARCH_AVX2)
        __m256i v_inv_scale = _mm256_set1_epi8(inv_scale_code_);
#endif

        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<H_out; ++h) {
                for(size_t w=0; w<W_out; ++w) {

                    size_t c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                    for(; c+32 <= C; c+=32) {
                        __m256i v_sum = _mm256_setzero_si256(); // 0 is code 0

                        for(size_t dy=0; dy<stride_; ++dy) {
                            for(size_t dx=0; dx<stride_; ++dx) {
                                const int8_t* p_src = in_ptr + ((n*H + h*stride_+dy)*W + w*stride_+dx)*C + c;
                                __m256i v_val = _mm256_loadu_si256((const __m256i*)p_src);
                                v_sum = hal::AlienOps::vec_add_apot_avx2(v_sum, v_val);
                            }
                        }

                        // Divide by N (Multiply by 1/N)
                        // Note: vec_mul_apot_avx2 takes 128-bit args. Split 256.
                        __m128i sum_lo = _mm256_castsi256_si128(v_sum);
                        __m128i sum_hi = _mm256_extracti128_si256(v_sum, 1);
                        __m128i scale_lo = _mm256_castsi256_si128(v_inv_scale);
                        __m128i scale_hi = _mm256_extracti128_si256(v_inv_scale, 1);

                        __m128i res_lo = hal::AlienOps::vec_mul_apot_avx2(sum_lo, scale_lo);
                        __m128i res_hi = hal::AlienOps::vec_mul_apot_avx2(sum_hi, scale_hi);

                        __m256i v_res = _mm256_set_m128i(res_hi, res_lo);
                        _mm256_storeu_si256((__m256i*)(out_ptr + ((n*H_out + h)*W_out + w)*C + c), v_res);
                    }
#endif
                    for(; c<C; ++c) {
                        int8_t sum = 0;
                        for(size_t dy=0; dy<stride_; ++dy) {
                            for(size_t dx=0; dx<stride_; ++dx) {
                                int8_t val = in_ptr[((n*H + h*stride_+dy)*W + w*stride_+dx)*C + c];
                                sum = hal::AlienOps::apot_add_lut(sum, val);
                            }
                        }
                        int8_t avg = hal::AlienOps::apot_mul_lut(sum, inv_scale_code_);
                        out_ptr[((n*H_out + h)*W_out + w)*C + c] = avg;
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
    int8_t inv_scale_code_;
};

/**
 * @brief Quantized Upscale 2D (Nearest Neighbor).
 * Input: Tensor<int8_t>
 * Output: Tensor<int8_t>
 * Logic: Direct code copy (Vectorized).
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

                    size_t c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                    for(; c+32 <= C; c+=32) {
                        __m256i v = _mm256_loadu_si256((const __m256i*)(p_src + c));
                        _mm256_storeu_si256((__m256i*)(p_dst + c), v);
                    }
#endif
                    std::copy(p_src + c, p_src + C, p_dst + c);
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


} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_QUANTIZED_STANDARD_HPP
