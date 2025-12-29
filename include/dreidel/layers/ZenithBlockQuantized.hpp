#pragma once

#include "Layer.hpp"
#include "../core/Memory.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <cmath>
#include <iostream>

namespace dreidel {
namespace layers {

/**
 * @brief Zenith Block Optimized for Quantized I/O (APoT).
 *
 * Takes Tensor<int8_t> as input and produces Tensor<int8_t> as output.
 * Internal computation is done in Float (accumulators).
 * Uses AVX intrinsics for Unpack -> Compute -> Pack pipeline.
 */
class ZenithBlockQuantized {
public:
    ZenithBlockQuantized(size_t channels, size_t kernel_size, size_t arena_size = 1024*1024)
        : channels_(channels), kernel_size_(kernel_size),
          arena_(arena_size),
          packed_weights_(channels * kernel_size * kernel_size)
    {
        // Random Init Weights (Simulated)
        for(auto& w : packed_weights_) w = 10; // Dummy APoT code
    }

    // Custom Forward for Quantized Tensor
    // We bypass Layer<T> inheritance to enforce int8 interface for this specific optimization.
    Tensor<int8_t> forward(const Tensor<int8_t>& input) {
        auto shape = input.shape();
        size_t batch = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3];

        Tensor<int8_t> output(shape);

        const int8_t* in_ptr = input.data();
        int8_t* out_ptr = output.data();
        const int8_t* w_ptr = packed_weights_.data();

        int k_rad = kernel_size_ / 2;

        arena_.reset();
        int8_t* pixel_buffer = arena_.allocate<int8_t>(C); // Integer accumulator (APoT domain)

        // Tiled Loop (8x8)
        constexpr int BLOCK_H = 8;
        constexpr int BLOCK_W = 8;

        for(size_t n=0; n<batch; ++n) {
            for(size_t by=0; by<H; by+=BLOCK_H) {
                for(size_t bx=0; bx<W; bx+=BLOCK_W) {

                    for(size_t dy=0; dy<BLOCK_H; ++dy) {
                        for(size_t dx=0; dx<BLOCK_W; ++dx) {
                            size_t h = by + dy;
                            size_t w = bx + dx;
                            if (h >= H || w >= W) continue;

                            // Reset Accumulator (0 in APoT is code 0)
                            for(size_t c=0; c<C; ++c) pixel_buffer[c] = 0;

                            // Convolution
                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int ih = h + ky;
                                    int iw = w + kx;
                                    if (ih>=0 && ih<H && iw>=0 && iw<W) {
                                        const int8_t* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                        int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                        const int8_t* p_w = w_ptr + k_idx * channels_;

                                        size_t c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                                        // Vectorized 32-wide integer pipeline using Parallel Shuffle LUT
                                        for(; c+32 <= C; c+=32) {
                                            // 1. Load 32 values (256-bit)
                                            __m256i v_in = _mm256_loadu_si256((const __m256i*)(p_in + c));
                                            __m256i v_w  = _mm256_loadu_si256((const __m256i*)(p_w + c));

                                            // 2. MUL: Process 128-bit lanes (16 bytes)
                                            __m128i in_lo = _mm256_castsi256_si128(v_in);
                                            __m128i in_hi = _mm256_extracti128_si256(v_in, 1);
                                            __m128i w_lo  = _mm256_castsi256_si128(v_w);
                                            __m128i w_hi  = _mm256_extracti128_si256(v_w, 1);

                                            __m128i prod_lo = hal::AlienOps::vec_mul_apot_avx2(in_lo, w_lo);
                                            __m128i prod_hi = hal::AlienOps::vec_mul_apot_avx2(in_hi, w_hi);

                                            __m256i v_prod = _mm256_set_m128i(prod_hi, prod_lo);

                                            // 3. ADD: Process 256-bit with Shuffle LUT
                                            __m256i v_acc = _mm256_loadu_si256((const __m256i*)(pixel_buffer + c));
                                            v_acc = hal::AlienOps::vec_add_apot_avx2(v_acc, v_prod);
                                            _mm256_storeu_si256((__m256i*)(pixel_buffer + c), v_acc);
                                        }
#endif
                                        // Scalar Fallback
                                        for(; c<C; ++c) {
                                            int8_t prod = hal::AlienOps::apot_mul_lut(p_in[c], p_w[c]);
                                            pixel_buffer[c] = hal::AlienOps::apot_add_lut(pixel_buffer[c], prod);
                                        }
                                    }
                                }
                            }

                            // Output (Accumulator is already int8 APoT code)
                            int8_t* p_out = out_ptr + ((n*H + h)*W + w)*C;
                            for(size_t c=0; c<C; ++c) {
                                // Branchless ReLU in APoT domain
                                // If val is code for negative number (sign bit set), clamp to 0.
                                // 0 has code 0 (0x00). Negative numbers have 0x80 bit set.
                                // Positive numbers have 0x00 bit clear.
                                // So (val & 0x80) means negative.
                                int8_t val = pixel_buffer[c];
                                if (val & 0x80) val = 0;
                                p_out[c] = val;
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

private:
    size_t channels_;
    size_t kernel_size_;
    core::Arena arena_;
    std::vector<int8_t> packed_weights_;
};

} // namespace layers
} // namespace dreidel
