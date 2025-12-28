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

                                        // Full LNS Pipeline via LUT (Never Unpack)
                                        for(size_t c=0; c<C; ++c) {
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
