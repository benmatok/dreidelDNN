#pragma once

#include "Layer.hpp"
#include "../core/Tensor.hpp"
#include "../kernels/AvxFwht.hpp"
#include "../hal/ops_f16.hpp"
#include "OptimizedConv2D_F16.hpp"
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>

namespace dreidel {
namespace layers {

class ZenithNanoBlock_F16 {
public:
    // F16 Variant of ZenithNanoBlock
    ZenithNanoBlock_F16(int channels, int res_h, int res_w)
        : channels_(channels), h_(res_h), w_(res_w),
          // Gates are still trained as Float usually, but we can store them as F16 or F32.
          // Let's store them as F16 to be consistent and save space.
          gate_h_({(size_t)res_w}), // [W]
          gate_w_({(size_t)res_h}), // [H]
          proj_conv_(channels, channels, 1, 1, 0),
          scratch_buffer_({1, (size_t)res_h, (size_t)res_w, (size_t)channels}),
          scratch_x_({1, (size_t)res_h, (size_t)res_w, (size_t)channels})
    {
        // Initialize gates to 1.0 (in F16)
        uint16_t one_f16 = 0x3c00; // 1.0 in float16
        std::fill(gate_h_.data(), gate_h_.data() + gate_h_.size(), one_f16);
        std::fill(gate_w_.data(), gate_w_.data() + gate_w_.size(), one_f16);
    }

    void set_gates(const Tensor<float>& gh, const Tensor<float>& gw) {
        // Convert to F16
        const float* src = gh.data();
        uint16_t* dst = gate_h_.data();
        for(size_t i=0; i<gh.size(); ++i) {
             __m128i h = _mm_cvtps_ph(_mm_set_ss(src[i]), 0);
             dst[i] = (uint16_t)_mm_cvtsi128_si32(h);
        }

        src = gw.data();
        dst = gate_w_.data();
        for(size_t i=0; i<gw.size(); ++i) {
             __m128i h = _mm_cvtps_ph(_mm_set_ss(src[i]), 0);
             dst[i] = (uint16_t)_mm_cvtsi128_si32(h);
        }
    }

    OptimizedConv2D_F16& get_conv() { return proj_conv_; }

    // Main Forward: F16 -> F16
    void forward(const Tensor<uint16_t>& input, Tensor<uint16_t>& output) {
        // Zero-Copy Path where possible.
        // 1. Copy Input to Scratch X
        const uint16_t* in_ptr = input.data();
        uint16_t* x_ptr = scratch_x_.data();
        size_t size = scratch_x_.size(); // Total elements
        std::memcpy(x_ptr, in_ptr, size * sizeof(uint16_t));

        uint16_t* buf_ptr = scratch_buffer_.data();

        // 1. Horizontal Spectral Mix (Row-wise)
        // FWHT -> Gate -> IFWHT
        fwht_horizontal_f16(x_ptr, h_, w_, channels_);

        // Gate H: Broadcast [W] across H and C
        // Layout is [H, W, C]. Gate is [W].
        // For each H: For each W: mul by gate[w] (across C).
        elementwise_mul_broadcast_w_f16(x_ptr, gate_h_.data(), h_, w_, channels_);

        fwht_horizontal_f16(x_ptr, h_, w_, channels_); // Inverse

        // Scale by 1/W
        scale_f16(x_ptr, size, 1.0f / w_);

        // 2. Vertical Spectral Mix (Column-wise)
        // Transpose -> FWHT -> Gate -> IFWHT -> Transpose Back
        transpose_block_64x64_f16(x_ptr, buf_ptr, h_, w_, channels_);

        // Now buf is [W, H, C].
        fwht_horizontal_f16(buf_ptr, w_, h_, channels_);

        // Gate W: Broadcast [H] across W (now rows) and C.
        // Layout [W, H, C]. Gate is [H].
        elementwise_mul_broadcast_w_f16(buf_ptr, gate_w_.data(), w_, h_, channels_);

        fwht_horizontal_f16(buf_ptr, w_, h_, channels_); // Inverse

        // Scale by 1/H
        scale_f16(buf_ptr, size, 1.0f / h_);

        transpose_block_64x64_f16(buf_ptr, x_ptr, w_, h_, channels_);

        // 3. Channel Mixing (Conv1x1)
        proj_conv_.forward(scratch_x_, output);

        // Residual connection: output += input
        uint16_t* out_ptr = output.data();
        // in_ptr is const input
        add_residual_f16(out_ptr, in_ptr, size);
    }

private:
    int channels_;
    int h_;
    int w_;

    Tensor<uint16_t> gate_h_;
    Tensor<uint16_t> gate_w_;

    OptimizedConv2D_F16 proj_conv_;

    // Persistent scratch buffers (F16)
    Tensor<uint16_t> scratch_buffer_;
    Tensor<uint16_t> scratch_x_;

    // Helpers

    // FWHT Horizontal on F16 data.
    // Converts to F32, runs FWHT, converts back.
    // Optimizing this is key.
    // A row is W*C. ZenithNano uses W=64, C=64.
    // Row is 4096 elements.
    // We can process chunks.
    // BUT FWHT is across W (spatial).
    // Standard FWHT_Horizontal_AVX expects [H, W, C].
    // It transforms the W dimension for each C.
    // Wait, FWHT_Horizontal_AVX documentation says:
    // "Performs 1D FWHT along the spatial dimension (of size N), treating the channel dimension as vector elements."
    // So for [H, W, C], it does H transforms of length W. Each point is a vector of C floats.
    // Since C=64, we can process 8 channels at a time (vectorized).
    // N=64.

    void fwht_horizontal_f16(uint16_t* data, int h, int w, int c) {
        // We iterate over H rows.
        // No OMP for small H (64) to avoid overhead
        for (int r = 0; r < h; ++r) {
            uint16_t* row_base = data + r * w * c;
            // We need to transform this row [W, C].
            // The transform is along W.
            // But we process C in chunks of 8.

            // We can lift C-loop out.
            for (int k = 0; k < c; k += 8) {
                 // For this block of 8 channels, we have a sequence of W vectors.
                 // We load the entire sequence into a temporary float buffer?
                 // W=64. 64 * 8 floats = 512 floats = 2KB.
                 // We can keep it in L1.
                 float buf[64 * 8]; // [W, 8]

                 // Load and Convert
                 for (int i = 0; i < w; ++i) {
                     __m256 v = hal::f16::load_f16(row_base + i * c + k);
                     _mm256_storeu_ps(buf + i * 8, v);
                 }

                 // Perform FWHT in-place on buf
                 // fwht_1d_vectorized_avx2 expects [N * C] but here C is 8.
                 hal::x86::fwht_1d_vectorized_avx2(buf, w, 8);

                 // Store and Convert Back
                 for (int i = 0; i < w; ++i) {
                     __m256 v = _mm256_loadu_ps(buf + i * 8);
                     hal::f16::store_f16(row_base + i * c + k, v);
                 }
            }
        }
    }

    void elementwise_mul_broadcast_w_f16(uint16_t* data, const uint16_t* gate, int h, int w, int c) {
        // Gate is [W].
        for (int r = 0; r < h; ++r) {
            for (int i = 0; i < w; ++i) {
                // Gate value for this column
                uint16_t g_u = gate[i];
                __m128i g_h = _mm_set1_epi16((short)g_u);
                __m256 g_v = _mm256_cvtph_ps(g_h);

                uint16_t* ptr = data + r * w * c + i * c;
                for (int k = 0; k < c; k += 8) {
                    __m256 val = hal::f16::load_f16(ptr + k);
                    val = _mm256_mul_ps(val, g_v);
                    hal::f16::store_f16(ptr + k, val);
                }
            }
        }
    }

    void scale_f16(uint16_t* data, size_t size, float scale) {
        __m256 s = _mm256_set1_ps(scale);
        size_t n = size;
        for (size_t i = 0; i < n; i += 8) {
            if (i + 8 <= n) {
                __m256 v = hal::f16::load_f16(data + i);
                v = _mm256_mul_ps(v, s);
                hal::f16::store_f16(data + i, v);
            } else {
                 // Tail
                 for(size_t j=i; j<n; ++j) {
                     __m128i h = _mm_set1_epi16((short)data[j]);
                      // cvtph_ps returns __m256 (8 floats), but we only populated the bottom one.
                      // We can cast to __m128 and extract.
                      __m256 v = _mm256_cvtph_ps(h);
                      float f = _mm_cvtss_f32(_mm256_castps256_ps128(v));
                     f *= scale;
                     __m128i r = _mm256_cvtps_ph(_mm256_set1_ps(f), 0);
                     data[j] = (uint16_t)_mm_cvtsi128_si32(r);
                 }
            }
        }
    }

    void transpose_block_64x64_f16(const uint16_t* src, uint16_t* dst, int h, int w, int c) {
        // Transpose [H, W, C] -> [W, H, C].
        // Elements are blocks of C uint16_t.
        // H=64, W=64.
        // Block size = C * sizeof(uint16_t) = 64 * 2 = 128 bytes.
        // Cache friendly block transpose.

        int block_sz = 8;
        for (int i = 0; i < h; i += block_sz) {
            for (int j = 0; j < w; j += block_sz) {
                for (int ii = i; ii < i + block_sz && ii < h; ++ii) {
                    for (int jj = j; jj < j + block_sz && jj < w; ++jj) {
                        const uint16_t* s = src + (ii * w + jj) * c;
                        uint16_t* d = dst + (jj * h + ii) * c;
                        std::memcpy(d, s, c * sizeof(uint16_t));
                    }
                }
            }
        }
    }

    void add_residual_f16(uint16_t* out, const uint16_t* in, size_t size) {
        for (size_t i = 0; i < size; i += 8) {
            if (i + 8 <= size) {
                __m256 a = hal::f16::load_f16(out + i);
                __m256 b = hal::f16::load_f16(in + i);
                a = _mm256_add_ps(a, b);
                hal::f16::store_f16(out + i, a);
            } else {
                 // tail
            }
        }
    }
};

} // namespace layers
} // namespace dreidel
