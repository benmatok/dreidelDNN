#pragma once

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <omp.h>
#include "AvxFwht.hpp"

namespace dreidel {
namespace kernels {

// Fused Vertical Pipeline
// Performs:
// 1. FWHT on Row (Vertical dim)
// 2. Gate (Scale)
// 3. IFWHT (Inverse Vertical)
// 4. Conv1x1 on Row (Channel Mix)
//
// Layout:
// data: [W, H, C]
// out: [W, H, Out]

template<int C_FIXED=64, int H_FIXED=64>
inline void Fused_Vertical_Pipeline_AVX2(
    float* data,
    float* out,
    const float* weights, // [In, Out]
    const float* bias,    // [Out]
    const float* gate,    // [H]
    int W,
    int In,
    int Out)
{
    // Iterate over W (Rows of the transposed buffer)
    #pragma omp parallel for
    for (int w = 0; w < W; ++w) {
        float* row_ptr = data + w * H_FIXED * In; // In == C
        float* out_ptr = out + w * H_FIXED * Out;

        // Step 1: FWHT
        kernels::FWHT_Horizontal_AVX(row_ptr, 1, H_FIXED, In);

        // Step 2: Gate
        // row[h, c] *= gate[h]
        for(int h=0; h<H_FIXED; ++h) {
             float g = gate[h];
             float* px = row_ptr + h * In;
             // Vectorized Scale
             int c=0;
             #ifdef __AVX2__
             __m256 vg = _mm256_set1_ps(g);
             for(; c+8<=In; c+=8) {
                 _mm256_storeu_ps(px+c, _mm256_mul_ps(_mm256_loadu_ps(px+c), vg));
             }
             #endif
             for(; c<In; ++c) px[c] *= g;
        }

        // Step 3: IFWHT
        kernels::FWHT_Horizontal_AVX(row_ptr, 1, H_FIXED, In);

        // Scale by 1/H (Normalizing FWHT)
        float scale_h = 1.0f / H_FIXED;
        for(int i=0; i<H_FIXED*In; ++i) row_ptr[i] *= scale_h;

        // Step 4: Conv 1x1
        for (int h = 0; h < H_FIXED; ++h) {
            float* pixel_out = out_ptr + h * Out;
            const float* pixel_in = row_ptr + h * In;
            const float* w_base = weights;

            size_t ob = 0;
            for(; ob + 32 <= (size_t)Out; ob += 32) {
#ifdef __AVX2__
                __m256 acc0 = _mm256_loadu_ps(bias + ob + 0);
                __m256 acc1 = _mm256_loadu_ps(bias + ob + 8);
                __m256 acc2 = _mm256_loadu_ps(bias + ob + 16);
                __m256 acc3 = _mm256_loadu_ps(bias + ob + 24);

                for(int i=0; i<In; ++i) {
                    float val = pixel_in[i];
                    __m256 v_val = _mm256_set1_ps(val);
                    const float* w_i = w_base + i * Out + ob;

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
                for(int k=0; k<32; ++k) pixel_out[ob+k] = bias[ob+k];
                for(int i=0; i<In; ++i) {
                     float val = pixel_in[i];
                     const float* w_i = w_base + i * Out + ob;
                     for(int k=0; k<32; ++k) pixel_out[ob+k] += val * w_i[k];
                }
#endif
            }
            // Tail
            if (ob < (size_t)Out) {
                 for(size_t o=ob; o<(size_t)Out; ++o) pixel_out[o] = bias[o];
                 for(int i=0; i<In; ++i) {
                     float val = pixel_in[i];
                     const float* w_i = w_base + i * Out;
                     for(size_t o=ob; o<(size_t)Out; ++o) pixel_out[o] += val * w_i[o];
                 }
            }
        }
    }
}

} // namespace kernels
} // namespace dreidel
