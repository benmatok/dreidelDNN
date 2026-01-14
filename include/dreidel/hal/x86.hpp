#ifndef DREIDEL_HAL_X86_HPP
#define DREIDEL_HAL_X86_HPP

#include <immintrin.h>
#include "defs.hpp"

namespace dreidel {
namespace hal {
namespace x86 {

#ifdef DREIDEL_ARCH_AVX512

struct Ops {
    static constexpr int SIMD_WIDTH = 16; // 16 floats in 512 bits

    // Load 16 floats
    static inline __m512 load(const float* src) {
        return _mm512_loadu_ps(src);
    }

    // Store 16 floats
    static inline void store(float* dst, __m512 val) {
        _mm512_storeu_ps(dst, val);
    }

    // Streaming Store (Non-Temporal)
    static inline void stream_store(float* dst, __m512 val) {
        _mm512_stream_ps(dst, val);
    }

    static inline __m512 add(__m512 a, __m512 b) {
        return _mm512_add_ps(a, b);
    }

    static inline __m512 sub(__m512 a, __m512 b) {
        return _mm512_sub_ps(a, b);
    }

    static inline __m512 mul(__m512 a, __m512 b) {
        return _mm512_mul_ps(a, b);
    }

    // Butterfly on registers
    static inline void butterfly(__m512& a, __m512& b) {
        __m512 u = a;
        __m512 v = b;
        a = _mm512_add_ps(u, v);
        b = _mm512_sub_ps(u, v);
    }

    // Gather 16 floats from base + indices * 4
    static inline __m512 gather(const float* base_addr, __m512i vindex) {
        // scale=4 because floats are 4 bytes
        return _mm512_i32gather_ps(vindex, (void*)base_addr, 4);
    }

    // Approximate Reciprocal
    static inline __m512 rcp_approx(__m512 a) {
        return _mm512_rcp14_ps(a);
    }

    // Approximate Reciprocal Sqrt
    static inline __m512 rsqrt_approx(__m512 a) {
        return _mm512_rsqrt14_ps(a);
    }
};

#elif defined(DREIDEL_ARCH_AVX2)

struct Ops {
    static constexpr int SIMD_WIDTH = 8; // 8 floats in 256 bits

    static inline __m256 load(const float* src) {
        return _mm256_loadu_ps(src);
    }

    static inline void store(float* dst, __m256 val) {
        _mm256_storeu_ps(dst, val);
    }

    // Streaming Store (Non-Temporal)
    static inline void stream_store(float* dst, __m256 val) {
        _mm256_stream_ps(dst, val);
    }

    static inline __m256 add(__m256 a, __m256 b) {
        return _mm256_add_ps(a, b);
    }

    static inline __m256 sub(__m256 a, __m256 b) {
        return _mm256_sub_ps(a, b);
    }

    static inline __m256 mul(__m256 a, __m256 b) {
        return _mm256_mul_ps(a, b);
    }

    static inline void butterfly(__m256& a, __m256& b) {
        __m256 u = a;
        __m256 v = b;
        a = _mm256_add_ps(u, v);
        b = _mm256_sub_ps(u, v);
    }

    // Gather 8 floats
    static inline __m256 gather(const float* base_addr, __m256i vindex) {
        return _mm256_i32gather_ps(base_addr, vindex, 4);
    }

    // Approximate Reciprocal (Error ~0.00036)
    static inline __m256 rcp_approx(__m256 a) {
        return _mm256_rcp_ps(a);
    }

    // Approximate Reciprocal Sqrt
    static inline __m256 rsqrt_approx(__m256 a) {
        return _mm256_rsqrt_ps(a);
    }
};

#else
// Fallback if included but no flags set (should not happen if guarded correctly)
using Ops = dreidel::hal::generic::Ops;
#endif

// --- Register-Resident FWHT Helpers (AVX2 Optimized) ---

// Perform FWHT on 8 elements within a YMM register (Float)
inline void fwht8_avx2(__m256& r) {
#ifdef DREIDEL_ARCH_AVX2
    // Stage 1 (Stride 1): Pairs (0,1), (2,3)...
    {
        __m256 u = _mm256_permute_ps(r, 0xA0); // 10100000 -> 0,0,2,2
        __m256 v = _mm256_permute_ps(r, 0xF5); // 11110101 -> 1,1,3,3

        __m256 s = _mm256_add_ps(u, v);
        __m256 d = _mm256_sub_ps(u, v);

        r = _mm256_blend_ps(s, d, 0xAA); // 10101010 -> Take d at odd, s at even
    }

    // Stage 2 (Stride 2): Pairs (0,2), (1,3)...
    {
        __m256 u = _mm256_permute_ps(r, 0x44); // 01000100 -> 0,1,0,1
        __m256 v = _mm256_permute_ps(r, 0xEE); // 11101110 -> 2,3,2,3

        __m256 s = _mm256_add_ps(u, v);
        __m256 d = _mm256_sub_ps(u, v);

        r = _mm256_blend_ps(s, d, 0xCC); // 11001100
    }

    // Stage 3 (Stride 4): Pairs (0,4), (1,5), (2,6), (3,7)
    {
        __m256 u = _mm256_permute2f128_ps(r, r, 0x00); // Low 128 to both
        __m256 v = _mm256_permute2f128_ps(r, r, 0x11); // High 128 to both

        __m256 s = _mm256_add_ps(u, v);
        __m256 d = _mm256_sub_ps(u, v);

        r = _mm256_blend_ps(s, d, 0xF0);
    }
#endif
}

// 16-point Register Resident FWHT (2 Registers)
inline void fwht16_avx2(float* data) {
#ifdef DREIDEL_ARCH_AVX2
    __m256 r0 = _mm256_loadu_ps(data + 0);
    __m256 r1 = _mm256_loadu_ps(data + 8);

    fwht8_avx2(r0); fwht8_avx2(r1);

    // Stride 8
    Ops::butterfly(r0, r1);

    _mm256_storeu_ps(data + 0, r0);
    _mm256_storeu_ps(data + 8, r1);
#endif
}

// 32-point Register Resident FWHT (4 Registers)
inline void fwht32_avx2(float* data) {
#ifdef DREIDEL_ARCH_AVX2
    __m256 r0 = _mm256_loadu_ps(data + 0);
    __m256 r1 = _mm256_loadu_ps(data + 8);
    __m256 r2 = _mm256_loadu_ps(data + 16);
    __m256 r3 = _mm256_loadu_ps(data + 24);

    fwht8_avx2(r0); fwht8_avx2(r1); fwht8_avx2(r2); fwht8_avx2(r3);

    // Stride 8
    Ops::butterfly(r0, r1);
    Ops::butterfly(r2, r3);

    // Stride 16
    Ops::butterfly(r0, r2);
    Ops::butterfly(r1, r3);

    _mm256_storeu_ps(data + 0, r0);
    _mm256_storeu_ps(data + 8, r1);
    _mm256_storeu_ps(data + 16, r2);
    _mm256_storeu_ps(data + 24, r3);
#endif
}

// 64-point Register Resident FWHT (8 Registers)
inline void fwht64_avx2(float* data) {
#ifdef DREIDEL_ARCH_AVX2
    __m256 r0 = _mm256_loadu_ps(data + 0);
    __m256 r1 = _mm256_loadu_ps(data + 8);
    __m256 r2 = _mm256_loadu_ps(data + 16);
    __m256 r3 = _mm256_loadu_ps(data + 24);
    __m256 r4 = _mm256_loadu_ps(data + 32);
    __m256 r5 = _mm256_loadu_ps(data + 40);
    __m256 r6 = _mm256_loadu_ps(data + 48);
    __m256 r7 = _mm256_loadu_ps(data + 56);

    fwht8_avx2(r0); fwht8_avx2(r1); fwht8_avx2(r2); fwht8_avx2(r3);
    fwht8_avx2(r4); fwht8_avx2(r5); fwht8_avx2(r6); fwht8_avx2(r7);

    // Stride 8
    Ops::butterfly(r0, r1);
    Ops::butterfly(r2, r3);
    Ops::butterfly(r4, r5);
    Ops::butterfly(r6, r7);

    // Stride 16
    Ops::butterfly(r0, r2);
    Ops::butterfly(r1, r3);
    Ops::butterfly(r4, r6);
    Ops::butterfly(r5, r7);

    // Stride 32
    Ops::butterfly(r0, r4);
    Ops::butterfly(r1, r5);
    Ops::butterfly(r2, r6);
    Ops::butterfly(r3, r7);

    _mm256_storeu_ps(data + 0, r0);
    _mm256_storeu_ps(data + 8, r1);
    _mm256_storeu_ps(data + 16, r2);
    _mm256_storeu_ps(data + 24, r3);
    _mm256_storeu_ps(data + 32, r4);
    _mm256_storeu_ps(data + 40, r5);
    _mm256_storeu_ps(data + 48, r6);
    _mm256_storeu_ps(data + 56, r7);
#endif
}

// 128-point Register Resident FWHT (16 Registers - Full YMM utilization)
inline void fwht128_avx2(float* data) {
#ifdef DREIDEL_ARCH_AVX2
    // Loads
    __m256 r0  = _mm256_loadu_ps(data + 0);
    __m256 r1  = _mm256_loadu_ps(data + 8);
    __m256 r2  = _mm256_loadu_ps(data + 16);
    __m256 r3  = _mm256_loadu_ps(data + 24);
    __m256 r4  = _mm256_loadu_ps(data + 32);
    __m256 r5  = _mm256_loadu_ps(data + 40);
    __m256 r6  = _mm256_loadu_ps(data + 48);
    __m256 r7  = _mm256_loadu_ps(data + 56);
    __m256 r8  = _mm256_loadu_ps(data + 64);
    __m256 r9  = _mm256_loadu_ps(data + 72);
    __m256 r10 = _mm256_loadu_ps(data + 80);
    __m256 r11 = _mm256_loadu_ps(data + 88);
    __m256 r12 = _mm256_loadu_ps(data + 96);
    __m256 r13 = _mm256_loadu_ps(data + 104);
    __m256 r14 = _mm256_loadu_ps(data + 112);
    __m256 r15 = _mm256_loadu_ps(data + 120);

    // Intra
    fwht8_avx2(r0); fwht8_avx2(r1); fwht8_avx2(r2); fwht8_avx2(r3);
    fwht8_avx2(r4); fwht8_avx2(r5); fwht8_avx2(r6); fwht8_avx2(r7);
    fwht8_avx2(r8); fwht8_avx2(r9); fwht8_avx2(r10); fwht8_avx2(r11);
    fwht8_avx2(r12); fwht8_avx2(r13); fwht8_avx2(r14); fwht8_avx2(r15);

    // Inter
    // Stride 8
    Ops::butterfly(r0, r1); Ops::butterfly(r2, r3);
    Ops::butterfly(r4, r5); Ops::butterfly(r6, r7);
    Ops::butterfly(r8, r9); Ops::butterfly(r10, r11);
    Ops::butterfly(r12, r13); Ops::butterfly(r14, r15);

    // Stride 16
    Ops::butterfly(r0, r2); Ops::butterfly(r1, r3);
    Ops::butterfly(r4, r6); Ops::butterfly(r5, r7);
    Ops::butterfly(r8, r10); Ops::butterfly(r9, r11);
    Ops::butterfly(r12, r14); Ops::butterfly(r13, r15);

    // Stride 32
    Ops::butterfly(r0, r4); Ops::butterfly(r1, r5);
    Ops::butterfly(r2, r6); Ops::butterfly(r3, r7);
    Ops::butterfly(r8, r12); Ops::butterfly(r9, r13);
    Ops::butterfly(r10, r14); Ops::butterfly(r11, r15);

    // Stride 64
    Ops::butterfly(r0, r8); Ops::butterfly(r1, r9);
    Ops::butterfly(r2, r10); Ops::butterfly(r3, r11);
    Ops::butterfly(r4, r12); Ops::butterfly(r5, r13);
    Ops::butterfly(r6, r14); Ops::butterfly(r7, r15);

    // Stores
    _mm256_storeu_ps(data + 0, r0);
    _mm256_storeu_ps(data + 8, r1);
    _mm256_storeu_ps(data + 16, r2);
    _mm256_storeu_ps(data + 24, r3);
    _mm256_storeu_ps(data + 32, r4);
    _mm256_storeu_ps(data + 40, r5);
    _mm256_storeu_ps(data + 48, r6);
    _mm256_storeu_ps(data + 56, r7);
    _mm256_storeu_ps(data + 64, r8);
    _mm256_storeu_ps(data + 72, r9);
    _mm256_storeu_ps(data + 80, r10);
    _mm256_storeu_ps(data + 88, r11);
    _mm256_storeu_ps(data + 96, r12);
    _mm256_storeu_ps(data + 104, r13);
    _mm256_storeu_ps(data + 112, r14);
    _mm256_storeu_ps(data + 120, r15);
#endif
}

// --- ZenithLite Kernels (AVX2 Optimized) ---

// Vectorized FWHT for NHWC layout.
// Performs 1D FWHT along the spatial dimension (of size N), treating the
// channel dimension (of size C) as vector elements.
// data: Pointer to the start of the row/column buffer [N * C]
// n: Transform length (spatial dimension). Must be a power of 2.
// c: Number of channels.
inline void fwht_1d_vectorized_avx2(float* data, size_t n, size_t c) {
#ifdef DREIDEL_ARCH_AVX2
    for (size_t len = 1; len < n; len *= 2) {
        for (size_t i = 0; i < n; i += 2 * len) {
            for (size_t j = 0; j < len; j++) {
                float* p_u = data + (i + j) * c;
                float* p_v = data + (i + len + j) * c;

                size_t k = 0;
                // Process 8 channels at a time
                for (; k + 8 <= c; k += 8) {
                    __m256 u_vec = _mm256_loadu_ps(p_u + k);
                    __m256 v_vec = _mm256_loadu_ps(p_v + k);

                    __m256 sum = _mm256_add_ps(u_vec, v_vec);
                    __m256 diff = _mm256_sub_ps(u_vec, v_vec);

                    _mm256_storeu_ps(p_u + k, sum);
                    _mm256_storeu_ps(p_v + k, diff);
                }
                // Scalar fallback for remaining channels
                for (; k < c; ++k) {
                    float u_val = p_u[k];
                    float v_val = p_v[k];
                    p_u[k] = u_val + v_val;
                    p_v[k] = u_val - v_val;
                }
            }
        }
    }
#else
    // Generic fallback if AVX2 not available (should be guarded by caller ideally)
     for (size_t len = 1; len < n; len *= 2) {
        for (size_t i = 0; i < n; i += 2 * len) {
            for (size_t j = 0; j < len; j++) {
                float* p_u = data + (i + j) * c;
                float* p_v = data + (i + len + j) * c;
                for (size_t k = 0; k < c; ++k) {
                     float u_val = p_u[k];
                     float v_val = p_v[k];
                     p_u[k] = u_val + v_val;
                     p_v[k] = u_val - v_val;
                }
            }
        }
    }
#endif
}

// Separable Spectral Gating with Broadcast.
// Multiplies each spatial position's channel vector by a scalar gate value.
// data: [N * C]
// gate: [N]
// n: Spatial size
// c: Channel size
inline void spectral_gate_separable_avx2(float* data, const float* gate, size_t n, size_t c) {
#ifdef DREIDEL_ARCH_AVX2
    for (size_t i = 0; i < n; ++i) {
        float g_val = gate[i];
        __m256 g_vec = _mm256_set1_ps(g_val);
        float* row = data + i * c;

        size_t k = 0;
        for (; k + 8 <= c; k += 8) {
            __m256 x = _mm256_loadu_ps(row + k);
            x = _mm256_mul_ps(x, g_vec);
            _mm256_storeu_ps(row + k, x);
        }
        for (; k < c; ++k) {
            row[k] *= g_val;
        }
    }
#else
    for (size_t i = 0; i < n; ++i) {
        float g_val = gate[i];
        float* row = data + i * c;
        for(size_t k=0; k<c; ++k) row[k] *= g_val;
    }
#endif
}

// 1x1 Group Convolution Helper
// Performs: Y = W * X + Bias (optional)
// Optimized for NHWC layout where we process pixels independently (or blocked).
// input: [NumPixels * Cin]
// output: [NumPixels * Cout]
// weights: [Groups, Cout/G, Cin/G] (Flattended)
// groups: G
// num_pixels: N*H*W
// cin: Cin
// cout: Cout
inline void group_conv_1x1_avx2(const float* input, float* output, const float* weights, const float* bias,
                                size_t num_pixels, size_t cin, size_t cout, size_t groups) {
    size_t cin_g = cin / groups;
    size_t cout_g = cout / groups;
    // We assume weights are packed as [Groups, Cout_g, Cin_g]

#ifdef DREIDEL_ARCH_AVX2
    // Process pixels
    // Block blocking could be added for cache, but simple pixel loop first.
    #pragma omp parallel for
    for (size_t i = 0; i < num_pixels; ++i) {
        const float* p_in = input + i * cin;
        float* p_out = output + i * cout;

        for (size_t g = 0; g < groups; ++g) {
            const float* w_g = weights + g * cout_g * cin_g;
            const float* in_g = p_in + g * cin_g;
            float* out_g = p_out + g * cout_g;
            const float* b_g = bias ? (bias + g * cout_g) : nullptr;

            for (size_t co = 0; co < cout_g; ++co) {
                __m256 sum_vec = _mm256_setzero_ps();
                size_t ci = 0;
                // Dot product
                for (; ci + 8 <= cin_g; ci += 8) {
                    __m256 v_in = _mm256_loadu_ps(in_g + ci);
                    __m256 v_w = _mm256_loadu_ps(w_g + co * cin_g + ci); // Weight layout [Cout_g, Cin_g]
                    sum_vec = _mm256_fmadd_ps(v_in, v_w, sum_vec);
                }
                // Horizontal sum
                float sum_arr[8];
                _mm256_storeu_ps(sum_arr, sum_vec);
                float val = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                            sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];

                for (; ci < cin_g; ++ci) {
                    val += in_g[ci] * w_g[co * cin_g + ci];
                }

                if (b_g) val += b_g[co];
                out_g[co] = val;
            }
        }
    }
#else
    // Fallback
     #pragma omp parallel for
    for (size_t i = 0; i < num_pixels; ++i) {
        const float* p_in = input + i * cin;
        float* p_out = output + i * cout;
        for (size_t g = 0; g < groups; ++g) {
            const float* w_g = weights + g * cout_g * cin_g;
            const float* in_g = p_in + g * cin_g;
            float* out_g = p_out + g * cout_g;
            const float* b_g = bias ? (bias + g * cout_g) : nullptr;
            for (size_t co = 0; co < cout_g; ++co) {
                float val = 0;
                for (size_t ci = 0; ci < cin_g; ++ci) {
                    val += in_g[ci] * w_g[co * cin_g + ci];
                }
                if (b_g) val += b_g[co];
                out_g[co] = val;
            }
        }
    }
#endif
}

} // namespace x86
} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_X86_HPP
