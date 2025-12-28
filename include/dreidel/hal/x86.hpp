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
        return _mm512_load_ps(src);
    }

    // Store 16 floats
    static inline void store(float* dst, __m512 val) {
        _mm512_store_ps(dst, val);
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
};

#elif defined(DREIDEL_ARCH_AVX2)

struct Ops {
    static constexpr int SIMD_WIDTH = 8; // 8 floats in 256 bits

    static inline __m256 load(const float* src) {
        return _mm256_load_ps(src);
    }

    static inline void store(float* dst, __m256 val) {
        _mm256_store_ps(dst, val);
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
};

#else
// Fallback if included but no flags set (should not happen if guarded correctly)
using Ops = dreidel::hal::generic::Ops;
#endif

// --- Register-Resident FWHT Helpers (AVX2 Optimized) ---
// Defined outside Ops to be generally available in namespace x86

// Perform FWHT on 8 elements within a YMM register
inline void fwht8_avx2(__m256& r) {
#ifdef DREIDEL_ARCH_AVX2
    // Stage 1 (Stride 1): Pairs (0,1), (2,3)...
    {
        // Swap adjacent: [1 0 3 2 ...]
        __m256 swapped = _mm256_permute_ps(r, 0xB1); // 10110001

        // Sums (0+1, 1+0...) -> We want 0+1 at pos 0.
        __m256 sum = _mm256_add_ps(r, swapped);

        // Diffs (0-1, 1-0...) -> We want 0-1 at pos 1.
        // Currently pos 1 is 1-0 = -(0-1).
        __m256 diff = _mm256_sub_ps(r, swapped);

        // Fix signs for diffs at odd positions
        // Mask for odd positions: 0, -0.0, 0, -0.0...
        // Actually simpler:
        // We want: [x0+x1, x0-x1, ...]
        // sum has [x0+x1, x1+x0...] -> Correct at even.
        // diff has [x0-x1, x1-x0...] -> x1-x0 = -(x0-x1). Correct magnitude at odd, wrong sign.
        // Multiply odd lanes by -1?
        // Or:
        // u = [x0 x0 x2 x2 ...]
        // v = [x1 x1 x3 x3 ...]
        // r = u + v*sign?
        // _mm256_blend_ps is fast.

        __m256 u = _mm256_permute_ps(r, 0xA0); // 10100000 -> 0,0,2,2
        __m256 v = _mm256_permute_ps(r, 0xF5); // 11110101 -> 1,1,3,3

        __m256 s = _mm256_add_ps(u, v);
        __m256 d = _mm256_sub_ps(u, v);

        r = _mm256_blend_ps(s, d, 0xAA); // 10101010 -> Take d at odd, s at even
    }

    // Stage 2 (Stride 2): Pairs (0,2), (1,3)...
    {
        // Align 0 with 2, 1 with 3.
        // Swap 64-bit blocks (pairs of floats)
        __m256 swapped = _mm256_permute_ps(r, 0x4E); // 01001110 -> 2301... swaps halves of 128-bit lane

        // We want:
        // Pos 0: x0+x2. Pos 2: x0-x2.
        // swapped has x2 at pos 0.

        // Again, explicit broadcast is clearer but maybe slower.
        // u = [0 1 0 1 ...] from [0 1 2 3 ...]
        __m256 u = _mm256_permute_ps(r, 0x44); // 01000100 -> 0,1,0,1
        // v = [2 3 2 3 ...]
        __m256 v = _mm256_permute_ps(r, 0xEE); // 11101110 -> 2,3,2,3

        __m256 s = _mm256_add_ps(u, v);
        __m256 d = _mm256_sub_ps(u, v);

        r = _mm256_blend_ps(s, d, 0xCC); // 11001100
    }

    // Stage 3 (Stride 4): Pairs (0,4), (1,5), (2,6), (3,7)
    {
        // Cross 128-bit lanes.
        // u = [0 1 2 3 0 1 2 3]
        __m256 u = _mm256_permute2f128_ps(r, r, 0x00); // Low 128 to both
        // v = [4 5 6 7 4 5 6 7]
        __m256 v = _mm256_permute2f128_ps(r, r, 0x11); // High 128 to both

        __m256 s = _mm256_add_ps(u, v);
        __m256 d = _mm256_sub_ps(u, v);

        // Blend not available for across lanes easily?
        // _mm256_blend_ps works on all 8 floats.
        // Mask 0xF0 (11110000) -> High 4 from d, Low 4 from s.
        r = _mm256_blend_ps(s, d, 0xF0);
    }
#endif
}

// 64-point Register Resident FWHT
// Requires 8 YMM registers (AVX2)
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

    // Step 1: Intra-Register FWHT (Rows)
    // Applies transforms for strides 1, 2, 4
    fwht8_avx2(r0); fwht8_avx2(r1); fwht8_avx2(r2); fwht8_avx2(r3);
    fwht8_avx2(r4); fwht8_avx2(r5); fwht8_avx2(r6); fwht8_avx2(r7);

    // Step 2: Inter-Register FWHT (Columns)
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

} // namespace x86
} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_X86_HPP
