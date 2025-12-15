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

    // Butterfly on registers
    static inline void butterfly(__m512& a, __m512& b) {
        __m512 u = a;
        __m512 v = b;
        a = _mm512_add_ps(u, v);
        b = _mm512_sub_ps(u, v);
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

    static inline void butterfly(__m256& a, __m256& b) {
        __m256 u = a;
        __m256 v = b;
        a = _mm256_add_ps(u, v);
        b = _mm256_sub_ps(u, v);
    }
};

#else
// Fallback if included but no flags set (should not happen if guarded correctly)
using Ops = dreidel::hal::generic::Ops;
#endif

} // namespace x86
} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_X86_HPP
