#ifndef DREIDEL_HAL_OPS_F16_HPP
#define DREIDEL_HAL_OPS_F16_HPP

#include <immintrin.h>
#include <cstdint>

namespace dreidel {
namespace hal {
namespace f16 {

// Convert 8 F16 (in __m128i) to 8 F32 (in __m256)
inline __m256 cvt_f16_to_f32(__m128i h) {
    return _mm256_cvtph_ps(h);
}

// Convert 8 F32 (in __m256) to 8 F16 (in __m128i)
inline __m128i cvt_f32_to_f16(__m256 f) {
    // Round to nearest even
    return _mm256_cvtps_ph(f, 0);
}

// Load 8 F16 from memory and convert to F32
inline __m256 load_f16(const uint16_t* ptr) {
    __m128i h = _mm_loadu_si128((const __m128i*)ptr);
    return _mm256_cvtph_ps(h);
}

// Store 8 F32 to memory as F16
inline void store_f16(uint16_t* ptr, __m256 f) {
    __m128i h = _mm256_cvtps_ph(f, 0);
    _mm_storeu_si128((__m128i*)ptr, h);
}

} // namespace f16
} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_OPS_F16_HPP
