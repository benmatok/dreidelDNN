#ifndef DREIDEL_HAL_OPS_HPP
#define DREIDEL_HAL_OPS_HPP

#include "defs.hpp"
#include "generic.hpp"
#include "x86.hpp"
#include "arm.hpp"
#include "opengl.hpp"
#include <vector>

namespace dreidel {
namespace hal {

// Select the active implementation based on macros
#if defined(DREIDEL_ARCH_AVX512) || defined(DREIDEL_ARCH_AVX2)
    using ActiveOps = x86::Ops;
#elif defined(DREIDEL_ARCH_ARM_NEON)
    using ActiveOps = arm::Ops;
#else
    using ActiveOps = generic::Ops;
#endif

// Helper function for sparse gather that abstracts away SIMD details
// Gather N elements from base using indices into out
inline void sparse_gather(const float* base, const int* indices, float* out, size_t n) {
    size_t i = 0;

#if defined(DREIDEL_ARCH_AVX512)
    // AVX-512 Loop
    for (; i + 16 <= n; i += 16) {
        __m512i vidx = _mm512_loadu_si512((const __m512i*)(indices + i));
        __m512 val = x86::Ops::gather(base, vidx);
        _mm512_storeu_ps(out + i, val);
    }
#elif defined(DREIDEL_ARCH_AVX2)
    // AVX2 Loop
    for (; i + 8 <= n; i += 8) {
        __m256i vidx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __m256 val = x86::Ops::gather(base, vidx);
        _mm256_storeu_ps(out + i, val);
    }
#endif

    // Scalar cleanup
    for (; i < n; ++i) {
        out[i] = base[indices[i]];
    }
}

} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_OPS_HPP
