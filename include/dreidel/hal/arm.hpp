#ifndef DREIDEL_HAL_ARM_HPP
#define DREIDEL_HAL_ARM_HPP

#include "defs.hpp"

#ifdef DREIDEL_ARCH_ARM_NEON
#include <arm_neon.h>
#endif

namespace dreidel {
namespace hal {
namespace arm {

#ifdef DREIDEL_ARCH_ARM_NEON

struct Ops {
    static constexpr int SIMD_WIDTH = 4; // 4 floats in 128 bits

    static inline float32x4_t load(const float* src) {
        return vld1q_f32(src);
    }

    static inline void store(float* dst, float32x4_t val) {
        vst1q_f32(dst, val);
    }

    static inline float32x4_t add(float32x4_t a, float32x4_t b) {
        return vaddq_f32(a, b);
    }

    static inline float32x4_t sub(float32x4_t a, float32x4_t b) {
        return vsubq_f32(a, b);
    }

    static inline void butterfly(float32x4_t& a, float32x4_t& b) {
        float32x4_t u = a;
        float32x4_t v = b;
        a = vaddq_f32(u, v);
        b = vsubq_f32(u, v);
    }
};

#else
using Ops = dreidel::hal::generic::Ops;
#endif

} // namespace arm
} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_ARM_HPP
