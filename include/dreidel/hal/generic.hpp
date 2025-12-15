#ifndef DREIDEL_HAL_GENERIC_HPP
#define DREIDEL_HAL_GENERIC_HPP

#include <cmath>

namespace dreidel {
namespace hal {
namespace generic {

struct Ops {
    static constexpr int SIMD_WIDTH = 1; // Scalar

    template <typename T>
    static inline T load(const T* src) {
        return *src;
    }

    template <typename T>
    static inline void store(T* dst, const T& val) {
        *dst = val;
    }

    template <typename T>
    static inline T add(const T& a, const T& b) {
        return a + b;
    }

    template <typename T>
    static inline T sub(const T& a, const T& b) {
        return a - b;
    }

    // In-place butterfly: a, b -> a+b, a-b
    template <typename T>
    static inline void butterfly(T& a, T& b) {
        T u = a;
        T v = b;
        a = u + v;
        b = u - v;
    }
};

} // namespace generic
} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_GENERIC_HPP
