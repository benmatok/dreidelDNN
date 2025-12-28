#ifndef DREIDEL_CORE_PACKET_HPP
#define DREIDEL_CORE_PACKET_HPP

#include "../hal/ops.hpp"
#include <type_traits>

namespace dreidel {
namespace core {

/**
 * @brief Abstraction for SIMD operations.
 *
 * Wraps the platform-specific HAL operations into a nicer API.
 */
template <typename T>
struct Packet {
    // Placeholder for non-specialized types
};

// Specialization for float using the ActiveOps from HAL
template <>
struct Packet<float> {
    using Ops = dreidel::hal::ActiveOps;

    // Determine native type safely
    // If Ops is generic, native_type is float
    // If Ops is x86, native_type is __m256 or __m512
    // We can deduce it from the return type of load(float*)
    using native_type = decltype(Ops::load(static_cast<const float*>(nullptr)));

    static constexpr int width = Ops::SIMD_WIDTH;

    native_type value;

    Packet() {}
    Packet(native_type v) : value(v) {}
    Packet(const float* ptr) : value(Ops::load(ptr)) {}

    void store(float* ptr) const {
        Ops::store(ptr, value);
    }

    Packet operator+(const Packet& other) const {
        return Packet(Ops::add(value, other.value));
    }

    Packet operator-(const Packet& other) const {
        return Packet(Ops::sub(value, other.value));
    }

    Packet operator*(const Packet& other) const {
        return Packet(Ops::mul(value, other.value));
    }

    // In-place Butterfly: a = a+b, b = a-b (original a)
    static void butterfly(Packet& a, Packet& b) {
        Ops::butterfly(a.value, b.value);
    }
};

} // namespace core
} // namespace dreidel

#endif // DREIDEL_CORE_PACKET_HPP
