#ifndef DREIDEL_CORE_BACKEND_HPP
#define DREIDEL_CORE_BACKEND_HPP

namespace dreidel {

// Enum to define supported backends
enum class BackendType {
    CPU,
    ARM,     // Placeholder for future ARM NEON support
    OpenGL   // Placeholder for future OpenGL/Vulkan support
};

// Base Backend traits
template <BackendType B>
struct BackendTraits;

// CPU Backend specialization
template <>
struct BackendTraits<BackendType::CPU> {
    static constexpr const char* name = "CPU";
    // Future: Add alignment requirements, vector width, etc.
};

template <>
struct BackendTraits<BackendType::ARM> {
    static constexpr const char* name = "ARM";
};

template <>
struct BackendTraits<BackendType::OpenGL> {
    static constexpr const char* name = "OpenGL";
};

} // namespace dreidel

#endif // DREIDEL_CORE_BACKEND_HPP
