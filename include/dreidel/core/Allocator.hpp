#ifndef DREIDEL_CORE_ALLOCATOR_HPP
#define DREIDEL_CORE_ALLOCATOR_HPP

#include <cstdlib>
#include <new>
#include <limits>
#include <vector>

namespace dreidel {
namespace core {

template <typename T, std::size_t Alignment = 64>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept {}

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    ~AlignedAllocator() = default;

    pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            throw std::bad_alloc();

        size_type bytes = n * sizeof(T);
        // Round up to multiple of Alignment
        if (bytes % Alignment != 0) {
            bytes = ((bytes / Alignment) + 1) * Alignment;
        }

        void* p = nullptr;

#if defined(_MSC_VER)
        p = _aligned_malloc(bytes, Alignment);
#else
        p = std::aligned_alloc(Alignment, bytes);
#endif

        if (!p) throw std::bad_alloc();
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) noexcept {
#if defined(_MSC_VER)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) noexcept {
    return false;
}

} // namespace core
} // namespace dreidel

#endif // DREIDEL_CORE_ALLOCATOR_HPP
