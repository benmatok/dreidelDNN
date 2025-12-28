#ifndef DREIDEL_CORE_ARENA_HPP
#define DREIDEL_CORE_ARENA_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace dreidel {
namespace core {

/**
 * @brief A simple linear allocator (Arena) for scratchpad memory.
 *
 * Manages a contiguous block of memory. Allocations are just pointer bumps.
 * Resetting the offset frees everything effectively.
 */
class Arena {
public:
    Arena(void* buffer, size_t size)
        : start_(static_cast<uint8_t*>(buffer)),
          current_(static_cast<uint8_t*>(buffer)),
          capacity_(size) {}

    template <typename T>
    T* allocate(size_t count, size_t alignment = 64) {
        // Align current pointer
        uintptr_t ptr_val = reinterpret_cast<uintptr_t>(current_);
        uintptr_t aligned_ptr_val = (ptr_val + alignment - 1) & ~(alignment - 1);
        uint8_t* aligned_ptr = reinterpret_cast<uint8_t*>(aligned_ptr_val);

        size_t bytes = count * sizeof(T);
        if (aligned_ptr + bytes > start_ + capacity_) {
            throw std::runtime_error("Arena Out Of Memory");
        }

        current_ = aligned_ptr + bytes;
        return reinterpret_cast<T*>(aligned_ptr);
    }

    void reset() {
        current_ = start_;
    }

    size_t used() const {
        return current_ - start_;
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    uint8_t* start_;
    uint8_t* current_;
    size_t capacity_;
};

} // namespace core
} // namespace dreidel

#endif // DREIDEL_CORE_ARENA_HPP
