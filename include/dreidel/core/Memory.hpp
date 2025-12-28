#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include "Allocator.hpp"

namespace dreidel {
namespace core {

/**
 * @brief Zero-Allocation Arena Allocator.
 *
 * A simple linear allocator that allocates from a fixed-size contiguous block.
 * Pointers are invalidated only when the Arena is destroyed (or if we implemented reset, which we do).
 */
class Arena {
public:
    Arena(size_t size_bytes) : size_(size_bytes), offset_(0) {
        // Use aligned alloc for the base
        size_t align = 64;
        // Round up size to alignment
        if (size_bytes % align != 0) {
            size_bytes = ((size_bytes / align) + 1) * align;
        }

#if defined(_MSC_VER)
        data_ = static_cast<uint8_t*>(_aligned_malloc(size_bytes, align));
#else
        data_ = static_cast<uint8_t*>(std::aligned_alloc(align, size_bytes));
#endif
        if (!data_) throw std::runtime_error("Failed to allocate Arena memory");
    }

    ~Arena() {
#if defined(_MSC_VER)
        if (data_) _aligned_free(data_);
#else
        if (data_) std::free(data_);
#endif
    }

    // Allocate n elements of type T
    template <typename T>
    T* allocate(size_t count) {
        size_t bytes = count * sizeof(T);

        // align offset to sizeof(T) or 64 bytes?
        // Let's align to 64 bytes for SIMD safety
        size_t align = 64;
        size_t padding = (align - (offset_ % align)) % align;

        if (offset_ + padding + bytes > size_) {
            throw std::runtime_error("Arena Out of Memory");
        }

        offset_ += padding;
        T* ptr = reinterpret_cast<T*>(data_ + offset_);
        offset_ += bytes;

        return ptr;
    }

    void reset() {
        offset_ = 0;
    }

    size_t used() const { return offset_; }
    size_t capacity() const { return size_; }

private:
    uint8_t* data_;
    size_t size_;
    size_t offset_;
};

/**
 * @brief Non-owning Tensor View.
 *
 * Operates on memory allocated elsewhere (e.g., Arena).
 */
template <typename T>
class TensorView {
public:
    TensorView() : data_(nullptr) {}
    TensorView(T* data, const std::vector<size_t>& shape) : data_(data), shape_(shape) {
        size_ = 1;
        for(auto s : shape) size_ *= s;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

    size_t size() const { return size_; }
    const std::vector<size_t>& shape() const { return shape_; }

    T& operator[](size_t idx) { return data_[idx]; }
    const T& operator[](size_t idx) const { return data_[idx]; }

    // Helper for 4D indexing (N, H, W, C)
    T& at(size_t n, size_t h, size_t w, size_t c) {
        // Assume NHWC
        size_t H = shape_[1];
        size_t W = shape_[2];
        size_t C = shape_[3];
        return data_[((n * H + h) * W + w) * C + c];
    }

private:
    T* data_;
    std::vector<size_t> shape_;
    size_t size_ = 0;
};

} // namespace core
} // namespace dreidel
