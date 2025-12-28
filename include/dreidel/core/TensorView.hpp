#ifndef DREIDEL_CORE_TENSORVIEW_HPP
#define DREIDEL_CORE_TENSORVIEW_HPP

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <array>

namespace dreidel {
namespace core {

/**
 * @brief A non-owning view of a tensor.
 *
 * TensorView does not allocate or free memory. It acts as a window over
 * a pre-allocated memory buffer. It is designed for zero-allocation
 * runtime environments.
 *
 * Shape is stored in a fixed-size array to avoid heap allocation.
 */
template <typename T, size_t MAX_RANK = 4>
class TensorView {
public:
    TensorView() : data_(nullptr), size_(0), rank_(0) {}

    /**
     * @brief Construct a TensorView from a raw pointer and shape (initializer list).
     */
    TensorView(T* data, std::initializer_list<size_t> shape)
        : data_(data) {
        if (shape.size() > MAX_RANK) throw std::length_error("Rank exceeds MAX_RANK");

        rank_ = shape.size();
        size_t idx = 0;
        size_ = 1;
        for (size_t s : shape) {
            shape_[idx++] = s;
            size_ *= s;
        }
    }

    /**
     * @brief Construct a TensorView from a raw pointer and shape (vector-like container).
     * Accept generic container but iterate to avoid copy allocation.
     */
    template <typename Container>
    TensorView(T* data, const Container& shape)
        : data_(data) {
        if (shape.size() > MAX_RANK) throw std::length_error("Rank exceeds MAX_RANK");

        rank_ = shape.size();
        size_t idx = 0;
        size_ = 1;
        for (size_t s : shape) {
            shape_[idx++] = s;
            size_ *= s;
        }
    }

    // 1D Constructor shortcut
    TensorView(T* data, size_t size)
        : data_(data), size_(size), rank_(1) {
        shape_[0] = size;
    }

    // Accessors
    T* data() { return data_; }
    const T* data() const { return data_; }

    size_t size() const { return size_; }

    // Returns a view of the shape (not vector)
    // For compatibility with code expecting vector, we might need an adapter,
    // but here we expose the raw array or a span-like object.
    // However, to keep it simple and strictly zero-alloc, we provide accessors.
    size_t dim(size_t idx) const {
        if (idx >= rank_) return 1; // Broadcast/Safe
        return shape_[idx];
    }

    const size_t* shape_ptr() const { return shape_; }
    size_t rank() const { return rank_; }

    // Legacy support: construct a vector on demand (ONLY FOR DEBUG/INIT)
    std::vector<size_t> shape() const {
        return std::vector<size_t>(shape_, shape_ + rank_);
    }

    // Indexing
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    T& at(size_t i) {
        if (i >= size_) throw std::out_of_range("TensorView index out of range");
        return data_[i];
    }

    const T& at(size_t i) const {
        if (i >= size_) throw std::out_of_range("TensorView index out of range");
        return data_[i];
    }

    // Helper to get raw pointer at offset
    T* ptr_at(size_t i) { return data_ + i; }
    const T* ptr_at(size_t i) const { return data_ + i; }

    // Reshape (returns new view over same data)
    TensorView<T, MAX_RANK> reshape(std::initializer_list<size_t> new_shape) const {
        size_t new_size = 1;
        for (size_t s : new_shape) new_size *= s;
        if (new_size != size_) {
            // In embedded, assert might be better
             throw std::runtime_error("Reshape size mismatch");
        }
        return TensorView<T, MAX_RANK>(data_, new_shape);
    }

    void fill(T val) {
        for (size_t i = 0; i < size_; ++i) data_[i] = val;
    }

private:
    T* data_;
    size_t shape_[MAX_RANK];
    size_t size_;
    size_t rank_;
};

} // namespace core
} // namespace dreidel

#endif // DREIDEL_CORE_TENSORVIEW_HPP
