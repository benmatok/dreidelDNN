#pragma once

#include <vector>
#include <memory>
#include <iostream>

namespace dreidel {
namespace core {

/**
 * @brief N-dimensional Tensor class.
 *
 * TODO: Implement aligned memory allocation for SIMD.
 * TODO: Implement shape/stride logic.
 * TODO: Add SIMD intrinsics for basic ops.
 */
template <typename T>
class Tensor {
public:
    Tensor() = default;

    // Placeholder constructor
    Tensor(std::vector<size_t> shape) : shape_(shape) {
        // Calculate size and allocate data
        size_t size = 1;
        for(auto s : shape) size *= s;
        data_.resize(size);
    }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    size_t size() const { return data_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }

    // TODO: matrix multiplication (GEMM)
    // TODO: element-wise ops

private:
    std::vector<size_t> shape_;
    std::vector<T> data_;
};

} // namespace core
} // namespace dreidel
