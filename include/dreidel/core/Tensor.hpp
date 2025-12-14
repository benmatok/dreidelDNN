#ifndef DREIDEL_CORE_TENSOR_HPP
#define DREIDEL_CORE_TENSOR_HPP

#include <vector>
#include <initializer_list>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include "Backend.hpp"

// Check for OpenMP support
#if defined(_OPENMP)
#include <omp.h>
#define DREIDEL_SIMD_LOOP _Pragma("omp simd")
#define DREIDEL_PARALLEL_LOOP _Pragma("omp parallel for")
#else
#define DREIDEL_SIMD_LOOP
#define DREIDEL_PARALLEL_LOOP
#endif

namespace dreidel {

template <typename T, BackendType B = BackendType::CPU>
class Tensor {
public:
    Tensor() = default;

    // Constructor with shape
    Tensor(const std::vector<size_t>& shape) : shape_(shape) {
        size_t total_size = 1;
        for (size_t s : shape) total_size *= s;
        data_.resize(total_size);
    }

    // Constructor with shape and initial data
    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data) : shape_(shape), data_(data) {
        size_t total_size = 1;
        for (size_t s : shape) total_size *= s;
        if (data.size() != total_size) {
             throw std::invalid_argument("Data size does not match shape dimensions.");
        }
    }

    virtual ~Tensor() = default;

    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

    // Element access (simple flat index for now)
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }

    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    // Element-wise operations
    Tensor<T, B> operator+(const Tensor<T, B>& other) const {
        if (this->shape_ != other.shape_) {
             throw std::invalid_argument("Shapes must match for addition.");
        }
        Tensor<T, B> result(this->shape_);
        size_t n = this->data_.size();

        DREIDEL_SIMD_LOOP
        for (size_t i = 0; i < n; ++i) {
            result.data_[i] = this->data_[i] + other.data_[i];
        }
        return result;
    }

    Tensor<T, B> operator*(const Tensor<T, B>& other) const {
        if (this->shape_ != other.shape_) {
             throw std::invalid_argument("Shapes must match for element-wise multiplication.");
        }
        Tensor<T, B> result(this->shape_);
        size_t n = this->data_.size();

        DREIDEL_SIMD_LOOP
        for (size_t i = 0; i < n; ++i) {
            result.data_[i] = this->data_[i] * other.data_[i];
        }
        return result;
    }

    // Matrix Multiplication (Naive GEMM with SIMD hints)
    // Assumes 2D Tensors for simplicity in Phase 2
    Tensor<T, B> matmul(const Tensor<T, B>& other) const {
        if (this->shape_.size() != 2 || other.shape_.size() != 2) {
             throw std::invalid_argument("Matmul only supports 2D tensors for now.");
        }
        size_t M = this->shape_[0];
        size_t K = this->shape_[1];
        size_t K2 = other.shape_[0];
        size_t N = other.shape_[1];

        if (K != K2) {
             throw std::invalid_argument("Inner dimensions must match for matmul.");
        }

        Tensor<T, B> result({M, N});
        result.fill(0);

        // Naive O(M*N*K) implementation with loop reordering (IKJ) for cache efficiency
        // and autovectorization hints
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                T a_val = this->data_[i * K + k];

                // Hint to compiler that these pointers are restrict-like (no aliasing) ideally,
                // but just SIMD pragma helps.
                DREIDEL_SIMD_LOOP
                for (size_t j = 0; j < N; ++j) {
                    result.data_[i * N + j] += a_val * other.data_[k * N + j];
                }
            }
        }
        return result;
    }

private:
    std::vector<size_t> shape_;
    std::vector<T> data_;
};

} // namespace dreidel

#endif // DREIDEL_CORE_TENSOR_HPP
