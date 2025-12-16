#ifndef DREIDEL_CORE_TENSOR_HPP
#define DREIDEL_CORE_TENSOR_HPP

#include <vector>
#include <initializer_list>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <random>
#include "Backend.hpp"
#include "Allocator.hpp"

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
    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data) : shape_(shape) {
        size_t total_size = 1;
        for (size_t s : shape) total_size *= s;
        if (data.size() != total_size) {
             throw std::invalid_argument("Data size does not match shape dimensions.");
        }
        data_.assign(data.begin(), data.end());
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

    // Helper for NHWC layout interpretation
    // Returns true if the tensor is 4D (N, H, W, C)
    bool is_nhwc() const {
        return shape_.size() == 4;
    }

    // Get stride of the last dimension (C in NHWC)
    // Should be 1 for contiguous last dimension
    size_t last_dim_stride() const {
        return 1;
    }

    void random(T mean, T stddev) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<T> d(mean, stddev);
        for (auto& v : data_) v = d(gen);
    }

    // Element-wise operations
    Tensor<T, B> operator+(const Tensor<T, B>& other) const {
        // Case 1: Shapes match exactly
        if (this->shape_ == other.shape_) {
            Tensor<T, B> result(this->shape_);
            size_t n = this->data_.size();
            DREIDEL_SIMD_LOOP
            for (size_t i = 0; i < n; ++i) {
                result.data_[i] = this->data_[i] + other.data_[i];
            }
            return result;
        }

        // Case 2: Broadcasting (Vector to Matrix)
        // Assume 'this' is (M, N) and 'other' is (N) or (1, N)
        if (this->shape_.size() == 2) {
            size_t M = this->shape_[0];
            size_t N = this->shape_[1];
            size_t other_sz = other.data_.size();

            if (other_sz == N) {
                Tensor<T, B> result(this->shape_);
                for (size_t i = 0; i < M; ++i) {
                    DREIDEL_SIMD_LOOP
                    for (size_t j = 0; j < N; ++j) {
                        result.data_[i * N + j] = this->data_[i * N + j] + other.data_[j];
                    }
                }
                return result;
            }
        }

        throw std::invalid_argument("Shapes incompatible for addition.");
    }

    Tensor<T, B> operator-(const Tensor<T, B>& other) const {
        if (this->shape_ != other.shape_) {
             throw std::invalid_argument("Shapes must match for subtraction.");
        }
        Tensor<T, B> result(this->shape_);
        size_t n = this->data_.size();

        DREIDEL_SIMD_LOOP
        for (size_t i = 0; i < n; ++i) {
            result.data_[i] = this->data_[i] - other.data_[i];
        }
        return result;
    }

    Tensor<T, B> operator*(const Tensor<T, B>& other) const {
        // Case 1: Shapes match exactly
        if (this->shape_ == other.shape_) {
            Tensor<T, B> result(this->shape_);
            size_t n = this->data_.size();
            DREIDEL_SIMD_LOOP
            for (size_t i = 0; i < n; ++i) {
                result.data_[i] = this->data_[i] * other.data_[i];
            }
            return result;
        }

        // Case 2: Generalized Broadcasting (Last dimension match)
        // If 'other' matches the last dimension of 'this', and 'other' is effectively 1D (or 1xN)
        if (!this->shape_.empty() && !other.shape_.empty()) {
            size_t last_dim = this->shape_.back();
            size_t other_sz = other.data_.size();

            // Check if other is broadcastable (size equals last_dim)
            if (other_sz == last_dim) {
                 Tensor<T, B> result(this->shape_);
                 size_t total_elements = this->data_.size();
                 size_t outer_dims = total_elements / last_dim;

                 // Parallelize over outer dimensions
                 DREIDEL_PARALLEL_LOOP
                 for (long i = 0; i < (long)outer_dims; ++i) {
                     size_t offset = i * last_dim;
                     DREIDEL_SIMD_LOOP
                     for (size_t j = 0; j < last_dim; ++j) {
                         result.data_[offset + j] = this->data_[offset + j] * other.data_[j];
                     }
                 }
                 return result;
            }
        }

        throw std::invalid_argument("Shapes incompatible for element-wise multiplication.");
    }

    // Scalar Multiplication
    Tensor<T, B> operator*(T scalar) const {
        Tensor<T, B> result(this->shape_);
        size_t n = this->data_.size();
        DREIDEL_SIMD_LOOP
        for(size_t i=0; i<n; ++i) {
            result.data_[i] = this->data_[i] * scalar;
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

    Tensor<T, B> transpose() const {
        if (this->shape_.size() != 2) {
             throw std::invalid_argument("Transpose only supports 2D tensors for now.");
        }
        size_t R = this->shape_[0];
        size_t C = this->shape_[1];
        Tensor<T, B> result({C, R});

        for (size_t i = 0; i < R; ++i) {
            for (size_t j = 0; j < C; ++j) {
                result.data_[j * R + i] = this->data_[i * C + j];
            }
        }
        return result;
    }

    // Sum over axis. 0 = sum cols (reduce rows), 1 = sum rows (reduce cols)
    Tensor<T, B> sum(int axis) const {
         if (this->shape_.size() != 2) throw std::runtime_error("Sum only 2D");
         size_t R = this->shape_[0];
         size_t C = this->shape_[1];

         if (axis == 0) {
             // Sum down the rows, result is (1, C)
             Tensor<T, B> result({1, C});
             result.fill(0);
             for(size_t i=0; i<R; ++i) {
                 for(size_t j=0; j<C; ++j) {
                     result.data_[j] += this->data_[i*C + j];
                 }
             }
             return result;
         } else if (axis == 1) {
             // Sum across columns, result is (R, 1)
             Tensor<T, B> result({R, 1});
             result.fill(0);
             for(size_t i=0; i<R; ++i) {
                 for(size_t j=0; j<C; ++j) {
                     result.data_[i * 1 + 0] += this->data_[i*C + j];
                 }
             }
             return result;
         }
         throw std::invalid_argument("Invalid axis");
    }

    template <typename Func>
    Tensor<T, B> apply(Func func) const {
        Tensor<T, B> result(this->shape_);
        size_t n = this->data_.size();
        for (size_t i = 0; i < n; ++i) {
            result.data_[i] = func(this->data_[i]);
        }
        return result;
    }

private:
    std::vector<size_t> shape_;
    std::vector<T, core::AlignedAllocator<T>> data_;
};

} // namespace dreidel

#endif // DREIDEL_CORE_TENSOR_HPP
