#ifndef DREIDEL_CORE_TENSOR_HPP
#define DREIDEL_CORE_TENSOR_HPP

#include <vector>
#include <initializer_list>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
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
    Tensor(const std::vector<size_t>& shape) : shape_(shape), owns_memory_(true) {
        size_ = 1;
        for (size_t s : shape) size_ *= s;
        if (posix_memalign((void**)&data_ptr_, 32, size_ * sizeof(T)) != 0) throw std::bad_alloc();
    }

    // Constructor with shape and external memory (Arena View)
    Tensor(const std::vector<size_t>& shape, T* data_ptr) : shape_(shape), data_ptr_(data_ptr), owns_memory_(false) {
        size_ = 1;
        for (size_t s : shape) size_ *= s;
    }

    // Constructor with shape and initial data
    Tensor(const std::vector<size_t>& shape, const std::vector<T>& data) : shape_(shape), owns_memory_(true) {
        size_ = 1;
        for (size_t s : shape) size_ *= s;
        if (data.size() != size_) {
             throw std::invalid_argument("Data size does not match shape dimensions.");
        }
        if (posix_memalign((void**)&data_ptr_, 32, size_ * sizeof(T)) != 0) throw std::bad_alloc();
        std::memcpy(data_ptr_, data.data(), size_ * sizeof(T));
    }

    // Copy Constructor (Deep Copy)
    Tensor(const Tensor& other) : shape_(other.shape_), size_(other.size_), owns_memory_(true) {
         if (posix_memalign((void**)&data_ptr_, 32, size_ * sizeof(T)) != 0) throw std::bad_alloc();
         std::memcpy(data_ptr_, other.data_ptr_, size_ * sizeof(T));
    }

    // Move Constructor
    Tensor(Tensor&& other) noexcept : shape_(std::move(other.shape_)), data_ptr_(other.data_ptr_), size_(other.size_), owns_memory_(other.owns_memory_) {
        other.data_ptr_ = nullptr;
        other.size_ = 0;
        other.owns_memory_ = false;
    }

    // Assignment Operator (Copy)
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (owns_memory_ && data_ptr_) free(data_ptr_);
            shape_ = other.shape_;
            size_ = other.size_;
            owns_memory_ = true;
            if (posix_memalign((void**)&data_ptr_, 32, size_ * sizeof(T)) != 0) throw std::bad_alloc();
            std::memcpy(data_ptr_, other.data_ptr_, size_ * sizeof(T));
        }
        return *this;
    }

    // Move Assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (owns_memory_ && data_ptr_) free(data_ptr_);
            shape_ = std::move(other.shape_);
            data_ptr_ = other.data_ptr_;
            size_ = other.size_;
            owns_memory_ = other.owns_memory_;
            other.data_ptr_ = nullptr;
            other.size_ = 0;
            other.owns_memory_ = false;
        }
        return *this;
    }

    virtual ~Tensor() {
        if (owns_memory_ && data_ptr_) free(data_ptr_);
    }

    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    T* data() { return data_ptr_; }
    const T* data() const { return data_ptr_; }

    // Deep copy from another tensor
    void copy_from(const Tensor<T, B>& other) {
        if (size_ != other.size()) {
             throw std::invalid_argument("Tensor size mismatch in copy_from");
        }
        std::memcpy(data_ptr_, other.data(), size_ * sizeof(T));
    }

    // Element access (simple flat index for now)
    T& operator[](size_t index) { return data_ptr_[index]; }
    const T& operator[](size_t index) const { return data_ptr_[index]; }

    void fill(T value) {
        if (data_ptr_) std::fill(data_ptr_, data_ptr_ + size_, value);
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
        // static std::random_device rd;
        // static std::mt19937 gen(rd());
        static std::mt19937 gen(42); // Fixed seed to avoid random_device issues and ensure reproducibility
        std::normal_distribution<T> d(mean, stddev);
        for(size_t i=0; i<size_; ++i) data_ptr_[i] = d(gen);
    }

    // Element-wise operations
    Tensor<T, B> operator+(const Tensor<T, B>& other) const {
        // Case 1: Shapes match exactly
        if (this->shape_ == other.shape_) {
            Tensor<T, B> result(this->shape_);
            size_t n = this->size_;
            DREIDEL_SIMD_LOOP
            for (size_t i = 0; i < n; ++i) {
                result[i] = this->data_ptr_[i] + other.data_ptr_[i];
            }
            return result;
        }

        // Case 2: Broadcasting (Vector to Matrix)
        // Assume 'this' is (M, N) and 'other' is (N) or (1, N)
        if (this->shape_.size() == 2) {
            size_t M = this->shape_[0];
            size_t N = this->shape_[1];
            size_t other_sz = other.size_;

            if (other_sz == N) {
                Tensor<T, B> result(this->shape_);
                for (size_t i = 0; i < M; ++i) {
                    DREIDEL_SIMD_LOOP
                    for (size_t j = 0; j < N; ++j) {
                        result[i * N + j] = this->data_ptr_[i * N + j] + other.data_ptr_[j];
                    }
                }
                return result;
            }
        }

        // Case 3: Generalized Broadcasting (Last dimension match)
        if (!this->shape_.empty() && !other.shape_.empty()) {
            size_t last_dim = this->shape_.back();
            size_t other_sz = other.size_;

            // Check if other is broadcastable (size equals last_dim)
            if (other_sz == last_dim) {
                 Tensor<T, B> result(this->shape_);
                 size_t total_elements = this->size_;
                 size_t outer_dims = total_elements / last_dim;

                 // Parallelize over outer dimensions
                 DREIDEL_PARALLEL_LOOP
                 for (long i = 0; i < (long)outer_dims; ++i) {
                     size_t offset = i * last_dim;
                     DREIDEL_SIMD_LOOP
                     for (size_t j = 0; j < last_dim; ++j) {
                         result[offset + j] = this->data_ptr_[offset + j] + other.data_ptr_[j];
                     }
                 }
                 return result;
            }
        }

        // Case 4: Reverse Broadcasting for + (Input + Bias where Bias is 1D)
        // If 'this' is larger and 'other' matches last dim
        // Wait, Case 3 handles (Large + Small).
        // What if we have (Small + Large)? e.g. Bias + Input?
        // Bias (1D) + Input (ND).
        if (!this->shape_.empty() && !other.shape_.empty()) {
            size_t last_dim = other.shape_.back();
            size_t this_sz = this->size_;

            if (this_sz == last_dim) {
                 Tensor<T, B> result(other.shape_);
                 size_t total_elements = other.size_;
                 size_t outer_dims = total_elements / last_dim;

                 DREIDEL_PARALLEL_LOOP
                 for (long i = 0; i < (long)outer_dims; ++i) {
                     size_t offset = i * last_dim;
                     DREIDEL_SIMD_LOOP
                     for (size_t j = 0; j < last_dim; ++j) {
                         result[offset + j] = this->data_ptr_[j] + other.data_ptr_[offset + j];
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
        size_t n = this->size_;

        DREIDEL_SIMD_LOOP
        for (size_t i = 0; i < n; ++i) {
            result[i] = this->data_ptr_[i] - other.data_ptr_[i];
        }
        return result;
    }

    Tensor<T, B> operator*(const Tensor<T, B>& other) const {
        // Case 1: Shapes match exactly
        if (this->shape_ == other.shape_) {
            Tensor<T, B> result(this->shape_);
            size_t n = this->size_;
            DREIDEL_SIMD_LOOP
            for (size_t i = 0; i < n; ++i) {
                result[i] = this->data_ptr_[i] * other.data_ptr_[i];
            }
            return result;
        }

        // Case 2: Generalized Broadcasting (Last dimension match)
        // If 'other' matches the last dimension of 'this', and 'other' is effectively 1D (or 1xN)
        if (!this->shape_.empty() && !other.shape_.empty()) {
            size_t last_dim = this->shape_.back();
            size_t other_sz = other.size_;

            // Check if other is broadcastable (size equals last_dim)
            if (other_sz == last_dim) {
                 Tensor<T, B> result(this->shape_);
                 size_t total_elements = this->size_;
                 size_t outer_dims = total_elements / last_dim;

                 // Parallelize over outer dimensions
                 DREIDEL_PARALLEL_LOOP
                 for (long i = 0; i < (long)outer_dims; ++i) {
                     size_t offset = i * last_dim;
                     DREIDEL_SIMD_LOOP
                     for (size_t j = 0; j < last_dim; ++j) {
                         result[offset + j] = this->data_ptr_[offset + j] * other.data_ptr_[j];
                     }
                 }
                 return result;
            }
        }

        // Case 3: Reverse Broadcasting (this is scale, other is input)
        // If 'this' is 1D and matches last dim of 'other'
        if (!this->shape_.empty() && !other.shape_.empty()) {
            size_t last_dim = other.shape_.back();
            size_t this_sz = this->size_;

            if (this_sz == last_dim) {
                 Tensor<T, B> result(other.shape_);
                 size_t total_elements = other.size_;
                 size_t outer_dims = total_elements / last_dim;

                 DREIDEL_PARALLEL_LOOP
                 for (long i = 0; i < (long)outer_dims; ++i) {
                     size_t offset = i * last_dim;
                     DREIDEL_SIMD_LOOP
                     for (size_t j = 0; j < last_dim; ++j) {
                         result[offset + j] = other.data_ptr_[offset + j] * this->data_ptr_[j];
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
        size_t n = this->size_;
        DREIDEL_SIMD_LOOP
        for(size_t i=0; i<n; ++i) {
            result[i] = this->data_ptr_[i] * scalar;
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
                T a_val = this->data_ptr_[i * K + k];

                // Hint to compiler that these pointers are restrict-like (no aliasing) ideally,
                // but just SIMD pragma helps.
                DREIDEL_SIMD_LOOP
                for (size_t j = 0; j < N; ++j) {
                    result[i * N + j] += a_val * other.data_ptr_[k * N + j];
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
                result[j * R + i] = this->data_ptr_[i * C + j];
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
                     result[j] += this->data_ptr_[i*C + j];
                 }
             }
             return result;
         } else if (axis == 1) {
             // Sum across columns, result is (R, 1)
             Tensor<T, B> result({R, 1});
             result.fill(0);
             for(size_t i=0; i<R; ++i) {
                 for(size_t j=0; j<C; ++j) {
                     result[i * 1 + 0] += this->data_ptr_[i*C + j];
                 }
             }
             return result;
         }
         throw std::invalid_argument("Invalid axis");
    }

    template <typename Func>
    Tensor<T, B> apply(Func func) const {
        Tensor<T, B> result(this->shape_);
        size_t n = this->size_;
        for (size_t i = 0; i < n; ++i) {
            result[i] = func(this->data_ptr_[i]);
        }
        return result;
    }

    Tensor<T, B> pad_last_dim(size_t new_dim) const {
        if (shape_.empty()) throw std::invalid_argument("Cannot pad scalar");
        size_t last_dim = shape_.back();
        if (new_dim < last_dim) throw std::invalid_argument("New dimension must be >= current dimension");
        if (new_dim == last_dim) return *this;

        std::vector<size_t> new_shape = shape_;
        new_shape.back() = new_dim;

        Tensor<T, B> result(new_shape);
        result.fill(0); // Initialize with zeros

        size_t total_elements = this->size_;
        size_t outer_dims = total_elements / last_dim;

        DREIDEL_PARALLEL_LOOP
        for (long i = 0; i < (long)outer_dims; ++i) {
            size_t src_offset = i * last_dim;
            size_t dst_offset = i * new_dim;
            for (size_t j = 0; j < last_dim; ++j) {
                result[dst_offset + j] = this->data_ptr_[src_offset + j];
            }
        }
        return result;
    }

    Tensor<T, B> slice_last_dim(size_t new_dim) const {
        if (shape_.empty()) throw std::invalid_argument("Cannot slice scalar");
        size_t last_dim = shape_.back();
        if (new_dim > last_dim) throw std::invalid_argument("New dimension must be <= current dimension");
        if (new_dim == last_dim) return *this;

        std::vector<size_t> new_shape = shape_;
        new_shape.back() = new_dim;

        Tensor<T, B> result(new_shape);

        size_t total_elements = result.size(); // Use result size
        size_t outer_dims = total_elements / new_dim;

        DREIDEL_PARALLEL_LOOP
        for (long i = 0; i < (long)outer_dims; ++i) {
            size_t src_offset = i * last_dim;
            size_t dst_offset = i * new_dim;
            for (size_t j = 0; j < new_dim; ++j) {
                result[dst_offset + j] = this->data_ptr_[src_offset + j];
            }
        }
        return result;
    }

private:
    std::vector<size_t> shape_;
    T* data_ptr_ = nullptr;
    size_t size_ = 0;
    bool owns_memory_ = true;
};

} // namespace dreidel

#endif // DREIDEL_CORE_TENSOR_HPP
