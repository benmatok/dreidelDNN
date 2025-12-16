#ifndef DREIDEL_OPTIM_BLOCK_DIAGONAL_HPP
#define DREIDEL_OPTIM_BLOCK_DIAGONAL_HPP

#include "Optimizer.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace dreidel {
namespace optim {

// Simple Gauss-Jordan elimination for matrix inversion
// A: n x n matrix stored in row-major order
// n: dimension
// Returns true if successful, false if singular
template <typename T>
bool invert_matrix(std::vector<T>& A, int n) {
    // Augment with Identity
    // We will do in-place if possible, but for simplicity let's use a temporary buffer for Identity part
    // Actually, typical implementation works on [A | I].

    std::vector<T> AI(n * 2 * n);
    for(int i=0; i<n; ++i) {
        for(int j=0; j<n; ++j) {
            AI[i * (2*n) + j] = A[i*n + j];
        }
        for(int j=0; j<n; ++j) {
            AI[i * (2*n) + n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan
    for (int i = 0; i < n; ++i) {
        // Pivot
        T pivot = AI[i * (2*n) + i];
        if (std::abs(pivot) < 1e-10) return false; // Singular

        // Normalize row i
        for (int j = 0; j < 2*n; ++j) {
            AI[i * (2*n) + j] /= pivot;
        }

        // Eliminate other rows
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                T factor = AI[k * (2*n) + i];
                for (int j = 0; j < 2*n; ++j) {
                    AI[k * (2*n) + j] -= factor * AI[i * (2*n) + j];
                }
            }
        }
    }

    // Extract inverse
    for(int i=0; i<n; ++i) {
        for(int j=0; j<n; ++j) {
            A[i*n + j] = AI[i * (2*n) + n + j];
        }
    }
    return true;
}

template <typename T, BackendType B = BackendType::CPU>
class BlockDiagonal : public Optimizer<T, B> {
public:
    BlockDiagonal(T learning_rate, int block_size)
        : learning_rate_(learning_rate), block_size_(block_size) {}

    void add_parameters(std::vector<Tensor<T, B>*> params,
                        std::vector<Tensor<T, B>*> grads,
                        std::vector<Tensor<T, B>*> curvs) {
        if (params.size() != grads.size()) {
             throw std::invalid_argument("Params and grads size mismatch");
        }
        for(auto* p : params) parameters_.push_back(p);
        for(auto* g : grads) gradients_.push_back(g);

        // Curvatures here are expected to be list of blocks (flattened or stacked)
        if (curvs.size() == params.size()) {
             for(auto* c : curvs) curvatures_.push_back(c);
        } else {
             // If mismatch or empty, push nulls
             for(size_t i=0; i<params.size(); ++i) curvatures_.push_back(nullptr);
        }
    }

    void step() override {
        // Parallel over parameters
        for (size_t i = 0; i < parameters_.size(); ++i) {
            Tensor<T, B>* param = parameters_[i];
            Tensor<T, B>* grad = gradients_[i];
            Tensor<T, B>* curv = curvatures_[i]; // Expected to contain Block approximations (Hessian blocks)

            if (!curv) {
                // Fallback SGD
                T* p = param->data();
                T* g = grad->data();
                size_t sz = param->size();
                DREIDEL_SIMD_LOOP
                for(size_t j=0; j<sz; ++j) p[j] -= learning_rate_ * g[j];
                continue;
            }

            // Block Logic
            // Assume 'curv' contains K matrices of size block_size_ x block_size_.
            // And 'param'/'grad' are effectively K vectors of size block_size_.

            // Example: Conv3D weights (K, K, K, C).
            // If we treat each channel as independent, block size = K*K*K = 27.
            // curv would be (C, 27, 27).

            // We iterate over blocks.
            size_t total_elements = param->size();
            size_t dim = block_size_;
            size_t num_blocks = total_elements / dim;

            if (total_elements % dim != 0) {
                 // Mismatch, fallback SGD
                 // Or throw. Let's fallback.
                 // std::cerr << "Block size mismatch" << std::endl;
                 continue;
            }

            T* p_data = param->data();
            T* g_data = grad->data();
            T* c_data = curv->data();

            #pragma omp parallel for
            for (long b = 0; b < (long)num_blocks; ++b) {
                // Block b
                // Gradient vector g_b (dim)
                // Hessian block H_b (dim x dim)

                // Copy H_b to local buffer for inversion
                std::vector<T> H_local(dim * dim);
                for(size_t k=0; k<dim*dim; ++k) {
                    H_local[k] = c_data[b * dim * dim + k];
                }

                // Add damping?
                for(size_t k=0; k<dim; ++k) H_local[k*dim+k] += 1e-5;

                // Invert
                bool success = invert_matrix(H_local, dim);

                if (success) {
                    // Update p_b = p_b - lr * H_inv * g_b
                    for(size_t r=0; r<dim; ++r) {
                        T sum = 0;
                        for(size_t c=0; c<dim; ++c) {
                            sum += H_local[r*dim + c] * g_data[b*dim + c];
                        }
                        p_data[b*dim + r] -= learning_rate_ * sum;
                    }
                } else {
                    // Singular, fallback to SGD for this block
                    for(size_t r=0; r<dim; ++r) {
                        p_data[b*dim + r] -= learning_rate_ * g_data[b*dim + r];
                    }
                }
            }
        }
    }

    void zero_grad() override {
        for (auto* grad : gradients_) grad->fill(0);
        for (auto* curv : curvatures_) if(curv) curv->fill(0);
    }

private:
    T learning_rate_;
    int block_size_;
    std::vector<Tensor<T, B>*> parameters_;
    std::vector<Tensor<T, B>*> gradients_;
    std::vector<Tensor<T, B>*> curvatures_;
};

} // namespace optim
} // namespace dreidel

#endif // DREIDEL_OPTIM_BLOCK_DIAGONAL_HPP
