#ifndef DREIDEL_UTILS_MATRIXOPS_HPP
#define DREIDEL_UTILS_MATRIXOPS_HPP

#include <vector>
#include <cmath>
#include <iostream>

namespace dreidel {
namespace utils {

// Simple Gauss-Jordan elimination for matrix inversion
// A: n x n matrix stored in row-major order
// n: dimension
// Returns true if successful, false if singular
template <typename T>
bool invert_matrix(std::vector<T>& A, int n) {
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

// Arena-compatible version using raw pointers
template <typename T>
bool invert_matrix_ptr(T* A, int n, T* workspace) {
    // workspace needs to be size n*2*n
    T* AI = workspace;

    // Setup Augmented Matrix
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
        T pivot = AI[i * (2*n) + i];
        if (std::abs(pivot) < 1e-10) return false;

        for (int j = 0; j < 2*n; ++j) {
            AI[i * (2*n) + j] /= pivot;
        }

        for (int k = 0; k < n; ++k) {
            if (k != i) {
                T factor = AI[k * (2*n) + i];
                for (int j = 0; j < 2*n; ++j) {
                    AI[k * (2*n) + j] -= factor * AI[i * (2*n) + j];
                }
            }
        }
    }

    // Extract
    for(int i=0; i<n; ++i) {
        for(int j=0; j<n; ++j) {
            A[i*n + j] = AI[i * (2*n) + n + j];
        }
    }
    return true;
}

} // namespace utils
} // namespace dreidel

#endif // DREIDEL_UTILS_MATRIXOPS_HPP
