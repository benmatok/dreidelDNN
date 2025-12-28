#ifndef DREIDEL_ALGO_WHT_HPP
#define DREIDEL_ALGO_WHT_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include "../core/Tensor.hpp"
#include "../core/TensorView.hpp"
#include "../core/Packet.hpp" // Use Packet Abstraction

// Check for OpenMP
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace dreidel {
namespace algo {

class WHT {
public:
    // Single Vector FWHT Kernel (Iterative, No Recursion)
    // Optimized for CPU using standard radix-2 iteration
    template <typename T>
    static void fwht_1d(T* data, size_t N) {
        using Packet = core::Packet<T>;

        // Iterative Cooley-Tukey-like butterfly
        // h starts at 1, doubles each time until N
        for (size_t h = 1; h < N; h <<= 1) {

            // For each block of size 2*h
            for (size_t i = 0; i < N; i += (h << 1)) {

                size_t j = i;
                size_t end_j = i + h;

                // Vectorized Loop using Packet Abstraction
                if constexpr (std::is_same_v<T, float>) {
                    size_t k = 0;
                    // Loop while k + SIMD_WIDTH <= h
                    for (; k + Packet::width <= h; k += Packet::width) {
                        // Load x and y via Packet
                        Packet x(data + j + k);
                        Packet y(data + j + k + h);

                        // Butterfly: x' = x+y, y' = x-y
                        Packet::butterfly(x, y);

                        // Store back
                        x.store(data + j + k);
                        y.store(data + j + k + h);
                    }

                    // Update scalar pointers to continue where SIMD left off
                    j += k;
                }

                // Scalar Cleanup Loop
                for (; j < end_j; ++j) {
                    T x = data[j];
                    T y = data[j + h];

                    data[j] = x + y;
                    data[j + h] = x - y;
                }
            }
        }
    }

    // Fast Walsh-Hadamard Transform (Iterative, In-Place)
    // Applies WHT on the last dimension of the tensor view.
    // The last dimension size must be a power of 2.
    template <typename T>
    static void FWHT(core::TensorView<T>& tensor) {
        const auto& shape = tensor.shape();
        if (shape.empty()) return; // Scalar has no WHT

        size_t N = tensor.dim(tensor.rank() - 1); // Last dim
        if (N == 0 || (N & (N - 1)) != 0) {
             throw std::invalid_argument("Last dimension must be power of 2 for WHT.");
        }

        size_t total_elements = tensor.size();
        size_t outer_dims = total_elements / N;

        // Parallelize over independent vectors
        #pragma omp parallel for if(total_elements > 16384)
        for (long i = 0; i < (long)outer_dims; ++i) {
            T* data_ptr = tensor.data() + i * N;
            fwht_1d(data_ptr, N);
        }
    }

    // Keep legacy support for owning Tensor
    template <typename T, BackendType B>
    static void FWHT(Tensor<T, B>& tensor) {
        // Reuse the logic via a temporary view
        core::TensorView<T> view(tensor.data(), tensor.shape());
        FWHT(view);
    }

    // Inverse FWHT
    template <typename T>
    static void InverseFWHT(core::TensorView<T>& tensor) {
        FWHT(tensor);

        size_t N = tensor.dim(tensor.rank() - 1);
        T scale = static_cast<T>(1.0) / static_cast<T>(N);

        size_t total = tensor.size();
        T* data = tensor.data();

        #pragma omp parallel for
        for (long i = 0; i < (long)total; ++i) {
            data[i] *= scale;
        }
    }

    // Legacy Inverse Support
    template <typename T, BackendType B>
    static void InverseFWHT(Tensor<T, B>& tensor) {
        core::TensorView<T> view(tensor.data(), tensor.shape());
        InverseFWHT(view);
    }
};

} // namespace algo
} // namespace dreidel

#endif // DREIDEL_ALGO_WHT_HPP
