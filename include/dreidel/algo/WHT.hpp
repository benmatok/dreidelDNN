#ifndef DREIDEL_ALGO_WHT_HPP
#define DREIDEL_ALGO_WHT_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include "../core/Tensor.hpp"
#include "../hal/ops.hpp"

// Check for OpenMP
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace dreidel {
namespace algo {

class WHT {
public:
    // Fast Walsh-Hadamard Transform (Iterative, In-Place)
    // Applies WHT on the last dimension of the tensor.
    // The last dimension size must be a power of 2.
    template <typename T, BackendType B>
    static void FWHT(Tensor<T, B>& tensor) {
        // Enforce last dimension is power of 2
        const auto& shape = tensor.shape();
        if (shape.empty()) return;

        size_t N = shape.back();
        if (N == 0 || (N & (N - 1)) != 0) {
            // Error handling or just return for now?
            // In C++ typically throw.
            throw std::invalid_argument("Last dimension must be power of 2 for WHT.");
        }

        size_t total_elements = tensor.size();
        size_t outer_dims = total_elements / N;

        // Use HAL ops
        using Ops = hal::ActiveOps;

        // Check for OpenGL Backend
        if constexpr (B == BackendType::OpenGL) {
            // For OpenGL, we assume the data is either already on device or we orchestrate it.
            // Since Tensor<T, OpenGL> in this refactor still holds std::vector (host memory) for now (based on Tensor.hpp),
            // we treat this as "Dispatch to GL".
            // However, the HAL stub throws error currently.
            // We iterate over outer dims on CPU and dispatch GL for each vector?
            // Or better: OpenGL backend usually handles large batches.
            // If the tensor is (M, N), we can dispatch (M * N/2) threads.

            // For now, call the stub.
            try {
                hal::opengl::Ops::fwht_dispatch(tensor.data(), total_elements);
                // Note: The shader logic provided in opengl.hpp assumes 1D N.
                // Handling batches requires shader modification or multiple dispatches.
                // Given the constraints, we will defer full GL batch implementation.
                return;
            } catch (const std::exception& e) {
                 // Fallback to CPU if GL fails or just rethrow?
                 // Usually rethrow.
                 throw;
            }
        }

        // CPU / SIMD Implementation

        // Iterate over all "vectors" in the last dimension
        // Parallelize over the outer dimensions
        #pragma omp parallel for
        for (long i = 0; i < (long)outer_dims; ++i) {
            T* data_ptr = tensor.data() + i * N;

            // Perform WHT on data_ptr of length N
            // Standard iterative Cooley-Tukey like butterfly
            for (size_t len = 1; len < N; len <<= 1) {
                // len is half the block size
                // block size is 2 * len

                for (size_t j = 0; j < N; j += 2 * len) {
                    // This inner loop can be vectorized
                    // We need to process 'len' butterflies
                    // k from 0 to len-1:
                    //   u = x[j + k]
                    //   v = x[j + k + len]
                    //   x[j + k] = u + v
                    //   x[j + k + len] = u - v

                    size_t k = 0;

                    // SIMD Loop
                    // Ops::SIMD_WIDTH is number of T elements
                    // Only use SIMD if T is float (Ops are hardcoded for float/generic)
                    // If T is double, we might need template specialization in HAL,
                    // currently HAL is float only or generic.

                    if constexpr (std::is_same_v<T, float>) {
                        for (; k + Ops::SIMD_WIDTH <= len; k += Ops::SIMD_WIDTH) {
                            auto u_vec = Ops::load(data_ptr + j + k);
                            auto v_vec = Ops::load(data_ptr + j + k + len);

                            Ops::butterfly(u_vec, v_vec);

                            Ops::store(data_ptr + j + k, u_vec);
                            Ops::store(data_ptr + j + k + len, v_vec);
                        }
                    }

                    // Scalar cleanup / Generic loop
                    for (; k < len; ++k) {
                        T u = data_ptr[j + k];
                        T v = data_ptr[j + k + len];
                        data_ptr[j + k] = u + v;
                        data_ptr[j + k + len] = u - v;
                    }
                }
            }
        }
    }

    // Inverse FWHT is just FWHT / N, but since FWHT is symmetric (unnormalized),
    // applying it twice gives N*x. So we apply FWHT then scale.
    // Or we can scale first.
    template <typename T, BackendType B>
    static void InverseFWHT(Tensor<T, B>& tensor) {
        FWHT(tensor);

        size_t N = tensor.shape().back();
        T scale = static_cast<T>(1.0) / static_cast<T>(N);

        // Scale in place
        // Can be parallelized and vectorized
        size_t total = tensor.size();
        T* data = tensor.data();

        #pragma omp parallel for
        for (long i = 0; i < (long)total; ++i) {
            data[i] *= scale;
        }
    }
};

} // namespace algo
} // namespace dreidel

#endif // DREIDEL_ALGO_WHT_HPP
