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
    // Single Vector FWHT Kernel (No OpenMP, for Fused Loops)
    template <typename T>
    static void fwht_1d(T* data, size_t N) {
        using Ops = hal::ActiveOps;

        for (size_t len = 1; len < N; len <<= 1) {
            for (size_t j = 0; j < N; j += 2 * len) {
                size_t k = 0;
                if constexpr (std::is_same_v<T, float>) {
                    for (; k + Ops::SIMD_WIDTH <= len; k += Ops::SIMD_WIDTH) {
                        auto u_vec = Ops::load(data + j + k);
                        auto v_vec = Ops::load(data + j + k + len);
                        Ops::butterfly(u_vec, v_vec);
                        Ops::store(data + j + k, u_vec);
                        Ops::store(data + j + k + len, v_vec);
                    }
                }
                for (; k < len; ++k) {
                    T u = data[j + k];
                    T v = data[j + k + len];
                    data[j + k] = u + v;
                    data[j + k + len] = u - v;
                }
            }
        }
    }

    // Fast Walsh-Hadamard Transform (Iterative, In-Place)
    // Applies WHT on the last dimension of the tensor.
    // The last dimension size must be a power of 2.
    template <typename T, BackendType B>
    static void FWHT(Tensor<T, B>& tensor) {
        const auto& shape = tensor.shape();
        if (shape.empty()) return;

        size_t N = shape.back();
        if (N == 0 || (N & (N - 1)) != 0) {
            throw std::invalid_argument("Last dimension must be power of 2 for WHT.");
        }

        size_t total_elements = tensor.size();
        size_t outer_dims = total_elements / N;

        if constexpr (B == BackendType::OpenGL) {
            try {
                hal::opengl::Ops::fwht_dispatch(tensor.data(), total_elements);
                return;
            } catch (const std::exception& e) {
                 throw;
            }
        }

        #pragma omp parallel for
        for (long i = 0; i < (long)outer_dims; ++i) {
            T* data_ptr = tensor.data() + i * N;
            fwht_1d(data_ptr, N);
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
