#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <immintrin.h>
#include "../include/dreidel/core/Allocator.hpp"

using namespace dreidel;

// Copy of the AVX2 logic from ZenithBlock.hpp
static inline void permute_avx2(float* out, const float* in, const int32_t* indices, size_t N) {
#ifdef __AVX2__
    size_t i = 0;
    // Unroll by 4 (8 floats * 4 = 32 floats per loop)
    for (; i + 32 <= N; i += 32) {
         // Gather 4 blocks of 8
         __m256i idx0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i));
         __m256i idx1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i + 8));
         __m256i idx2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i + 16));
         __m256i idx3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i + 24));

         __m256 r0 = _mm256_i32gather_ps(in, idx0, 4);
         __m256 r1 = _mm256_i32gather_ps(in, idx1, 4);
         __m256 r2 = _mm256_i32gather_ps(in, idx2, 4);
         __m256 r3 = _mm256_i32gather_ps(in, idx3, 4);

         _mm256_store_ps(out + i, r0);
         _mm256_store_ps(out + i + 8, r1);
         _mm256_store_ps(out + i + 16, r2);
         _mm256_store_ps(out + i + 24, r3);
    }
    // Fallback for remaining (still AVX for multiples of 8)
    for (; i + 8 <= N; i += 8) {
        __m256i idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices + i));
        __m256 r = _mm256_i32gather_ps(in, idx, 4);
        _mm256_store_ps(out + i, r);
    }
    // Scalar tail
    for (; i < N; ++i) {
        out[i] = in[indices[i]];
    }
#else
    // Fallback if compiled without AVX2 (should not happen in this test env)
    for (size_t i = 0; i < N; ++i) out[i] = in[indices[i]];
#endif
}

void test_permute_correctness() {
    const size_t N = 1024;

    // Data
    std::vector<float, core::AlignedAllocator<float>> in(N);
    std::vector<float, core::AlignedAllocator<float>> out_avx(N);
    std::vector<float, core::AlignedAllocator<float>> out_ref(N);

    // Indices (Map)
    std::vector<int32_t, core::AlignedAllocator<int32_t>> indices(N);

    // Initialize
    std::iota(in.begin(), in.end(), 0.0f); // 0.0, 1.0, 2.0...

    // Random permutation indices
    std::vector<int32_t> temp_indices(N);
    std::iota(temp_indices.begin(), temp_indices.end(), 0);
    std::mt19937 g(42);
    std::shuffle(temp_indices.begin(), temp_indices.end(), g);

    for(size_t i=0; i<N; ++i) indices[i] = temp_indices[i];

    // Reference (Scalar)
    for(size_t i=0; i<N; ++i) {
        out_ref[i] = in[indices[i]];
    }

    // AVX2
    permute_avx2(out_avx.data(), in.data(), indices.data(), N);

    // Compare
    float max_diff = 0.0f;
    for(size_t i=0; i<N; ++i) {
        float diff = std::abs(out_avx[i] - out_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": Ref=" << out_ref[i] << " AVX=" << out_avx[i] << std::endl;
            exit(1);
        }
    }

    std::cout << "AVX2 Permute Test Passed! Max diff: " << max_diff << std::endl;
}

int main() {
#ifndef __AVX2__
    std::cerr << "Test compiled without AVX2! Skipping." << std::endl;
    return 0;
#endif
    test_permute_correctness();
    return 0;
}
