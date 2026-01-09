#pragma once

#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <vector>

namespace dreidel {
namespace optim {

// Helper: Horizontal sum of __m256 to get a single float (AVX2)
inline float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums        = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Applies Group Lasso Proximal Operator
// Ensures "True Zeros" for blocks with energy < threshold
inline void apply_group_lasso_avx(float* weights, size_t total_floats, float lambda, float lr) {
    // Threshold to shrink by (The "Tax")
    float threshold = lambda * lr;

    // Iterate in blocks of 8 (Zenith Block Size)
    // Assuming total_floats is a multiple of 8, or handle tail if necessary.
    // Zenith spectral mixer weights are usually aligned and multiples of 8 (channels power of 2).
    for (size_t i = 0; i < total_floats; i += 8) {
        // Safety check for tail (though Zenith blocks should be aligned)
        if (i + 8 > total_floats) break;

        // 1. Load Block
        __m256 w = _mm256_loadu_ps(&weights[i]); // Use loadu for safety, though likely aligned

        // 2. Calculate L2 Norm of the Block
        __m256 w2 = _mm256_mul_ps(w, w);
        float norm_scalar = sqrtf(hsum256_ps_avx(w2));

        // 3. Calculate Shrinkage Factor: max(0, 1 - (Threshold / Norm))
        // We use scalar math for the factor since it's shared for the whole block
        float ratio = threshold / (norm_scalar + 1e-9f);
        float factor = std::max(0.0f, 1.0f - ratio);

        // 4. Apply Scale (If factor is 0, block becomes 0.00000)
        __m256 v_factor = _mm256_set1_ps(factor);
        __m256 w_new = _mm256_mul_ps(w, v_factor);

        // 5. Store
        _mm256_storeu_ps(&weights[i], w_new);
    }
}

} // namespace optim
} // namespace dreidel
