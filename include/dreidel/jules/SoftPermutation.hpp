#ifndef DREIDEL_JULES_SOFTPERMUTATION_HPP
#define DREIDEL_JULES_SOFTPERMUTATION_HPP

#include "../core/TensorView.hpp"
#include <cmath>

namespace dreidel {
namespace jules {

/**
 * @brief Local Soft Permutation (Block Mixing).
 *
 * Mixes adjacent elements in the feature dimension using 2x2 rotations.
 * [ x_2i   ]   [ cos t   -sin t ] [ x_2i   ]
 * [ x_2i+1 ] = [ sin t    cos t ] [ x_2i+1 ]
 *
 * This adds local interaction after the global mixing of FWHT.
 */
template <size_t DIM>
class SoftPermutation {
public:
    // Angles for each pair
    float thetas[DIM / 2];

    // Precomputed sin/cos to save runtime math?
    // User requested "APoT" for weights, but this is a structural permutation.
    // Let's store precomputed coeffs.
    float cos_t[DIM / 2];
    float sin_t[DIM / 2];

    SoftPermutation() {
        // Init with random rotations
        for(size_t i=0; i<DIM/2; ++i) {
            float t = (float)(i % 100) * 0.01f; // Mock random
            thetas[i] = t;
            cos_t[i] = std::cos(t);
            sin_t[i] = std::sin(t);
        }
    }

    void forward(core::TensorView<float>& input, core::TensorView<float>& output) {
        // Can be in-place
        float* in_ptr = input.data();
        float* out_ptr = output.data();

        // Loop over pairs
        // #pragma omp simd
        for (size_t i = 0; i < DIM / 2; ++i) {
            float x0 = in_ptr[2*i];
            float x1 = in_ptr[2*i+1];

            float c = cos_t[i];
            float s = sin_t[i];

            out_ptr[2*i]     = c * x0 - s * x1;
            out_ptr[2*i+1]   = s * x0 + c * x1;
        }
    }
};

} // namespace jules
} // namespace dreidel

#endif // DREIDEL_JULES_SOFTPERMUTATION_HPP
