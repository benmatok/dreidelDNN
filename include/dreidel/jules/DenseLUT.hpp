#ifndef DREIDEL_JULES_DENSELUT_HPP
#define DREIDEL_JULES_DENSELUT_HPP

#include "../core/TensorView.hpp"
#include "../quant/APoT.hpp"
#include <vector>

namespace dreidel {
namespace jules {

/**
 * @brief Dense Layer using APoT/LUT quantization.
 *
 * Logic:
 * Output = Input * Weights
 * Weights are stored as uint8_t codes.
 *
 * Uses shared APoT::product_lut for zero-math multiplication.
 */
template <size_t INPUT_DIM, size_t OUTPUT_DIM>
class DenseLUT {
public:
    // Weights in quantized format
    uint8_t weights[INPUT_DIM * OUTPUT_DIM];

    DenseLUT() {
        // Ensure global LUT is init
        // In strictly static env, this might be called explicitly in main()
        // but checking here is safe for now.
        if (!quant::APoT::lut_initialized) {
            quant::APoT::init();
        }

        // Mock weights
        for(size_t i=0; i<INPUT_DIM * OUTPUT_DIM; ++i) weights[i] = (i % 256);
    }

    // For manual init call
    static void init_lut() {
        quant::APoT::init();
    }

    void forward(core::TensorView<float>& input, core::TensorView<float>& output) {
        output.fill(0.0f);
        float* out_ptr = output.data();
        const float* in_ptr = input.data();

        // Loop: For each input element
        for (size_t i = 0; i < INPUT_DIM; ++i) {
            // Quantize Input on the fly
            uint8_t q_in = quant::APoT::quantize(in_ptr[i]);

            // Row of product lookups
            const float* lut_row = quant::APoT::product_lut[q_in];
            const uint8_t* w_row = weights + i * OUTPUT_DIM;

            // Accumulate
            // No Multiplications! Just Adds and Lookups.
            // This loop is very amenable to unrolling but maybe not SIMD
            // unless we have gather support for 8-bit indices.
            for (size_t j = 0; j < OUTPUT_DIM; ++j) {
                out_ptr[j] += lut_row[w_row[j]];
            }
        }
    }
};

} // namespace jules
} // namespace dreidel

#endif // DREIDEL_JULES_DENSELUT_HPP
