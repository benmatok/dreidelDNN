#ifndef DREIDEL_JULES_SPARSEBLOCK_HPP
#define DREIDEL_JULES_SPARSEBLOCK_HPP

#include "../core/TensorView.hpp"
#include "../quant/APoT.hpp"
#include <vector>
#include <cmath>

namespace dreidel {
namespace jules {

/**
 * @brief Sparse Block using ALSH for Expert Selection.
 *
 * Logic:
 * 1. Hash Input -> Index ID
 * 2. Select Expert (slice of weights) based on Index
 * 3. Compute Output = Input * ExpertWeights
 *
 * In this simplified version:
 * - We have N Experts.
 * - ALSH returns an index in [0, N).
 * - Only that expert is executed.
 */
template <size_t INPUT_DIM, size_t OUTPUT_DIM, size_t NUM_EXPERTS>
class SparseBlock {
public:
    // Experts: Quantized weights (APoT codes)
    // Shape: [NUM_EXPERTS, INPUT_DIM, OUTPUT_DIM]
    // Reduced memory footprint: 1 byte per weight.
    uint8_t experts[NUM_EXPERTS * INPUT_DIM * OUTPUT_DIM];

    // ALSH Projections (Hyperplanes)
    // Simplified: Just random vectors for now, stored statically
    // Hashing: sign(dot(x, h))
    // We need log2(NUM_EXPERTS) bits.
    static constexpr int NUM_BITS = (NUM_EXPERTS <= 2) ? 1 :
                                    (NUM_EXPERTS <= 4) ? 2 :
                                    (NUM_EXPERTS <= 8) ? 3 : 4; // etc.

    float hyperplanes[NUM_BITS * INPUT_DIM];

    SparseBlock() {
        if (!quant::APoT::lut_initialized) {
            quant::APoT::init();
        }

        // Initialize with random/identity for mock
        for(size_t i=0; i<NUM_EXPERTS * INPUT_DIM * OUTPUT_DIM; ++i) experts[i] = (i % 256);
        for(size_t i=0; i<NUM_BITS * INPUT_DIM; ++i) hyperplanes[i] = (i%2 == 0) ? 1.0f : -1.0f;
    }

    void forward(core::TensorView<float>& input, core::TensorView<float>& output) {
        // 1. Compute Hash (Select Expert)
        uint32_t expert_idx = compute_hash(input);

        // 2. Execute Expert
        // Weights offset
        size_t weight_offset = expert_idx * INPUT_DIM * OUTPUT_DIM;
        const uint8_t* w_ptr = experts + weight_offset;

        // MatMul: 1xInput * InputxOutput = 1xOutput
        // Dense accumulation
        float* out_ptr = output.data();
        const float* in_ptr = input.data();

        // Zero output
        output.fill(0.0f);

        // Mixed Precision Loop: Float Input * Quantized Weight
        // We can use the product_lut if we quantize input on the fly,
        // OR we can just dequantize weight on the fly.
        // Let's Dequantize weight on the fly to keep precision high for now,
        // or follow DenseLUT "Zero-Math" style.
        // The prompt asked for "Alien Kernels" so let's use the LUT approach for consistency with DenseLUT
        // if we assume inputs are normalized.

        // Using LUT (Quantize Input -> Lookup)
        for (size_t i = 0; i < INPUT_DIM; ++i) {
            // Quantize Input
            uint8_t q_in = quant::APoT::quantize(in_ptr[i]);

            // Row of product lookups
            const float* lut_row = quant::APoT::product_lut[q_in];
            const uint8_t* w_row = w_ptr + i * OUTPUT_DIM;

            for (size_t j = 0; j < OUTPUT_DIM; ++j) {
                out_ptr[j] += lut_row[w_row[j]];
            }
        }
    }

private:
    uint32_t compute_hash(core::TensorView<float>& input) {
        uint32_t hash = 0;
        const float* in_ptr = input.data();

        for (int b = 0; b < NUM_BITS; ++b) {
            float dot = 0.0f;
            const float* h_ptr = hyperplanes + b * INPUT_DIM;

            // Dot product
            for (size_t i = 0; i < INPUT_DIM; ++i) {
                dot += in_ptr[i] * h_ptr[i];
            }

            if (dot >= 0) {
                hash |= (1 << b);
            }
        }

        // Clamp to expert range if bits > experts
        return hash % NUM_EXPERTS;
    }
};

} // namespace jules
} // namespace dreidel

#endif // DREIDEL_JULES_SPARSEBLOCK_HPP
