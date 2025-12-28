#ifndef DREIDEL_QUANT_APOT_HPP
#define DREIDEL_QUANT_APOT_HPP

#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace dreidel {
namespace quant {

// Approximate Power of Two (APoT) Quantization Utilities

/**
 * @brief Precomputed LUT for APoT multiplication.
 *
 * Supports quantization where weights are represented as sums of powers of 2.
 */
class APoT {
public:
    static constexpr int LUT_SIZE = 256;

    // Global LUTs (Static)
    static float dequant_table[LUT_SIZE];

    // The "Global" LUT for Product of two 8-bit codes
    // product_lut[a][b] = dequant(a) * dequant(b)
    // Moved here to avoid duplication in templates
    static float product_lut[256][256];
    static bool lut_initialized;

    /**
     * @brief Initialize the dequantization table and product LUT.
     */
    static void init() {
        if (lut_initialized) return;

        // 1. Init Dequant Table (Mock APoT distribution)
        // [-1.0, 1.0] mapped to [0, 255] linearly for now
        for (int i = 0; i < LUT_SIZE; ++i) {
            float v = (float)i / 255.0f;
            v = v * 2.0f - 1.0f; // Range -1 to 1
            dequant_table[i] = v;
        }

        // 2. Init Product LUT
        for(int i=0; i<256; ++i) {
            float val_i = dequant_table[i];
            for(int j=0; j<256; ++j) {
                float val_j = dequant_table[j];
                product_lut[i][j] = val_i * val_j;
            }
        }

        lut_initialized = true;
    }

    static uint8_t quantize(float val) {
        if (val < -1.0f) val = -1.0f;
        if (val > 1.0f) val = 1.0f;
        float normalized = (val + 1.0f) * 0.5f;
        return static_cast<uint8_t>(normalized * 255.0f + 0.5f);
    }

    static float dequantize(uint8_t code) {
        return dequant_table[code];
    }
};

// Definition of static members
inline float APoT::dequant_table[APoT::LUT_SIZE];
inline float APoT::product_lut[256][256];
inline bool APoT::lut_initialized = false;

} // namespace quant
} // namespace dreidel

#endif // DREIDEL_QUANT_APOT_HPP
