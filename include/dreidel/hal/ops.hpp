#ifndef DREIDEL_HAL_OPS_HPP
#define DREIDEL_HAL_OPS_HPP

#include "defs.hpp"
#include "generic.hpp"
#include "x86.hpp"
#include "arm.hpp"
#include "opengl.hpp"
#include <vector>
#include <cmath>

namespace dreidel {
namespace hal {

// Select the active implementation based on macros
#if defined(DREIDEL_ARCH_AVX512) || defined(DREIDEL_ARCH_AVX2)
    using ActiveOps = x86::Ops;
#elif defined(DREIDEL_ARCH_ARM_NEON)
    using ActiveOps = arm::Ops;
#else
    using ActiveOps = generic::Ops;
#endif

// Helper function for sparse gather that abstracts away SIMD details
inline void sparse_gather(const float* base, const int* indices, float* out, size_t n) {
    size_t i = 0;

#if defined(DREIDEL_ARCH_AVX512)
    for (; i + 16 <= n; i += 16) {
        __m512i vidx = _mm512_loadu_si512((const __m512i*)(indices + i));
        __m512 val = x86::Ops::gather(base, vidx);
        _mm512_storeu_ps(out + i, val);
    }
#elif defined(DREIDEL_ARCH_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256i vidx = _mm256_loadu_si256((const __m256i*)(indices + i));
        __m256 val = x86::Ops::gather(base, vidx);
        _mm256_storeu_ps(out + i, val);
    }
#endif

    for (; i < n; ++i) {
        out[i] = base[indices[i]];
    }
}

/**
 * @brief Alien Ops: Advanced Hardware Intrinsics Wrapper.
 *
 * Provides "Cheat Codes" for:
 * - POPCNT (Hamming Distance)
 * - VPERMB/VPSHUFB (LUT Lookup)
 * - Bitwise Tricks
 */
struct AlienOps {

    // --- 1. POPCNT Oracle ---

    // Extract sign bit from float. 1 if negative, 0 if positive.
    static inline uint32_t sign_bit(float x) {
        union { float f; uint32_t i; } u;
        u.f = x;
        return (u.i >> 31);
    }

    // Population Count (Hamming Weight)
    static inline int popcnt32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_popcount(x);
#elif defined(_MSC_VER)
        return __popcnt(x);
#else
        // Fallback
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
    }

    // Vectorized Sign Mask Extraction
    // Returns a bitmask where bit i is 1 if vector[i] < 0
    static inline uint32_t vec_sign_mask(const float* data, size_t n) {
        uint32_t mask = 0;
        size_t i = 0;

#if defined(DREIDEL_ARCH_AVX512)
        // 16 floats per step
        // _mm512_movepi8_mask is for bytes, for floats use cmp_ps_mask
        // x < 0 check
        __m512 zero = _mm512_setzero_ps();
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(data + i);
            __mmask16 m = _mm512_cmp_ps_mask(v, zero, _CMP_LT_OQ);
            mask |= (static_cast<uint32_t>(m) << i); // This logic assumes n <= 32 for simple scalar mask accumulation
            // But usually this function is called for small N (e.g. 16 or 32 channels).
            // For generic N, we'd need to return std::vector<bool> or BitSet.
            // Assuming Channels <= 32 for Oracle check for now.
        }
#elif defined(DREIDEL_ARCH_AVX2)
        // 8 floats per step
        __m256 zero = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            __m256 cmp = _mm256_cmp_ps(v, zero, _CMP_LT_OQ);
            int m = _mm256_movemask_ps(cmp);
            mask |= (static_cast<uint32_t>(m) << i);
        }
#endif
        // Scalar fallback
        for (; i < n; ++i) {
            if (data[i] < 0.0f) mask |= (1u << i);
        }
        return mask;
    }

    // --- 2. Teleporting Eyes (LUT Lookup) ---

    // Quantize Float (-1.0 to 1.0) to UInt8 (0 to 255)
    // Simple linear mapping: (x + 1) * 127.5
    static inline uint8_t quantize_u8(float x) {
        float q = (x + 1.0f) * 127.5f;
        q = std::min(255.0f, std::max(0.0f, q));
        return static_cast<uint8_t>(q);
    }

    // Vectorized Quantize
    static inline void vec_quantize_u8(const float* src, uint8_t* dst, size_t n) {
        size_t i = 0;
#if defined(DREIDEL_ARCH_AVX2)
        __m256 v_1 = _mm256_set1_ps(1.0f);
        __m256 v_127_5 = _mm256_set1_ps(127.5f);
        for(; i+8 <= n; i+=8) {
            __m256 v = _mm256_loadu_ps(src + i);
            v = _mm256_add_ps(v, v_1);
            v = _mm256_mul_ps(v, v_127_5);
            // Convert to int32 first, then pack? AVX2 pack sequence is annoying.
            // _mm256_cvtps_epi32
            __m256i vi = _mm256_cvtps_epi32(v);
            // We need to pack 32-bit ints to 8-bit.
            // 8 ints -> 8 bytes.
            // Simple scalar store might be faster than complex pack unless N is large.
            // But let's try a simple store loop for now or optimize later.
            // Actually, packssdw (32->16) then packuswb (16->8) works.
            // But we have 256 bits (8 ints).
            // Let's rely on auto-vectorization or scalar for this simple step for now.
            // Just scalar fallback loop below is fine for standard benchmarks unless N is huge.
        }
#endif
        for(; i < n; ++i) dst[i] = quantize_u8(src[i]);
    }

    /**
     * @brief Parallel LUT Lookup using Shuffle.
     *
     * Uses `vpermb` (AVX512) or `vpshufb` (AVX2/SSSE3).
     *
     * @param indices Input bytes (indices into LUT).
     * @param lut The 16-entry LUT (replicated if needed).
     * @param count Number of elements to process.
     * @param out Output buffer.
     */
    static inline void lut_lookup_16(const uint8_t* indices, const uint8_t* lut, uint8_t* out, size_t count) {
        size_t i = 0;

        // LUT Preparation
        // vpshufb uses the lower 4 bits of indices to look up in a 16-byte table.
        // If the MSB of index is set, output is 0.
        // Our nibbles are 0-15, so MSB is 0.

#if defined(DREIDEL_ARCH_AVX512_VBMI) // Hypothetical macro for VBMI
        // AVX-512 VPERMB (Full table lookup for 64 bytes)
        // lut can be up to 64 bytes if we use indices 0-63.
        // But here we do 16-entry LUT repeated?
        // Actually vpshufb exists in AVX-512 too (vpshufb is limited to 128-bit blocks or uses control mask).
        // vpermb is fully general.
        // Assuming we stick to AVX2-compatible vpshufb logic for 4-bit nibbles (0-15).
#endif

#if defined(DREIDEL_ARCH_AVX2)
        // Load LUT: replicated across 128-bit lanes of YMM
        // We assume LUT is 16 bytes.
        __m128i lut_128 = _mm_loadu_si128((const __m128i*)lut);
        __m256i lut_256 = _mm256_broadcastsi128_si256(lut_128);

        for(; i + 32 <= count; i += 32) {
            __m256i idx = _mm256_loadu_si256((const __m256i*)(indices + i));
            // vpshufb looks up in 16-byte blocks independently.
            // idx bytes must act as indices into lut_256's corresponding 16-byte lane.
            // Since lut_256 has the same 16-byte LUT in both lanes, this works perfectly for 0-15 indices.
            __m256i res = _mm256_shuffle_epi8(lut_256, idx);
            _mm256_storeu_si256((__m256i*)(out + i), res);
        }
#endif
        // Scalar Fallback
        for(; i < count; ++i) {
            out[i] = lut[indices[i] & 0x0F]; // Ensure 4-bit range safety
        }
    }

    // Split byte into low/high nibbles for 2 lookups per byte (if input was 8-bit quantized)
    // In "Eyes", we might treat 8-bit input as 2x 4-bit indices.
    static inline void split_nibbles(const uint8_t* input, uint8_t* low, uint8_t* high, size_t n) {
        size_t i = 0;
#if defined(DREIDEL_ARCH_AVX2)
        __m256i mask_low = _mm256_set1_epi8(0x0F);
        for(; i+32 <= n; i+=32) {
            __m256i v = _mm256_loadu_si256((const __m256i*)(input + i));

            __m256i l = _mm256_and_si256(v, mask_low);

            // High nibble: shift right 4
            // No direct epi8 right shift in AVX2.
            // Trick: (v & 0xF0) >> 4.
            // Use vpsrlw (shift words) then mask?
            // Correct way: _mm256_srli_epi16(v, 4) & 0x0F0F... (shifts 16-bit units)
            // If bytes are [h1 l1] [h0 l0], shifting word right 4 gives:
            // [0 h1 l1_hi] [0 h0 l0_hi]...
            // It gets messy.
            // Better: use multiplication or just look up logical shift?
            // Actually, for shuffle LUT, we don't strictly need shift if we have a LUT that handles (x & 0xF0).
            // But we want 0-15 indices.
            // Standard trick for high nibble isolation in vector:
            // v_high = (v >> 4) & 0x0F.
            // Implementation: Unpack to 16-bit, shift, pack? Slow.
            // Multiply by 16 (shift left 4) usually easier? No.
            // Use _mm256_srli_epi64 is bitwise.
            // Wait, vpshufb can mask high bits? No.
            // Let's use scalar fallback for splitting unless we have AVX-512 vdbpsadbw or sim.
            // Actually, we can use logical 'and' with 0xF0 then use a larger LUT (256 entries) for high nibble? No, fits in register.
            // Let's stick to scalar split for simplicity or optimized bitwise if easy.
            // (v & 0xF0) is easy. But shifting down...
            // Use _mm256_srli_epi16 + masking.
            // [B1 B0] -> srli 4 -> [0xxx B1_hi] [0xxx B0_hi] ...
            // Then AND with 0x0F0F.
            // This works correctly because the high nibble of B0 moves to low nibble of B0.
            // The low nibble of B0 moves out.
            // The low nibble of B1 moves into high nibble of B0 (corrupts it).
            // So we must mask.
            __m256i h = _mm256_srli_epi16(v, 4);
            h = _mm256_and_si256(h, mask_low);

            _mm256_storeu_si256((__m256i*)(low + i), l);
            _mm256_storeu_si256((__m256i*)(high + i), h);
        }
#endif
        for(; i < n; ++i) {
            low[i] = input[i] & 0x0F;
            high[i] = (input[i] >> 4) & 0x0F;
        }
    }

    // --- 3. Branchless Gate (Phase 5) ---

    /**
     * @brief Branchless ReLU / Threshold.
     * Sets values < 0 to 0.
     * Uses bitwise masking to avoid branch misprediction.
     */
    static inline void branchless_relu(float* data, size_t n) {
        size_t i = 0;
#if defined(DREIDEL_ARCH_AVX2)
        __m256 zero = _mm256_setzero_ps();
        for(; i+8 <= n; i+=8) {
            __m256 v = _mm256_loadu_ps(data + i);
            v = _mm256_max_ps(v, zero);
            _mm256_storeu_ps(data + i, v);
        }
#endif
        for(; i < n; ++i) {
            union { float f; uint32_t i; } u;
            u.f = data[i];
            // If sign bit (31) is 1, mask is 0xFFFFFFFF, else 0
            // We want to keep x if x > 0.
            // (x >> 31) gives 0 or 1.
            // If we cast to int32, right shift arithmetic fills with sign bit?
            // (int32_t)u.i >> 31 -> -1 (0xFFFFFFFF) if neg, 0 if pos.
            // If neg, we want result 0.
            // result = x & (~mask).
            int32_t mask = (int32_t)u.i >> 31;
            u.i = u.i & (~mask);
            data[i] = u.f;
        }
    }

    // --- 4. APoT Compression (Phase 6 Memory Footprint) ---

    /**
     * @brief Pack Float to 8-bit APoT Code.
     * Format: [Sign:1] [Exponent+64:7]
     * Exponent range: -63 to +63.
     * Stored Exp 0 means Value 0.
     */
    static inline int8_t pack_apot(float val) {
        if (val == 0.0f) return 0;

        float abs_val = std::abs(val);
        // k = round(log2(|val|))
        // Fast approximation?
        // float exponent is in bits 23-30.
        // int exp_f = ((u.i >> 23) & 0xFF) - 127;
        // This is floor(log2). Rounding might be better.
        // Let's use std::log2 for now as packing is one-time (init).
        int k = static_cast<int>(std::round(std::log2(abs_val)));

        // Clamp to -63..63
        if (k < -63) return 0; // Underflow to 0
        if (k > 63) k = 63;

        uint8_t stored = (k + 64) & 0x7F;
        if (val < 0) stored |= 0x80;
        return static_cast<int8_t>(stored);
    }

    /**
     * @brief Unpack 8-bit APoT Code to Float using LUT.
     * Replaces bitwise unpacking with memory lookup (The LUT Trick).
     */
    static inline float unpack_apot(int8_t code) {
        // Simpler initialization:
        static const auto LUT_wrapper = []() {
            struct Table { float data[256]; };
            Table t;
            for (int i = 0; i < 256; ++i) {
                uint8_t u = static_cast<uint8_t>(i);
                uint8_t stored_exp = u & 0x7F;
                if (stored_exp == 0) {
                    t.data[i] = 0.0f;
                } else {
                    int k = static_cast<int>(stored_exp) - 64;
                    float val = std::pow(2.0f, k);
                    if (u & 0x80) val = -val;
                    t.data[i] = val;
                }
            }
            return t;
        }();

        return LUT_wrapper.data[static_cast<uint8_t>(code)];
    }

    // --- 5. Morton / Z-Curve Tools (Phase 6) ---

    /**
     * @brief Interleave bits for Z-Curve (Morton Code).
     * Only supports up to 16-bit coordinates (32-bit result).
     * "Magic Bit Shift" method.
     */
    static inline uint32_t morton_2d(uint16_t x, uint16_t y) {
        auto part1by1 = [](uint16_t n) -> uint32_t {
            uint32_t x = n;
            x = (x | (x << 8)) & 0x00FF00FF;
            x = (x | (x << 4)) & 0x0F0F0F0F;
            x = (x | (x << 2)) & 0x33333333;
            x = (x | (x << 1)) & 0x55555555;
            return x;
        };
        return (part1by1(y) << 1) | part1by1(x);
    }

};

} // namespace hal
} // namespace dreidel

#endif // DREIDEL_HAL_OPS_HPP
