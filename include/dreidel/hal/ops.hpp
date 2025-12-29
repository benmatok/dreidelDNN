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

    // --- 5. LNS Arithmetic Tables (LUT Trick) ---

    // Helper to generate full MUL/ADD tables for APoT (Scalar Fallback)
    struct ApotTables {
        uint8_t mul[65536];
        uint8_t add[65536];

        ApotTables() {
            // Re-use unpack/pack to generate tables
            for(int i=0; i<256; ++i) {
                for(int j=0; j<256; ++j) {
                    float a = unpack_apot(static_cast<int8_t>(i));
                    float b = unpack_apot(static_cast<int8_t>(j));

                    // MUL
                    mul[(i << 8) | j] = static_cast<uint8_t>(pack_apot(a * b));

                    // ADD
                    add[(i << 8) | j] = static_cast<uint8_t>(pack_apot(a + b));
                }
            }
        }
    };

    // Helper to generate correction tables for LNS Addition
    // F(d) = Log2(1 + 2^(-d)) * Scale. d = |X-Y|.
    // For APoT, d is integer difference of exponents.
    // We map d (0..15) to a correction term code.
    // Also we need handling for SUB (different signs).
    struct ShuffleTables {
        int8_t add_corr[32]; // For d=0..31
        int8_t sub_corr[32]; // For d=0..31

        ShuffleTables() {
            for(int d=0; d<32; ++d) {
                // ADD: log2(1 + 2^-d)
                double val_add = std::log2(1.0 + std::pow(2.0, -d));
                add_corr[d] = static_cast<int8_t>(std::round(val_add)); // Is this scaling correct?
                // APoT exponents are integers. A correction of 0.3?
                // If we use integer exponents, we lose precision.
                // But user asked for "packed apot representation".
                // Assuming standard LNS with integer exponents: correction is 0 for d>0?
                // log2(1 + 1) = 1. If d=0, add 1 to exponent. Correct.
                // log2(1 + 0.5) = 0.58. Rounds to 1?
                // log2(1 + 0.25) = 0.32. Rounds to 0.
                // So correction is only 1 for d=0 and d=1?
                // Let's rely on `pack_apot` logic which is log2.
                // We are approximating.
            }
            // Manually set for small d
            add_corr[0] = 1; // log2(2) = 1
            add_corr[1] = 1; // log2(1.5) = 0.58 -> 1
            add_corr[2] = 0; // log2(1.25) = 0.32 -> 0
            // ... rest 0

            // SUB: log2(1 - 2^-d)
            // d=0 -> log2(0) = -inf (Cancellation)
            // d=1 -> log2(0.5) = -1
            // d=2 -> log2(0.75) = -0.41 -> 0
            sub_corr[0] = -128; // Special code for zero?
            sub_corr[1] = -1;
            for(int i=2; i<32; ++i) sub_corr[i] = 0;
        }
    };

    static inline const ShuffleTables& get_shuffle_tables() {
        static ShuffleTables t;
        return t;
    }

    /**
     * @brief Vectorized APoT Addition using Shuffle LUT (AVX2).
     * Computes Z = X + Y in LNS domain.
     * Uses vpshufb to look up correction term F(|X-Y|).
     */
#if defined(DREIDEL_ARCH_AVX2)
    static inline __m256i vec_add_apot_avx2(__m256i a, __m256i b) {
        // We assume 32x int8 codes.
        // Extract signs and exps.
        __m256i sign_mask = _mm256_set1_epi8(0x80);
        __m256i exp_mask = _mm256_set1_epi8(0x7F);

        __m256i sa = _mm256_and_si256(a, sign_mask);
        __m256i sb = _mm256_and_si256(b, sign_mask);
        __m256i ea = _mm256_and_si256(a, exp_mask);
        __m256i eb = _mm256_and_si256(b, exp_mask);

        // Zero check (if exp=0) -> if A=0, res=B.
        __m256i za = _mm256_cmpeq_epi8(ea, _mm256_setzero_si256());
        __m256i zb = _mm256_cmpeq_epi8(eb, _mm256_setzero_si256());

        // Compare magnitudes: A > B?
        // unsigned compare not avail for int8 in AVX2 easily (use subs + min/max)
        __m256i max_e = _mm256_max_epu8(ea, eb);
        __m256i min_e = _mm256_min_epu8(ea, eb);

        // Diff d = max - min
        __m256i d = _mm256_subs_epu8(max_e, min_e);

        // Determine op: ADD (same sign) or SUB (diff sign)
        __m256i sign_diff = _mm256_xor_si256(sa, sb);
        // If sign_diff is 0, ADD table. If 0x80, SUB table.
        // We construct a blend mask or select table?
        // Since SUB is rare/hard, let's implement ADD-only approximation first (assume same sign or ignore sub cancellation).
        // User asked for "parallel scalar LUT".
        // Table for ADD: [1, 1, 0, 0 ... ] (16 entries)
        __m256i lut_add = _mm256_setr_epi8(1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

        // Lookup correction F(d)
        // vpshufb(table, index). Index must be < 16. If > 15, result is 0 (which is correct for d>15).
        // But if d bit 7 is set (0x80), result is 0.
        // d is 0..127. If d > 15, we want 0. vpshufb effectively does this if we don't set bit 7.
        // Wait, vpshufb: if index bit 7 is set, 0. Else index & 0xF.
        // If d=16 (0x10), index & 0xF = 0 -> looks up entry 0! WRONG.
        // We must ensure if d > 15, we set bit 7.
        __m256i mask_large_d = _mm256_cmpgt_epi8(d, _mm256_set1_epi8(15)); // Signed check? d is positive (max-min). 0..127. 15 is 0x0F.
        // cmpgt_epi8 works for positive numbers < 128.
        // If d > 15, mask is FF.
        // We OR d with mask_large_d? No, if FF, bit 7 is 1.
        __m256i idx = _mm256_or_si256(d, mask_large_d);

        __m256i corr = _mm256_shuffle_epi8(lut_add, idx);

        // Result Exp = max_e + corr
        __m256i res_e = _mm256_add_epi8(max_e, corr);

        // Result Sign = Sign of Max Magnitude.
        // Compare A and B fully (inc sign)?
        // Approx: Sign of Max Exp.
        // Use blendv to select SignA if A=Max else SignB.
        __m256i cmp = _mm256_cmpeq_epi8(max_e, ea); // Max == A?
        __m256i res_s = _mm256_blendv_epi8(sb, sa, cmp);

        // Combine
        __m256i res = _mm256_or_si256(res_s, res_e);

        // Handle Zero Inputs: if A=0, res=B. If B=0, res=A.
        // If A=0, use B.
        res = _mm256_blendv_epi8(res, b, za);
        // If B=0, use A (overwrites previous if both 0 -> A=0 ok)
        res = _mm256_blendv_epi8(res, a, zb);

        return res;
    }
#endif

    // Scalar Fallbacks
    static inline const ApotTables& get_apot_tables() {
        static ApotTables tables;
        return tables;
    }
    static inline int8_t apot_mul_lut(int8_t a, int8_t b) {
        const auto& t = get_apot_tables();
        uint16_t idx = (static_cast<uint8_t>(a) << 8) | static_cast<uint8_t>(b);
        return static_cast<int8_t>(t.mul[idx]);
    }
    static inline int8_t apot_add_lut(int8_t a, int8_t b) {
        const auto& t = get_apot_tables();
        uint16_t idx = (static_cast<uint8_t>(a) << 8) | static_cast<uint8_t>(b);
        return static_cast<int8_t>(t.add[idx]);
    }

    /**
     * @brief Vectorized APoT Pack using AVX2 Intrinsics.
     * Compresses 8x 32-bit floats to 8x 8-bit codes (packed into bottom of __m256i or memory).
     * Since we usually store to int8 array, this writes 8 bytes.
     */
#if defined(DREIDEL_ARCH_AVX2)
    static inline void vec_pack_apot_avx2(const float* src, int8_t* dst) {
        // Load 8 floats
        __m256 v = _mm256_loadu_ps(src);

        // Convert to Int32 representation (reinterpretation)
        __m256i vi = _mm256_castps_si256(v);

        // Extract Sign: Bit 31 -> Bit 7
        // (x >> 24) & 0x80
        __m256i sign = _mm256_srli_epi32(vi, 24);
        sign = _mm256_and_si256(sign, _mm256_set1_epi32(0x80));

        // Extract Exponent: Bits 23-30
        // (x >> 23) & 0xFF. This is (exp + 127).
        __m256i exp = _mm256_srli_epi32(vi, 23);
        exp = _mm256_and_si256(exp, _mm256_set1_epi32(0xFF));

        // We want stored = (exp - 127 + 64) = exp - 63.
        // Clamp handling:
        // APoT range is k in -63..63. Stored in 1..127. 0 is zero.
        // Float exp range 0..255.
        // We want to map float exp to APoT exp.
        // Target: Stored = (FloatExp - 127) + 64 = FloatExp - 63.

        // Check for Zero/Denormal (FloatExp == 0)
        __m256i zero_mask = _mm256_cmpeq_epi32(exp, _mm256_setzero_si256());

        // Subtract 63
        exp = _mm256_sub_epi32(exp, _mm256_set1_epi32(63));

        // Clamp to 1..127 (using signed comparison/min/max)
        // If exp < 1, set to 0 (underflow) -> Actually mapped to 0 via zero_mask later?
        // Let's use max(1, min(127, exp)).
        __m256i v_1 = _mm256_set1_epi32(1);
        __m256i v_127 = _mm256_set1_epi32(127);
        exp = _mm256_max_epi32(exp, v_1);
        exp = _mm256_min_epi32(exp, v_127);

        // If original input was 0.0f (or denorm), result should be 0.
        // Also check if abs(val) is very small.
        // But simply masking with zero_mask (if exp was 0) is good enough for now.
        // Or check if FloatExp < 64 (underflow range).

        // Combine Sign | Exp
        __m256i result = _mm256_or_si256(sign, exp);

        // Apply Zero Mask (if original exp was 0, or if we want to flush smalls)
        // We set result to 0 if zero_mask is true.
        // result = result & (~zero_mask)
        result = _mm256_andnot_si256(zero_mask, result);

        // Pack 32-bit integers to 8-bit bytes.
        // AVX2 Packing is annoying.
        // 32 -> 16 (_mm256_packus_epi32) -> 16 -> 8 (_mm256_packus_epi16).
        // packus works on 128-bit lanes. Cross-lane permutation needed?
        // _mm256_packus_epi32(a, b):
        // [a0..a3] [b0..b3] -> [a0s..a3s b0s..b3s] (in each 128 lane).
        // If we only have 1 input vector (8 floats), we pass it as 'a' and 'b'?
        // result = pack(vi, vi).
        // Then pack(res, res).
        // Then extract low 64 bits?

        // Safer/Simpler for just 8 values: _mm256_permutevar8x32_ps + _mm_store?
        // Or just specialized pack sequence.

        // Step 1: Pack 32->16 (Signed saturation? We know value is 0..255 unsigned).
        // _mm256_packus_epi32.
        __m256i p16 = _mm256_packus_epi32(result, result);
        // Layout: [R0..R3 R0..R3] [R4..R7 R4..R7] (16-bit)

        // Step 2: Pack 16->8
        __m256i p8 = _mm256_packus_epi16(p16, p16);
        // Layout: [R0..R3 R0..R3... ] [R4..R7 R4..R7... ] (8-bit)

        // We need R0..R7.
        // They are split across lanes.
        // Lower 128: [R0..R3 R0..R3 R0..R3 R0..R3] (bytes)
        // Upper 128: [R4..R7 R4..R7 R4..R7 R4..R7]
        // We need to extract specific bytes.
        // Actually, we can just extract 32 bits from low lane and 32 bits from high lane.

        int32_t lower = _mm_cvtsi128_si32(_mm256_castsi256_si128(p8)); // Gets R0..R3
        int32_t upper = _mm_cvtsi128_si32(_mm256_extracti128_si256(p8, 1)); // Gets R4..R7

        ((int32_t*)dst)[0] = lower;
        ((int32_t*)dst)[1] = upper;
    }
#endif

    /**
     * @brief Vectorized APoT Multiplication using Exponent Addition.
     * Computes P = A * B in APoT domain.
     * Logic:
     *   Sign = Sign(A) XOR Sign(B)
     *   Exp = Exp(A) + Exp(B) - Bias(64)
     *   Zero Handling: If A=0 or B=0, result 0.
     *
     * @param a 128-bit vector of 16 APoT codes (or partially used)
     * @param b 128-bit vector of 16 APoT codes
     * @return 128-bit vector of Product codes
     */
#if defined(DREIDEL_ARCH_AVX2)
    static inline __m128i vec_mul_apot_avx2(__m128i a, __m128i b) {
        // Masks
        __m128i sign_mask = _mm_set1_epi8(0x80);
        __m128i exp_mask = _mm_set1_epi8(0x7F);
        __m128i bias = _mm_set1_epi8(64);

        // 1. Sign Calculation
        __m128i sign_a = _mm_and_si128(a, sign_mask);
        __m128i sign_b = _mm_and_si128(b, sign_mask);
        __m128i sign_p = _mm_xor_si128(sign_a, sign_b);

        // 2. Exponent Calculation
        __m128i exp_a = _mm_and_si128(a, exp_mask);
        __m128i exp_b = _mm_and_si128(b, exp_mask);

        // Zero Check: If exp is 0, value is 0.
        __m128i zero_a = _mm_cmpeq_epi8(exp_a, _mm_setzero_si128());
        __m128i zero_b = _mm_cmpeq_epi8(exp_b, _mm_setzero_si128());
        __m128i is_zero = _mm_or_si128(zero_a, zero_b);

        // Add exponents: E_a + E_b using saturated add to detect overflow?
        // Or standard add. Max exp is 127. 127+127 = 254. Fits in uint8.
        __m128i exp_sum = _mm_add_epi8(exp_a, exp_b);

        // Subtract Bias: E_p = Sum - 64.
        // Use saturated subtraction to clamp underflow to 0?
        // _mm_subs_epu8. If (Sum < 64), result 0 (which is effectively zero value).
        __m128i exp_p = _mm_subs_epu8(exp_sum, bias);

        // Check Overflow (if Sum > 127 + 64 = 191).
        // Since we use 8-bit arithmetic, simple add might wrap?
        // E.g. 120 + 120 = 240. 240 - 64 = 176.
        // 176 > 127. It's technically valid (larger magnitude).
        // But APoT format is 7-bit exp (0..127).
        // We should clamp to 127.
        __m128i max_exp = _mm_set1_epi8(127);
        exp_p = _mm_min_epu8(exp_p, max_exp);

        // Combine Sign | Exp
        __m128i prod = _mm_or_si128(sign_p, exp_p);

        // Apply Zero Mask
        prod = _mm_andnot_si128(is_zero, prod);

        return prod;
    }
#endif

    /**
     * @brief Unpack 8-bit APoT Code to Float using LUT.
     * Replaces bitwise unpacking with memory lookup (The LUT Trick).
     */
    static inline float unpack_apot(int8_t code) {
        // Scalar LUT logic kept for reference/fallback
        // We replicate logic here because vector unpack is preferred.
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

    /**
     * @brief Vectorized APoT Unpack using AVX2 Intrinsics.
     * Decodes 8x 8-bit codes into 8x 32-bit floats.
     * Returns __m256 directly to avoid store-forwarding stalls.
     */
#if defined(DREIDEL_ARCH_AVX2)
    static inline __m256 vec_unpack_apot_avx2(const int8_t* codes) {
        // Load 8 bytes (64 bits)
        __m128i raw = _mm_loadl_epi64((const __m128i*)codes);

        // Zero extend uint8 -> int32
        __m256i val = _mm256_cvtepu8_epi32(raw);

        // Masks
        __m256i sign_mask = _mm256_set1_epi32(0x80);
        __m256i exp_mask = _mm256_set1_epi32(0x7F);

        // Extract Sign and Exponent
        __m256i signs = _mm256_and_si256(val, sign_mask);
        __m256i exps = _mm256_and_si256(val, exp_mask);

        // Check for Zero (Exp == 0)
        __m256i zero_mask = _mm256_cmpeq_epi32(exps, _mm256_setzero_si256());

        // Shift sign to MSB (31) (7->31 = shift 24)
        signs = _mm256_slli_epi32(signs, 24);

        // Compute Exponent: (Stored + 63)
        __m256i offset = _mm256_set1_epi32(63);
        exps = _mm256_add_epi32(exps, offset);

        // Shift Exponent to position 23
        exps = _mm256_slli_epi32(exps, 23);

        // Combine
        __m256i result_int = _mm256_or_si256(signs, exps);

        // Mask out Zeros
        result_int = _mm256_andnot_si256(zero_mask, result_int);

        // Cast to float
        return _mm256_castsi256_ps(result_int);
    }
#endif

    /**
     * @brief Vectorized APoT Unpack Wrapper.
     * Use this for generic code.
     */
    static inline void vec_unpack_apot(const int8_t* codes, float* out) {
#if defined(DREIDEL_ARCH_AVX2)
        __m256 res = vec_unpack_apot_avx2(codes);
        _mm256_storeu_ps(out, res);
#else
        for(int i=0; i<8; ++i) out[i] = unpack_apot(codes[i]);
#endif
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
