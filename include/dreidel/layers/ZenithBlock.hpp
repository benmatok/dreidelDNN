#pragma once

#include "Layer.hpp"
#include "../core/Memory.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

namespace dreidel {
namespace layers {

// Forward declare specialized ZenithBlock
class ZenithBlock;

/**
 * @brief The Zenith Block (Strictly Optimized for APoT).
 *
 * Replaces Standard Conv2D.
 * STRICT APoT MODE: This block only supports optimized APoT execution on int8 tensors.
 *
 * Pipeline:
 * 1. Oracle (Gating)
 * 2. Eyes (Spatial Conv)
 * 3. Mixer (Permute -> FWHT -> Scale -> Bias -> ReLU)
 */
class ZenithBlock : public Layer<int8_t> {
public:
    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim, size_t arena_size = 1024*1024, bool use_gating = false)
        : channels_(channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          arena_(arena_size),
          use_gating_(use_gating),
          packed_weights_(channels * kernel_size * kernel_size),
          spectral_scales_(channels),
          bias_(channels),
          perm_indices_(channels)
    {
        // Random Init Weights
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist_code(0, 255);

        for(auto& w : packed_weights_) w = static_cast<int8_t>(dist_code(gen));
        for(auto& s : spectral_scales_) s = static_cast<int8_t>(dist_code(gen));

        std::fill(bias_.begin(), bias_.end(), 0);
        std::iota(perm_indices_.begin(), perm_indices_.end(), 0);

        oracle_projection_.resize(channels);
        for(auto& p : oracle_projection_) p = static_cast<int8_t>(dist_code(gen));
    }

#if defined(DREIDEL_ARCH_AVX2)
    // Intra-Register FWHT for 32 elements (YMM)
    // Performs stages 1, 2, 4, 8, 16 in-register.
    static inline __m256i fwht_avx2_intra(__m256i v) {
        // We use vpshufb to permute for each stride.
        // Stride 1: [0 1 2 3] -> a=[0 2], b=[1 3] ?
        // Or adjacent: x+y, x-y.
        // x = v (masked?), y = v shifted?
        // Note: vec_add_apot_avx2 adds two vectors elementwise.
        // To do [x0+x1, x0-x1, x2+x3, x2-x3...], we need to form:
        // A = [x0 x0 x2 x2 ...]
        // B = [x1 x1 x3 x3 ...]
        // Then Sum = A+B, Sub = A-B.
        // Then merge.
        // Mask based merge might be slow.
        // Better: Deinterleave.
        // A = [x0 x2 ...], B = [x1 x3 ...].
        // S = A+B, D = A-B.
        // Res = Interleave(S, D) -> [s0 d0 s1 d1 ...]

        // --- Stride 1 ---
        // A = Shuffle(v, mask_A) -> 0 2 4 6 ...
        // B = Shuffle(v, mask_B) -> 1 3 5 7 ...
        // S = A+B, D=A-B
        // Res = Unpack(S, D)

        // This is generic for any stride S inside register.
        // A = v & (indices having bit k=0) shifted?
        // Actually, straightforward implementation:

        // Stride 1
        {
            // Mask 0x0, 0x2, 0x4... repeated? vpshufb only creates duplicates if we ask.
            // But vpshufb index is 4-bit? No, 0..15.
            // Lane 0: 0, 2, 4... 14. (8 elements).
            // We need 16 elements per lane (bytes).
            // A: 0, 2, 4, 6, 8, 10, 12, 14, 0, 2... ?
            // No, we process 32 bytes.
            // Lane 1 mirrors Lane 0 logic.
            // A_idx: 0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4... (Repeated 2x in YMM const?)
            // B_idx: 1, 3, 5, 7, 9, 11, 13, 15...

            __m256i idx_a = _mm256_setr_epi8(0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14);
            __m256i idx_b = _mm256_setr_epi8(1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15);
            // Wait, standard _mm256_setr_epi8 sets 32 separate bytes.
            // But vpshufb uses in-lane indices (0-15). Index 0 refers to byte 0 of that 128-lane.
            // So [0,2..14] works for both low and high lane.
            // But we need to define the 32-byte constant carefully.
            // Ideally:
            __m256i mask_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,2,4,6,8,10,12,14, 0,0,0,0,0,0,0,0)); // Wait, we need all pairs.
            // Deinterleave stride 1:
            // Lane: [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
            // A:    [0 2 4 6 8 10 12 14 ...]
            // B:    [1 3 5 7 9 11 13 15 ...]
            // Sum = A+B, Diff = A-B.
            // Res = [s0 d0 s1 d1 ...] = Interleave(Sum, Diff).

            // Mask A:
            __m256i ma = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14)); // Actually 8 items? No 16 bytes. 0,2..14 is 8 items.
            // Wait, 16 bytes. 0,2,4,6,8,10,12,14 is 8 indices.
            // Where do the other 8 go?
            // Ah, we are splitting the vector size 32 into two vectors of size 16?
            // No, we want to perform 16 butterflies.
            // Input 32 bytes. 16 pairs.
            // A needs 16 bytes (the 'left' operands). B needs 16 bytes.
            // But A and B must be in __m256i to use vec_add_apot.
            // So A has 16 useful bytes, 16 garbage?
            // No, we process 32 bytes in parallel.
            // But stride 1 couples (0,1).
            // A = [0 2 4 ... 30]. (16 items).
            // To fill a 32-byte register, we need 32 items?
            // Ah, we can't do the full 32-byte set in one go if we reduce 32 -> 16.
            // But FWHT is in-place size 32.
            // 32 inputs -> 32 outputs.
            // Pairs are local. (0,1)->(0',1'). (2,3)->(2',3').
            // A = [0 2 4 ...]. B = [1 3 5 ...].
            // S = A+B. D = A-B.
            // Result = [S0 D0 S1 D1 ...].
            // To do this vectorized:
            // A = v & mask? Or shuffle?
            // To construct A=[0 2 4...], we duplicate?
            // Actually, we can just do:
            // A = [0 2 4 ... | 0 2 4 ...]
            // B = [1 3 5 ... | 1 3 5 ...]
            // Compute S, D (each has 16 valid results, replicated or zero padded).
            // Then Shuffle/Unpack S and D to interleave.
            // AVX2 unpacklo_epi8 / unpackhi_epi8 does exactly [s0 d0 s1 d1...].

            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14)); // Fills lane with evens
            // Wait, lane has 16 bytes. We want 0,2,4,6,8,10,12,14 (8 bytes).
            // What about the rest?
            // We can't fit 32 bytes of "A" into 32 bytes if A is half of V.
            // We perform calculation on permuted V.
            // A: [0 2 4 ... 14  0 2 4 ... 14] (16 items per lane? No, 8 unique evens per lane).
            // So A contains 2 copies of even bytes? Or garbage?
            // Let's assume we ignore upper half of A result.
            // Calculate S = A+B. D = A-B.
            // S will have [s0 s1 ... s7  garbage].
            // D will have [d0 d1 ... d7  garbage].
            // Unpacklo(S, D) -> [s0 d0 s1 d1 ... s7 d7]. Perfect!

            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(1,3,5,7,9,11,13,15, 1,3,5,7,9,11,13,15));
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);

            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);

            v = _mm256_unpacklo_epi8(s, d);
            // Note: unpacklo works lane-wise.
            // Lane 0: unpack(s_lo, d_lo). s_lo has valid sums. d_lo valid diffs. Result [s0 d0...].
            // Lane 1: same.
            // This works for Stride 1.
        }

        // Stride 2
        {
            // Pairs (0,2), (1,3).
            // A = [0 1 4 5 8 9 12 13]
            // B = [2 3 6 7 10 11 14 15]
            // Res = [s0 s1 d0 d1 s2 s2 d2 d3...]
            // Use unpacklo_epi16 to merge?

            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,1,4,5,8,9,12,13, 0,1,4,5,8,9,12,13));
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(2,3,6,7,10,11,14,15, 2,3,6,7,10,11,14,15));

            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);

            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);

            v = _mm256_unpacklo_epi16(s, d);
        }

        // Stride 4
        {
            // A = [0-3, 8-11]
            // B = [4-7, 12-15]
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,1,2,3,8,9,10,11, 0,1,2,3,8,9,10,11));
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(4,5,6,7,12,13,14,15, 4,5,6,7,12,13,14,15));

            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);

            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);

            v = _mm256_unpacklo_epi32(s, d);
        }

        // Stride 8
        {
            // A = [0-7]
            // B = [8-15]
            __m256i shuf_a = _mm256_broadcastsi128_si256(_mm_setr_epi8(0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7));
            __m256i shuf_b = _mm256_broadcastsi128_si256(_mm_setr_epi8(8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15));

            __m256i a = _mm256_shuffle_epi8(v, shuf_a);
            __m256i b = _mm256_shuffle_epi8(v, shuf_b);

            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);

            v = _mm256_unpacklo_epi64(s, d);
        }

        // Stride 16
        {
            // 16-stride is across 128-bit lanes in YMM (32 bytes).
            // Lane 0: [0..15]. Lane 1: [16..31].
            // A = Lane 0. B = Lane 1.
            // S = A+B. D = A-B.
            // Result = [S, D].

            __m128i lane0 = _mm256_castsi256_si128(v);
            __m128i lane1 = _mm256_extracti128_si256(v, 1);

            // To use vec_add_apot_avx2 (256 bit), we could process A and B in parallel?
            // Actually, we can perform A+B and A-B using 128-bit ops if available, or expand.
            // Ops are defined for __m256i.
            // Let's construct 256 inputs.
            // A_256 = [A, A]. B_256 = [B, B].
            // Calc S_256, D_256.
            // Extract halves.
            __m256i a = _mm256_broadcastsi128_si256(lane0);
            __m256i b = _mm256_broadcastsi128_si256(lane1);

            __m256i s = hal::AlienOps::vec_add_apot_avx2(a, b);
            __m256i sign_mask = _mm256_set1_epi8(0x80);
            __m256i neg_b = _mm256_xor_si256(b, sign_mask);
            __m256i d = hal::AlienOps::vec_add_apot_avx2(a, neg_b);

            // Result is [S_lower, D_lower]. (Since A, B were broadcast, S/D upper same as lower)
            // Need to assemble: [s_lo, d_lo].
            // _mm256_permute2x128_si256(s, d, mask)?
            // Mask 0x20: Lo from A (s), Lo from B (d). -> [s_lo, d_lo].
            v = _mm256_permute2x128_si256(s, d, 0x20);
        }

        return v;
    }
#endif

    Tensor<int8_t> forward(const Tensor<int8_t>& input) override {
        auto shape = input.shape();
        size_t batch = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3];

        Tensor<int8_t> output(shape);

        const int8_t* in_ptr = input.data();
        int8_t* out_ptr = output.data();
        const int8_t* w_ptr = packed_weights_.data();
        const int8_t* scale_ptr = spectral_scales_.data();
        const int8_t* bias_ptr = bias_.data();
        const int* perm_ptr = perm_indices_.data();
        const int8_t* oracle_ptr = oracle_projection_.data();

        int k_rad = kernel_size_ / 2;

        arena_.reset();
        int8_t* pixel_buffer = arena_.allocate<int8_t>(C);
        int8_t* mixer_buffer = arena_.allocate<int8_t>(C);

        constexpr int BLOCK_H = 8;
        constexpr int BLOCK_W = 8;

        uint32_t oracle_mask = 0;
        if(use_gating_) {
            for(size_t i=0; i<C; ++i) {
                if (oracle_ptr[i] & 0x80) oracle_mask |= (1u << i);
            }
        }

        for(size_t n=0; n<batch; ++n) {

            if (use_gating_) {
                size_t ch = H/2, cw = W/2;
                const int8_t* p_center = in_ptr + ((n*H + ch)*W + cw)*C;
                uint32_t input_mask = 0;
#if defined(DREIDEL_ARCH_AVX2)
                __m256i v = _mm256_loadu_si256((const __m256i*)p_center);
                input_mask = _mm256_movemask_epi8(v);
#else
                for(size_t i=0; i<std::min(C, (size_t)32); ++i) {
                    if (p_center[i] & 0x80) input_mask |= (1u << i);
                }
#endif
                int dist = hal::AlienOps::popcnt32(input_mask ^ oracle_mask);
                if (dist > 16) {
                    int8_t* p_out_start = out_ptr + n * H * W * C;
                    std::fill(p_out_start, p_out_start + H * W * C, 0);
                    continue;
                }
            }

            for(size_t by=0; by<H; by+=BLOCK_H) {
                for(size_t bx=0; bx<W; bx+=BLOCK_W) {
                    for(size_t dy=0; dy<BLOCK_H; ++dy) {
                        for(size_t dx=0; dx<BLOCK_W; ++dx) {
                            size_t h = by + dy;
                            size_t w = bx + dx;
                            if (h >= H || w >= W) continue;

                            // 2. Eyes (Spatial Convolution)
                            for(size_t c=0; c<C; ++c) pixel_buffer[c] = 0;

                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int ih = h + ky;
                                    int iw = w + kx;
                                    if (ih>=0 && ih<H && iw>=0 && iw<W) {
                                        const int8_t* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                        int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                        const int8_t* p_w = w_ptr + k_idx * channels_;
                                        size_t c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                                        for(; c+32 <= C; c+=32) {
                                            __m256i v_in = _mm256_loadu_si256((const __m256i*)(p_in + c));
                                            __m256i v_w  = _mm256_loadu_si256((const __m256i*)(p_w + c));
                                            __m128i in_lo = _mm256_castsi256_si128(v_in);
                                            __m128i in_hi = _mm256_extracti128_si256(v_in, 1);
                                            __m128i w_lo  = _mm256_castsi256_si128(v_w);
                                            __m128i w_hi  = _mm256_extracti128_si256(v_w, 1);
                                            __m128i prod_lo = hal::AlienOps::vec_mul_apot_avx2(in_lo, w_lo);
                                            __m128i prod_hi = hal::AlienOps::vec_mul_apot_avx2(in_hi, w_hi);
                                            __m256i v_prod = _mm256_set_m128i(prod_hi, prod_lo);
                                            __m256i v_acc = _mm256_loadu_si256((const __m256i*)(pixel_buffer + c));
                                            v_acc = hal::AlienOps::vec_add_apot_avx2(v_acc, v_prod);
                                            _mm256_storeu_si256((__m256i*)(pixel_buffer + c), v_acc);
                                        }
#endif
                                        for(; c<C; ++c) {
                                            int8_t prod = hal::AlienOps::apot_mul_lut(p_in[c], p_w[c]);
                                            pixel_buffer[c] = hal::AlienOps::apot_add_lut(pixel_buffer[c], prod);
                                        }
                                    }
                                }
                            }

                            // 3. Mixer (Spectral)
                            for(size_t c=0; c<C; ++c) mixer_buffer[c] = pixel_buffer[perm_ptr[c]];

                            // FWHT: Hybrid Strategy
                            // Use Intra-Register for initial stages (stride 1..16)
                            // Use Inter-Register for later stages (stride 32..)

                            // A. Intra-Register Pass (Strides 1, 2, 4, 8, 16)
                            size_t c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                            // Process in chunks of 32
                            for (; c + 32 <= C; c += 32) {
                                __m256i v = _mm256_loadu_si256((const __m256i*)(mixer_buffer + c));
                                v = fwht_avx2_intra(v);
                                _mm256_storeu_si256((__m256i*)(mixer_buffer + c), v);
                            }
#endif
                            // Scalar fallback for remaining (if C % 32 != 0)
                            // Or if AVX not avail.
                            // Note: If C=16, we can load 16 bytes into YMM (with garbage upper), process, store 16.
                            // Or handle stride 1..8 with 16 bytes.
                            // Our intra helper does 32. For C=16, we can load 16, expand to 32?
                            if (c < C) {
                                // Scalar loop for small C or tail
                                size_t limit = C - c; // e.g. 16
                                size_t sub_h = 1;
                                while (sub_h < 32 && sub_h < limit) {
                                    for(size_t i=c; i<C; i+=sub_h*2) {
                                        for(size_t j=i; j<i+sub_h; ++j) {
                                            if (j+sub_h < C) {
                                                int8_t x = mixer_buffer[j];
                                                int8_t y = mixer_buffer[j + sub_h];
                                                mixer_buffer[j] = hal::AlienOps::apot_add_lut(x, y);
                                                mixer_buffer[j + sub_h] = hal::AlienOps::apot_add_lut(x, y ^ 0x80);
                                            }
                                        }
                                    }
                                    sub_h *= 2;
                                }
                            }

                            // B. Inter-Register Pass (Strides 32, 64...)
                            size_t h_len = 32;
                            while (h_len < C) {
                                bool handled = false;
#if defined(DREIDEL_ARCH_AVX2)
                                // Strides >= 32 are memory-contiguous blocks
                                for (size_t i = 0; i < C; i += h_len * 2) {
                                    for (size_t j = i; j < i + h_len; j += 32) {
                                        __m256i x = _mm256_loadu_si256((const __m256i*)(mixer_buffer + j));
                                        __m256i y = _mm256_loadu_si256((const __m256i*)(mixer_buffer + j + h_len));
                                        __m256i sum = hal::AlienOps::vec_add_apot_avx2(x, y);
                                        __m256i sign_mask = _mm256_set1_epi8(0x80);
                                        __m256i neg_y = _mm256_xor_si256(y, sign_mask);
                                        __m256i sub = hal::AlienOps::vec_add_apot_avx2(x, neg_y);
                                        _mm256_storeu_si256((__m256i*)(mixer_buffer + j), sum);
                                        _mm256_storeu_si256((__m256i*)(mixer_buffer + j + h_len), sub);
                                    }
                                }
                                handled = true;
#endif
                                if (!handled) {
                                    for (size_t i = 0; i < C; i += h_len * 2) {
                                        for (size_t j = i; j < i + h_len; ++j) {
                                            int8_t x = mixer_buffer[j];
                                            int8_t y = mixer_buffer[j + h_len];
                                            mixer_buffer[j] = hal::AlienOps::apot_add_lut(x, y);
                                            mixer_buffer[j + h_len] = hal::AlienOps::apot_add_lut(x, y ^ 0x80);
                                        }
                                    }
                                }
                                h_len *= 2;
                            }

                            // c. Scale & Bias & ReLU
                            c = 0;
#if defined(DREIDEL_ARCH_AVX2)
                            for(; c+32 <= C; c+=32) {
                                __m256i v_val = _mm256_loadu_si256((const __m256i*)(mixer_buffer + c));
                                __m256i v_scale = _mm256_loadu_si256((const __m256i*)(scale_ptr + c));
                                __m256i v_bias = _mm256_loadu_si256((const __m256i*)(bias_ptr + c));
                                __m128i val_lo = _mm256_castsi256_si128(v_val);
                                __m128i val_hi = _mm256_extracti128_si256(v_val, 1);
                                __m128i sc_lo = _mm256_castsi256_si128(v_scale);
                                __m128i sc_hi = _mm256_extracti128_si256(v_scale, 1);
                                __m128i res_lo = hal::AlienOps::vec_mul_apot_avx2(val_lo, sc_lo);
                                __m128i res_hi = hal::AlienOps::vec_mul_apot_avx2(val_hi, sc_hi);
                                __m256i v_res = _mm256_set_m128i(res_hi, res_lo);
                                v_res = hal::AlienOps::vec_add_apot_avx2(v_res, v_bias);
                                _mm256_storeu_si256((__m256i*)(mixer_buffer + c), v_res);
                            }
#endif
                            for(; c<C; ++c) {
                                int8_t val = mixer_buffer[c];
                                val = hal::AlienOps::apot_mul_lut(val, scale_ptr[c]);
                                val = hal::AlienOps::apot_add_lut(val, bias_ptr[c]);
                                if (val & 0x80) val = 0;
                                mixer_buffer[c] = val;
                            }

                            int8_t* p_out = out_ptr + ((n*H + h)*W + w)*C;
                            for(size_t c=0; c<C; ++c) p_out[c] = mixer_buffer[c];
                        }
                    }
                }
            }
        }
        return output;
    }

    Tensor<int8_t> backward(const Tensor<int8_t>& grad_output) override {
        return Tensor<int8_t>();
    }

    std::vector<Tensor<int8_t>*> parameters() override {
        return {};
    }

    std::string name() const override { return "ZenithBlock"; }

private:
    size_t channels_;
    size_t kernel_size_;
    size_t spectral_dim_;
    core::Arena arena_;
    bool use_gating_;
    std::vector<int8_t> packed_weights_;
    std::vector<int8_t> spectral_scales_;
    std::vector<int8_t> bias_;
    std::vector<int> perm_indices_;
    std::vector<int8_t> oracle_projection_;
};

} // namespace layers
} // namespace dreidel
