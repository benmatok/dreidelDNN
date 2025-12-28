#pragma once

#include "Layer.hpp"
#include "../core/Memory.hpp"
#include "../algo/WHT.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>

namespace dreidel {
namespace layers {

// Helper: Quantize to Additive Power-of-Two (APoT)
// Finds nearest val = sign * 2^k
template <typename T>
T quantize_apot(T val) {
    if (val == 0) return 0;
    T sign = (val > 0) ? 1.0 : -1.0;
    T abs_val = std::abs(val);
    T log2_val = std::log2(abs_val);
    T k = std::round(log2_val);
    // Clamp range if needed, e.g., -127 to 127 exponents
    return sign * std::pow(static_cast<T>(2.0), k);
}

// Helper: Simulate LUT Multiplication (Pixel * Weight)
// In hardware this is a fetch. Here we just multiply but ensure weight is APoT.
template <typename T>
T lut_mul(T pixel, T weight_apot) {
    // In strict simulation, we'd cast pixel to byte, weight to code, lookup.
    // Here we just multiply.
    return pixel * weight_apot;
}

/**
 * @brief The Zenith Block.
 *
 * Replaces Standard Conv2D.
 * Pipeline: Oracle (ALSH) -> Eyes (Spatial APoT) -> Mixer (Spectral APoT).
 * Memory: Zero-Allocation via Arena.
 *
 * NOTE: This block is stateful due to the `Arena` allocator.
 * It assumes serial execution within a single instance.
 * For parallel processing, each thread must have its own instance or a thread-local Arena.
 */
template <typename T>
class ZenithBlock : public Layer<T> {
public:
    ZenithBlock(size_t channels, size_t kernel_size, size_t spectral_dim, size_t arena_size = 1024*1024, bool use_gating = false, bool use_apot = true)
        : channels_(channels), kernel_size_(kernel_size), spectral_dim_(spectral_dim),
          arena_(arena_size), // 1MB Scratchpad default
          use_gating_(use_gating),
          use_apot_(use_apot),
          spatial_weights_({channels, 1, kernel_size, kernel_size}),
          spectral_scales_({1, spectral_dim}),
          perm_indices_(spectral_dim),
          grad_spatial_({channels, 1, kernel_size, kernel_size}),
          grad_scales_({1, spectral_dim})
    {
        // Initialization
        spatial_weights_.random(0.0, std::sqrt(2.0 / (kernel_size * kernel_size * channels)));

        // Quantize spatial weights to APoT immediately
        T* sw_ptr = spatial_weights_.data();
        for(size_t i=0; i<spatial_weights_.size(); ++i) {
            sw_ptr[i] = quantize_apot(sw_ptr[i]);
        }

        // Compress Weights for Phase 6
        packed_weights_.resize(spatial_weights_.size());
        for(size_t i=0; i<spatial_weights_.size(); ++i) {
            // Only float supported for APoT pack currently in ops
            packed_weights_[i] = hal::AlienOps::pack_apot(static_cast<float>(sw_ptr[i]));
        }

        // Fix: Initialize scales to 1/N to counteract FWHT gain (which is N for unnormalized, or sqrt(N)?)
        // Standard WHT unnormalized has energy gain N^2, amplitude gain N?
        // Actually H*H = N*I. So x -> Hx scales L2 norm by sqrt(N).
        // To preserve variance, we should scale by 1/sqrt(N).
        // If we want range preservation, maybe 1/N.
        // Let's use 1.0 / sqrt(channels)
        T scale_init = 1.0 / std::sqrt(static_cast<T>(channels));
        spectral_scales_.fill(scale_init);

        // Permutation init
        std::iota(perm_indices_.begin(), perm_indices_.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(perm_indices_.begin(), perm_indices_.end(), g);

        // Oracle Init (Simplified ALSH)
        // Random projection vector for hashing
        oracle_projection_.resize(channels);
        std::uniform_real_distribution<T> dist(-1.0, 1.0);
        for(auto& v : oracle_projection_) v = dist(g);

        // Assign Active Bucket (0 or 1)
        active_bucket_ = 1; // We only run if hash(input) == 1.
        // For simulation, we want it to run most of the time unless we really have many blocks.
        // Let's assume a 1-bit hash for now (Sign of projection).
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Input assumed (N, H, W, C)
        auto shape = input.shape();
        size_t batch = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3]; // Must match channels_

        // Prepare Output
        Tensor<T> output(shape);
        output.fill(0); // If sleep, output 0

        // Reset Arena for this pass
        arena_.reset();

        // Allocate scratch buffers in Arena for single pixel operation
        // Fused Pipeline requires just one buffer of size C for intermediate results
        T* pixel_buffer = arena_.allocate<T>(C);
        T* temp_perm_buffer = arena_.allocate<T>(C);

        // Alien Buffers removed as per code review (they were unused in simulation)

        // Pointers
        const T* in_ptr = input.data();
        T* out_ptr = output.data();

        const T* sw_ptr = spatial_weights_.data();
        const T* scale_ptr = spectral_scales_.data();
        int k_rad = kernel_size_ / 2;

        // Quantize scales for forward pass (APoT)
        // In real training we'd keep float weights and quantize on forward.
        // Here we just quantize on the fly or use stored.
        std::vector<T> effective_scales(C);
        for(size_t i=0; i<C; ++i) effective_scales[i] = quantize_apot(scale_ptr[i]);

        // Pre-compute Oracle Sign Mask
        uint32_t oracle_mask = 0;
        if(use_gating_) {
            oracle_mask = hal::AlienOps::vec_sign_mask(oracle_projection_.data(), C);
        }

        // Execution
        for(size_t n=0; n<batch; ++n) {

            // 1. The Psychic Oracle (Optimized ALSH)
            if (use_gating_) {
                // Check center pixel sign signature
                size_t ch = H/2, cw = W/2;
                const T* p_center = in_ptr + ((n*H + ch)*W + cw)*C;

                uint32_t input_mask = hal::AlienOps::vec_sign_mask(p_center, C);

                // Hamming Distance
                uint32_t xor_val = input_mask ^ oracle_mask;
                int dist = hal::AlienOps::popcnt32(xor_val);

                // Threshold (heuristic: if distance < C/2)
                if (dist > (int)(C/2)) continue; // Sleep
            }

            // Phase 6: Super-Block Tiling (Blocked Z-Curve Approximation)
            // Iterate in 8x8 blocks to improve cache locality.
            constexpr int BLOCK_H = 8;
            constexpr int BLOCK_W = 8;

            for(size_t by=0; by<H; by+=BLOCK_H) {
                for(size_t bx=0; bx<W; bx+=BLOCK_W) {

                    // Iterate within block
                    for(size_t dy=0; dy<BLOCK_H; ++dy) {
                        for(size_t dx=0; dx<BLOCK_W; ++dx) {
                            size_t h = by + dy;
                            size_t w = bx + dx;
                            if (h >= H || w >= W) continue;

                            // 2. The Teleporting Eyes (Spatial LUT & AVX)
                            // Reset accumulator
                            for(size_t c=0; c<C; ++c) pixel_buffer[c] = 0;

                            for(int ky=-k_rad; ky<=k_rad; ++ky) {
                                for(int kx=-k_rad; kx<=k_rad; ++kx) {
                                    int ih = h + ky;
                                    int iw = w + kx;
                                    if (ih>=0 && ih<H && iw>=0 && iw<W) {
                                        const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                        int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);

                                        if (use_apot_) {
                                            // Phase 6: Use packed weights to reduce bandwidth
                                            const int8_t* p_w_packed = packed_weights_.data() + k_idx * channels_;

                                            // Vectorized Unpack & FMA (AVX2 if available)
                                            size_t c = 0;
#if defined(DREIDEL_ARCH_AVX2) && defined(__AVX2__)
                                            // Assuming T is float and channels aligned
                                            if (std::is_same<T, float>::value) {
                                                for(; c+8 <= C; c+=8) {
                                                    // 1. Unpack 8 weights (In Registers)
                                                    __m256 v_w = hal::AlienOps::vec_unpack_apot_avx2(p_w_packed + c);

                                                    // 2. Load 8 pixels
                                                    __m256 v_in = _mm256_loadu_ps((const float*)(p_in + c));

                                                    // 3. Load accumulator
                                                    __m256 v_acc = _mm256_loadu_ps((const float*)(pixel_buffer + c));

                                                    // 4. FMA
                                                    v_acc = _mm256_fmadd_ps(v_in, v_w, v_acc);

                                                    // 5. Store back
                                                    _mm256_storeu_ps((float*)(pixel_buffer + c), v_acc);
                                                }
                                            }
#endif
                                            // Scalar Fallback
                                            for(; c<C; ++c) {
                                                float w_val = hal::AlienOps::unpack_apot(p_w_packed[c]);
                                                pixel_buffer[c] += p_in[c] * static_cast<T>(w_val);
                                            }
                                        } else {
                                            // Standard Float Path (Baseline)
                                            // Directly read float weights (higher bandwidth)
                                            const T* p_w = sw_ptr + k_idx * channels_;
                                            for(size_t c=0; c<C; ++c) {
                                                // Simulated standard convolution accumulation
                                                // lut_mul check: In float baseline, it's just multiply.
                                                // We use lut_mul here to keep logic consistent if it did something special,
                                                // but for APoT comparison we want raw float multiply.
                                                // Actually lut_mul is just multiply in current implementation.
                                                pixel_buffer[c] += p_in[c] * p_w[c];
                                            }
                                        }
                                    }
                                }
                            }

                            // 3. The Mixer (Spectral Engine) -> In-Place on pixel_buffer

                            // a. Soft Permutation (Ghost Phase)
                            // If float and C=64 on AVX2, use optimized gather
                            if (std::is_same<T, float>::value && C == 64) {
#if defined(DREIDEL_ARCH_AVX2)
                                hal::sparse_gather((const float*)pixel_buffer, perm_indices_.data(), (float*)temp_perm_buffer, C);

                                // b. In-Place FWHT (Register Mixer)
                                hal::x86::fwht64_avx2((float*)temp_perm_buffer);

                                // c. Learnable APoT Scaling (Bit-Shifts)
                                for(size_t i=0; i<C; ++i) {
                                    temp_perm_buffer[i] = lut_mul(temp_perm_buffer[i], effective_scales[i]);
                                }

                                // Phase 5: Branchless Gate
                                hal::AlienOps::branchless_relu((float*)temp_perm_buffer, C);

                                // Write to Output from temp_perm_buffer
                                T* p_out = out_ptr + ((n*H + h)*W + w)*C;
                                for(size_t c=0; c<C; ++c) {
                                    p_out[c] = temp_perm_buffer[c];
                                }

                                continue; // Skip the standard path
#endif
                            }

                            // Standard Path (Fallback)
                            for(size_t i=0; i<C; ++i) temp_perm_buffer[i] = pixel_buffer[i];
                            for(size_t i=0; i<C; ++i) {
                                pixel_buffer[i] = temp_perm_buffer[perm_indices_[i]];
                            }

                            // b. In-Place FWHT
                            algo::WHT::fwht_1d(pixel_buffer, C);

                            // c. Learnable APoT Scaling (Bit-Shifts)
                            for(size_t i=0; i<C; ++i) {
                                pixel_buffer[i] = lut_mul(pixel_buffer[i], effective_scales[i]);
                            }

                            // Phase 5: Branchless Gate (Standard Path)
                            if (std::is_same<T, float>::value) {
                                hal::AlienOps::branchless_relu((float*)pixel_buffer, C);
                            } else {
                                // Fallback
                                for(size_t i=0; i<C; ++i) if (pixel_buffer[i] < 0) pixel_buffer[i] = 0;
                            }

                            // Write to Output
                            T* p_out = out_ptr + ((n*H + h)*W + w)*C;
                            for(size_t c=0; c<C; ++c) {
                                p_out[c] = pixel_buffer[c];
                            }
                        }
                    }
                }
            }
        }

        input_ = input; // Cache for backward
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Simplified Backward for APoT/Spectral
        // Same as optimized implementation before

        auto shape = input_.shape();
        size_t batch = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3];

        Tensor<T> grad_input(shape);
        grad_input.fill(0);

        grad_spatial_.fill(0);
        grad_scales_.fill(0);

        T* gs_ptr = grad_spatial_.data();
        T* gscale_ptr = grad_scales_.data();
        const T* sw_ptr = spatial_weights_.data();
        const T* scale_ptr = spectral_scales_.data();
        int k_rad = kernel_size_ / 2;

        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();
        const T* in_ptr = input_.data();

        arena_.reset();
        T* d_vec = arena_.allocate<T>(C);
        T* eyes_out = arena_.allocate<T>(C);
        T* mixer_in = arena_.allocate<T>(C);
        T* d_unperm = arena_.allocate<T>(C);

        // Backward needs oracle mask too
        uint32_t oracle_mask = 0;
        if(use_gating_) {
            oracle_mask = hal::AlienOps::vec_sign_mask(oracle_projection_.data(), C);
        }

        for(size_t n=0; n<batch; ++n) {

            if (use_gating_) {
               size_t ch = H/2, cw = W/2;
               const T* p_center = in_ptr + ((n*H + ch)*W + cw)*C;
               uint32_t input_mask = hal::AlienOps::vec_sign_mask(p_center, C);
               uint32_t xor_val = input_mask ^ oracle_mask;
               int dist = hal::AlienOps::popcnt32(xor_val);
               if (dist > (int)(C/2)) continue;
            }

            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {

                    const T* p_go = go_ptr + ((n*H + h)*W + w)*C;
                    for(size_t c=0; c<C; ++c) d_vec[c] = p_go[c];

                    // Recompute forward (Eyes Output)
                    for(size_t c=0; c<C; ++c) eyes_out[c] = 0;
                    for(int ky=-k_rad; ky<=k_rad; ++ky) {
                        for(int kx=-k_rad; kx<=k_rad; ++kx) {
                            int ih = h + ky;
                            int iw = w + kx;
                            if (ih>=0 && ih<H && iw>=0 && iw<W) {
                                const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                const T* p_w = sw_ptr + k_idx * channels_;
                                for(size_t c=0; c<C; ++c) {
                                    eyes_out[c] += lut_mul(p_in[c], p_w[c]);
                                }
                            }
                        }
                    }

                     for(size_t i=0; i<C; ++i) mixer_in[i] = eyes_out[perm_indices_[i]];
                    algo::WHT::fwht_1d(mixer_in, C);

                    for(size_t c=0; c<C; ++c) {
                        gscale_ptr[c] += d_vec[c] * mixer_in[c];
                        d_vec[c] *= scale_ptr[c];
                    }

                    algo::WHT::fwht_1d(d_vec, C);

                    for(size_t i=0; i<C; ++i) {
                        d_unperm[perm_indices_[i]] = d_vec[i];
                    }

                    for(int ky=-k_rad; ky<=k_rad; ++ky) {
                        for(int kx=-k_rad; kx<=k_rad; ++kx) {
                            int ih = h + ky;
                            int iw = w + kx;
                            if (ih>=0 && ih<H && iw>=0 && iw<W) {
                                int k_idx = (ky+k_rad)*kernel_size_ + (kx+k_rad);
                                T* p_gs = gs_ptr + k_idx * channels_;
                                const T* p_in = in_ptr + ((n*H + ih)*W + iw)*C;
                                const T* p_w = sw_ptr + k_idx * channels_;
                                T* p_gi = gi_ptr + ((n*H + ih)*W + iw)*C;

                                for(size_t c=0; c<C; ++c) {
                                    T dy = d_unperm[c];
                                    p_gs[c] += dy * p_in[c];
                                    p_gi[c] += dy * p_w[c];
                                }
                            }
                        }
                    }
                }
            }
        }

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&spatial_weights_, &spectral_scales_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_spatial_, &grad_scales_};
    }

    std::string name() const override { return "ZenithBlock"; }

private:
    size_t channels_;
    size_t kernel_size_;
    size_t spectral_dim_;

    // Core Unit: Memory Arena
    core::Arena arena_;

    // Config
    bool use_gating_;
    bool use_apot_;

    // Parameters
    Tensor<T> spatial_weights_; // (C, 1, K, K)
    Tensor<T> spectral_scales_; // (1, C)
    std::vector<int> perm_indices_;

    // Phase 6: Memory Footprint Reduction
    std::vector<int8_t> packed_weights_; // Compressed Spatial Weights

    // Oracle
    std::vector<T> oracle_projection_;
    int active_bucket_;

    // Gradients
    Tensor<T> grad_spatial_;
    Tensor<T> grad_scales_;

    // Cache
    Tensor<T> input_;
};

} // namespace layers
} // namespace dreidel
