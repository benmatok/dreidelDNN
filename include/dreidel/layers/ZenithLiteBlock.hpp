#pragma once

#include "Layer.hpp"
#include "../core/Tensor.hpp"
#include "../hal/x86.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <memory>
#include <cstring>
#include <chrono>
#include <iostream>

namespace dreidel {
namespace layers {

// ZenithLiteBlock: Designed for L1/L2 Cache residency and minimal RAM bandwidth.
// Structure:
// 1. Channel Compress: 1x1 Conv (Groups=4).
// 2. Row Spectral Mixer: FWHT (1D, H) -> Mul (Gate H) -> IFWHT (1D, H).
// 3. Col Spectral Mixer: FWHT (1D, W) -> Mul (Gate W) -> IFWHT (1D, W).
// 4. Channel Expand: 1x1 Conv (Groups=4).
// 5. Residual Add: Input + Output.
template <typename T>
class ZenithLiteBlock : public Layer<T> {
public:
    // Timers
    static inline double time_compress = 0;
    static inline double time_row_mix = 0;
    static inline double time_col_transpose = 0;
    static inline double time_col_mix = 0;
    static inline double time_col_transpose_back = 0;
    static inline double time_expand = 0;
    static inline double time_residual = 0;
    static inline long long count_ops = 0;

    static void reset_timers() {
        time_compress = 0;
        time_row_mix = 0;
        time_col_transpose = 0;
        time_col_mix = 0;
        time_col_transpose_back = 0;
        time_expand = 0;
        time_residual = 0;
        count_ops = 0;
    }

    static void print_timers() {
        std::cout << "ZenithLiteBlock Timers (ms):" << std::endl;
        std::cout << "  Compress:    " << time_compress * 1000.0 << std::endl;
        std::cout << "  Row Mix:     " << time_row_mix * 1000.0 << std::endl;
        std::cout << "  Col Transp:  " << time_col_transpose * 1000.0 << std::endl;
        std::cout << "  Col Mix:     " << time_col_mix * 1000.0 << std::endl;
        std::cout << "  Col TrBack:  " << time_col_transpose_back * 1000.0 << std::endl;
        std::cout << "  Expand:      " << time_expand * 1000.0 << std::endl;
        std::cout << "  Residual:    " << time_residual * 1000.0 << std::endl;
        double total = time_compress + time_row_mix + time_col_transpose + time_col_mix + time_col_transpose_back + time_expand + time_residual;
        std::cout << "  TOTAL:       " << total * 1000.0 << std::endl;
        std::cout << "  Data Flow (Transp/Res): " << (time_col_transpose + time_col_transpose_back + time_residual) * 1000.0 << " (" << (time_col_transpose + time_col_transpose_back + time_residual)/total*100.0 << "%)" << std::endl;
        std::cout << "  Compute (Conv/Mix):     " << (time_compress + time_row_mix + time_col_mix + time_expand) * 1000.0 << " (" << (time_compress + time_row_mix + time_col_mix + time_expand)/total*100.0 << "%)" << std::endl;
    }

    ZenithLiteBlock(size_t channels, size_t height, size_t width,
                    size_t compress_factor = 4, bool use_residual = true)
        : channels_(channels), height_(height), width_(width),
          compress_factor_(compress_factor), use_residual_(use_residual)
    {
        // Dimensions
        inner_channels_ = channels_ / compress_factor_;
        if (inner_channels_ == 0) inner_channels_ = 1;
        groups_ = 4; // As per spec

        // Ensure divisibility for groups
        if (channels_ % groups_ != 0) throw std::invalid_argument("Channels must be divisible by groups (4).");
        if (inner_channels_ % groups_ != 0) throw std::invalid_argument("Inner channels must be divisible by groups (4).");

        size_t c_in_g = channels_ / groups_;
        size_t c_mid_g = inner_channels_ / groups_;

        // Weights:
        // Compress: [Groups, C_mid_g, C_in_g] (flat) -> Size: Groups * C_mid_g * C_in_g
        compress_weights_ = Tensor<T>({groups_, c_mid_g, c_in_g});
        compress_bias_ = Tensor<T>({inner_channels_});

        // Expand: [Groups, C_in_g, C_mid_g] (flat)
        expand_weights_ = Tensor<T>({groups_, c_in_g, c_mid_g});
        expand_bias_ = Tensor<T>({channels_});

        // Gates: [Height] and [Width]
        gate_h_ = Tensor<T>({width_});
        gate_v_ = Tensor<T>({height_});

        initialize();
    }

    void initialize() {
        // Init weights with He/Xavier or similar
        // For now, simple random
        compress_weights_.random(0.0f, 0.05f);
        compress_bias_.fill(0.0f);
        expand_weights_.random(0.0f, 0.05f);
        expand_bias_.fill(0.0f);

        // Gates init to 1.0 (identity pass-through initially) or random small noise?
        // Zenith gates usually start random but close to 0 or 1.
        gate_h_.fill(1.0f); // / Width? No, FWHT is unnormalized usually.
        gate_v_.fill(1.0f);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Input: [N, H, W, C]
        // Verify dimensions
        if (input.shape()[3] != channels_) throw std::runtime_error("Channel mismatch");
        size_t N = input.shape()[0];

        // Phase 1: Compress
        // Input -> Compressed
        // We can write directly to a scratch buffer.
        Tensor<T> output({N, height_, width_, channels_});
        size_t num_pixels = N * height_ * width_;

        // Lazy Resize
        size_t required_size = num_pixels * inner_channels_;
        if (compressed_buf_.size() < required_size) {
            compressed_buf_.resize(required_size);
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // Call Optimized Group Conv
        hal::x86::group_conv_1x1_avx2(
            input.data(),
            compressed_buf_.data(),
            compress_weights_.data(),
            compress_bias_.data(),
            num_pixels,
            channels_,
            inner_channels_,
            groups_
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        time_compress += std::chrono::duration<double>(t1 - t0).count();

        // Phase 2: Row Spectral Mixer (Horizontal)
        // Operate on W dimension.
        // Layout is NHWC.
        // We have N*H rows of length W. Each point has InnerC channels.
        // Iterate N, H.
        // Pointer to start of row: &buf[ (n*H + h)*W*InnerC ]
        // Length W. Channels InnerC.

        // Pre-scale gates?
        // IFWHT(FWHT(x)) = N * x.
        // We need 1/N scaling.
        // We can fuse it into Gate.
        // GateH_effective = GateH / W.
        // GateV_effective = GateV / H.

        // Make local copy of gates scaled? Or modify in place if constant?
        // Better to use a temporary scaled gate.
        std::vector<T> scaled_gate_h(width_);
        T scale_w = 1.0f / static_cast<T>(width_);
        for(size_t i=0; i<width_; ++i) scaled_gate_h[i] = gate_h_[i] * scale_w;

        // Since we process rows independently, we can parallelize.
        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<height_; ++h) {
                T* row_ptr = compressed_buf_.data() + ((n*height_ + h)*width_) * inner_channels_;

                // FWHT
                hal::x86::fwht_1d_vectorized_avx2(row_ptr, width_, inner_channels_);

                // Gate
                hal::x86::spectral_gate_separable_avx2(row_ptr, scaled_gate_h.data(), width_, inner_channels_);

                // IFWHT (Same as FWHT for Walsh-Hadamard)
                hal::x86::fwht_1d_vectorized_avx2(row_ptr, width_, inner_channels_);
            }
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        time_row_mix += std::chrono::duration<double>(t2 - t1).count();

        // Phase 3: Col Spectral Mixer (Vertical)
        // Lazy Resize Transpose Buffer
        if (transposed_buf_.size() < required_size) {
            transposed_buf_.resize(required_size);
        }

        // Parallel Transpose (Blocked)
        // Block size for L1 cache (e.g. 32x32 tiles of vectors? or just spatial tiles)
        const size_t TILE = 32;
        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t h0=0; h0<height_; h0+=TILE) {
                for(size_t w0=0; w0<width_; w0+=TILE) {
                    size_t h_end = std::min(h0 + TILE, height_);
                    size_t w_end = std::min(w0 + TILE, width_);

                    for(size_t h=h0; h<h_end; ++h) {
                        for(size_t w=w0; w<w_end; ++w) {
                            const T* src = compressed_buf_.data() + ((n*height_ + h)*width_ + w)*inner_channels_;
                            T* dst = transposed_buf_.data() + ((n*width_ + w)*height_ + h)*inner_channels_;
                            // Small memcpy is okay if in L1. inner_channels_ is likely 16-128 floats (64-512 bytes).
                            std::memcpy(dst, src, inner_channels_ * sizeof(T));
                        }
                    }
                }
            }
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        time_col_transpose += std::chrono::duration<double>(t3 - t2).count();

        // Now apply Col Mixer (on H dimension, which is now dense-spatial)
        std::vector<T> scaled_gate_v(height_);
        T scale_h = 1.0f / static_cast<T>(height_);
        for(size_t i=0; i<height_; ++i) scaled_gate_v[i] = gate_v_[i] * scale_h;

        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t w=0; w<width_; ++w) {
                 T* col_ptr = transposed_buf_.data() + ((n*width_ + w)*height_) * inner_channels_;

                 hal::x86::fwht_1d_vectorized_avx2(col_ptr, height_, inner_channels_);
                 hal::x86::spectral_gate_separable_avx2(col_ptr, scaled_gate_v.data(), height_, inner_channels_);
                 hal::x86::fwht_1d_vectorized_avx2(col_ptr, height_, inner_channels_);
            }
        }
        auto t4 = std::chrono::high_resolution_clock::now();
        time_col_mix += std::chrono::duration<double>(t4 - t3).count();

        // Transpose back: [N, W, H, C] -> [N, H, W, C]
        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t w0=0; w0<width_; w0+=TILE) {
                for(size_t h0=0; h0<height_; h0+=TILE) {
                    size_t w_end = std::min(w0 + TILE, width_);
                    size_t h_end = std::min(h0 + TILE, height_);

                    for(size_t w=w0; w<w_end; ++w) {
                        for(size_t h=h0; h<h_end; ++h) {
                            const T* src = transposed_buf_.data() + ((n*width_ + w)*height_ + h)*inner_channels_;
                            T* dst = compressed_buf_.data() + ((n*height_ + h)*width_ + w)*inner_channels_;
                            std::memcpy(dst, src, inner_channels_ * sizeof(T));
                        }
                    }
                }
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        time_col_transpose_back += std::chrono::duration<double>(t5 - t4).count();

        // Phase 4: Channel Expand
        hal::x86::group_conv_1x1_avx2(
            compressed_buf_.data(),
            output.data(),
            expand_weights_.data(),
            expand_bias_.data(),
            num_pixels,
            inner_channels_,
            channels_,
            groups_
        );
        auto t6 = std::chrono::high_resolution_clock::now();
        time_expand += std::chrono::duration<double>(t6 - t5).count();

        // Phase 5: Residual Add
        if (use_residual_) {
             // In-place add input to output
             // output += input
             size_t total_size = output.size();
             T* out_d = output.data();
             const T* in_d = input.data();

             #pragma omp parallel for
             for(size_t i=0; i<total_size; ++i) {
                 out_d[i] += in_d[i];
             }
        }
        auto t7 = std::chrono::high_resolution_clock::now();
        time_residual += std::chrono::duration<double>(t7 - t6).count();

        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Placeholder for backward.
        // Implementing full backward for ZenithLite is complex and maybe not needed if training is Python.
        // But for completeness, we return zeros or throw.
        throw std::runtime_error("ZenithLiteBlock backward not implemented yet.");
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&compress_weights_, &compress_bias_, &expand_weights_, &expand_bias_, &gate_h_, &gate_v_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {}; // Not implemented
    }

    std::string name() const override { return "ZenithLiteBlock"; }

    // Setters for weights (for loading from Python)
    Tensor<T>& get_gate_h() { return gate_h_; }
    Tensor<T>& get_gate_v() { return gate_v_; }
    Tensor<T>& get_compress_weights() { return compress_weights_; }
    Tensor<T>& get_compress_bias() { return compress_bias_; }
    Tensor<T>& get_expand_weights() { return expand_weights_; }
    Tensor<T>& get_expand_bias() { return expand_bias_; }

private:
    size_t channels_;
    size_t height_;
    size_t width_;
    size_t compress_factor_;
    size_t inner_channels_;
    size_t groups_;
    bool use_residual_;

    Tensor<T> compress_weights_; // [Groups, Inner/G, In/G]
    Tensor<T> compress_bias_;    // [Inner]

    Tensor<T> expand_weights_;   // [Groups, Out/G, Inner/G]
    Tensor<T> expand_bias_;      // [Out]

    Tensor<T> gate_h_; // [Width]
    Tensor<T> gate_v_; // [Height]

    // Persistent Scratchpads (avoid re-alloc)
    std::vector<T, core::AlignedAllocator<T>> compressed_buf_;
    std::vector<T, core::AlignedAllocator<T>> transposed_buf_;
};

} // namespace layers
} // namespace dreidel
