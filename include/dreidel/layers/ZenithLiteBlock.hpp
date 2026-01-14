#pragma once

#include "Layer.hpp"
#include "../core/Tensor.hpp"
#include "../hal/x86.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <memory>
#include <cstring>

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
        // Row Mixer (Along Width): Gate H? Wait.
        // Spec: "Row Spectral Mixer: FWHT (1D, Horizontal) -> Mul (Gate H) -> IFWHT"
        // Horizontal is Width. So Gate should correspond to Width frequencies?
        // Prompt says "Mul (Gate H)". Usually H stands for Height?
        // Let's assume Gate H is for Horizontal (Width) dimension, size W.
        // And Gate V (Col) is for Vertical (Height) dimension, size H.
        // Re-reading spec:
        // "Row Spectral Mixer: FWHT (1D, Horizontal) ... Mul (Gate H)" -> Gate Horizontal? Size W?
        // "Col Spectral Mixer: FWHT (1D, Vertical) ... Mul (Gate V)" -> Gate Vertical? Size H?
        // I will name them gate_h_ (size Width) and gate_v_ (size Height).

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

        // Scratchpad allocation (per thread if parallel, or just use vector)
        // We need buffers for Compressed [N, H, W, InnerC]
        // Ideally we reuse memory.

        Tensor<T> output({N, height_, width_, channels_});

        // Temporary buffer for Compressed State
        // Size: N * H * W * InnerC
        // If N is large, this is big. But ZenithLite is for low memory.
        // Can we process pixel by pixel?
        // Conv 1x1 is pixel-wise.
        // Row Mixer needs full Row. Col Mixer needs full Col.
        // So we need at least H*W*InnerC buffer per image.

        // Phase 1: Compress
        // Input -> Compressed
        // We can write directly to a scratch buffer.
        size_t num_pixels = N * height_ * width_;
        std::vector<T, core::AlignedAllocator<T>> compressed_buf(num_pixels * inner_channels_);

        // Call Optimized Group Conv
        // Compress Weights: [Groups, Cout/G, Cin/G]
        // compress_weights_ shape is {Groups, c_mid_g, c_in_g}. Matches expected layout for helper.
        hal::x86::group_conv_1x1_avx2(
            input.data(),
            compressed_buf.data(),
            compress_weights_.data(),
            compress_bias_.data(),
            num_pixels,
            channels_,
            inner_channels_,
            groups_
        );

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
                T* row_ptr = compressed_buf.data() + ((n*height_ + h)*width_) * inner_channels_;

                // FWHT
                hal::x86::fwht_1d_vectorized_avx2(row_ptr, width_, inner_channels_);

                // Gate
                hal::x86::spectral_gate_separable_avx2(row_ptr, scaled_gate_h.data(), width_, inner_channels_);

                // IFWHT (Same as FWHT for Walsh-Hadamard)
                hal::x86::fwht_1d_vectorized_avx2(row_ptr, width_, inner_channels_);
            }
        }

        // Phase 3: Col Spectral Mixer (Vertical)
        // Operate on H dimension.
        // This is tricky for NHWC layout because H is strided by W*C.
        // Stride = W * InnerC.
        // Our fwht_1d_vectorized_avx2 assumes contiguous spatial dimension (Stride C).
        // Here "Spatial" is H, but elements are separated.
        // We must Transpose!
        // Transpose H and W?
        // Block Transpose?
        // Or implement Strided FWHT?
        // Transposing the whole [H, W, C] to [W, H, C] allows using the same kernel.
        // Transpose cost vs Strided access cost?
        // Strided access with stride W*C (where W=128, C=32 -> 4096 floats) is cache killer.
        // Transpose is better.

        // Transpose compressed_buf [N, H, W, C] -> [N, W, H, C]
        // This puts H in the contiguous-like dimension (stride C).
        // Then we apply FWHT on H.

        // Allocate transpose buffer? Or in-place transpose if H==W?
        // If H!=W, need buffer.
        // Even if H==W, in-place transpose of blocks of C?
        // Let's allocate a buffer. It's ZenithLite, we want low memory, but transpose needs auxiliary.
        // Or we can process by blocks (Tile) to stay in L1.

        std::vector<T, core::AlignedAllocator<T>> transposed_buf(num_pixels * inner_channels_);

        // Parallel Transpose
        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t h=0; h<height_; ++h) {
                for(size_t w=0; w<width_; ++w) {
                    // Src: n, h, w. Dest: n, w, h
                    const T* src = compressed_buf.data() + ((n*height_ + h)*width_ + w)*inner_channels_;
                    T* dst = transposed_buf.data() + ((n*width_ + w)*height_ + h)*inner_channels_;
                    std::memcpy(dst, src, inner_channels_ * sizeof(T));
                }
            }
        }

        // Now apply Col Mixer (on H dimension, which is now dense-spatial)
        std::vector<T> scaled_gate_v(height_);
        T scale_h = 1.0f / static_cast<T>(height_);
        for(size_t i=0; i<height_; ++i) scaled_gate_v[i] = gate_v_[i] * scale_h;

        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t w=0; w<width_; ++w) {
                 T* col_ptr = transposed_buf.data() + ((n*width_ + w)*height_) * inner_channels_;

                 hal::x86::fwht_1d_vectorized_avx2(col_ptr, height_, inner_channels_);
                 hal::x86::spectral_gate_separable_avx2(col_ptr, scaled_gate_v.data(), height_, inner_channels_);
                 hal::x86::fwht_1d_vectorized_avx2(col_ptr, height_, inner_channels_);
            }
        }

        // Transpose back: [N, W, H, C] -> [N, H, W, C]
        // We can write directly to compressed_buf again.
        #pragma omp parallel for collapse(2)
        for(size_t n=0; n<N; ++n) {
            for(size_t w=0; w<width_; ++w) {
                for(size_t h=0; h<height_; ++h) {
                    const T* src = transposed_buf.data() + ((n*width_ + w)*height_ + h)*inner_channels_;
                    T* dst = compressed_buf.data() + ((n*height_ + h)*width_ + w)*inner_channels_;
                    std::memcpy(dst, src, inner_channels_ * sizeof(T));
                }
            }
        }

        // Phase 4: Channel Expand
        // Compressed -> Output
        // Expand Weights: [Groups, Cout/G, Cin/G]
        // Note: For expand, Cin is InnerC, Cout is Channels.
        // expand_weights_ is {Groups, c_in_g, c_mid_g}.
        // Wait, helper expects [Groups, Cout/G, Cin/G].
        // Cout/G = Channels/4. Cin/G = InnerC/4.
        // My definition: expand_weights_ = {groups_, c_in_g, c_mid_g}.
        // c_in_g = channels_/groups = Cout/G.
        // c_mid_g = inner_channels_/groups = Cin/G.
        // So dimensions match.

        hal::x86::group_conv_1x1_avx2(
            compressed_buf.data(),
            output.data(),
            expand_weights_.data(),
            expand_bias_.data(),
            num_pixels,
            inner_channels_,
            channels_,
            groups_
        );

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
};

} // namespace layers
} // namespace dreidel
