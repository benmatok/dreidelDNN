#pragma once

#include "Layer.hpp"
#include "../core/Tensor.hpp"
#include "../kernels/AvxFwht.hpp"
#include "../layers/OptimizedConv2D.hpp"
#include <vector>
#include <memory>

namespace dreidel {
namespace layers {

class ZenithNanoBlock : public Layer<float> {
public:
    // ZenithNanoBlock:
    // Designed for 64x64 resolution, 64 channels (Zenith-Nano "Brain" stage).
    // Uses kernels from AvxFwht.hpp.

    ZenithNanoBlock(int channels, int res_h, int res_w)
        : channels_(channels), h_(res_h), w_(res_w),
          gate_h_({(size_t)res_w}),
          gate_w_({(size_t)res_h}),
          proj_conv_(channels, channels, 1, 1, 0), // 1x1 Conv
          scratch_buffer_({1, (size_t)res_h, (size_t)res_w, (size_t)channels})
    {
        // Initialize gates to 1.0 (Identity)
        gate_h_.fill(1.0f);
        gate_w_.fill(1.0f);
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        // Input: [1, H, W, C]
        // We modify input in place? No, Layer::forward usually returns new tensor.
        // But for efficiency ("Zero-Copy"), we might want in-place.
        // Let's copy first, then modify 'x'.
        Tensor<float> x = input; // Copy

        float* x_ptr = x.data();
        float* buf_ptr = scratch_buffer_.data();

        // 1. Horizontal Spectral Mix (Row-wise)
        // FWHT -> Gate -> IFWHT
        kernels::FWHT_Horizontal_AVX(x_ptr, h_, w_, channels_);
        kernels::ElementWiseMul_Broadcast(x_ptr, gate_h_.data(), {(size_t)1, (size_t)h_, (size_t)w_, (size_t)channels_});
        kernels::FWHT_Horizontal_AVX(x_ptr, h_, w_, channels_); // Inverse (Symmetric)

        // Scale by 1/W
        float scale_w = 1.0f / w_;
        size_t size = x.size();
        #pragma omp parallel for
        for(size_t i=0; i<size; ++i) x_ptr[i] *= scale_w;

        // 2. Vertical Spectral Mix (Column-wise)
        // Transpose -> FWHT -> Gate -> IFWHT -> Transpose Back
        kernels::Transpose_Block_64x64(x_ptr, buf_ptr, h_, w_, channels_);

        // Now buf is [W, H, C]. Rows are length H.
        kernels::FWHT_Horizontal_AVX(buf_ptr, w_, h_, channels_);

        // Gate W acts on H dimension (which is now rows).
        kernels::ElementWiseMul_Broadcast(buf_ptr, gate_w_.data(), {(size_t)1, (size_t)w_, (size_t)h_, (size_t)channels_});

        kernels::FWHT_Horizontal_AVX(buf_ptr, w_, h_, channels_); // Inverse

        // Scale by 1/H
        float scale_h = 1.0f / h_;
        #pragma omp parallel for
        for(size_t i=0; i<size; ++i) buf_ptr[i] *= scale_h;

        kernels::Transpose_Block_64x64(buf_ptr, x_ptr, w_, h_, channels_);

        // 3. Channel Mixing (Standard GEMM) -> Residual
        // Conv1x1(x) + input
        // OptimizedConv2D returns new tensor.
        Tensor<float> proj = proj_conv_.forward(x);

        // Residual connection: input + proj
        // 'input' is the original input. 'proj' is the result of spectral+conv path.
        // Wait. "Input + Output".
        // The spectral path modifies 'x'.
        // Is the residual over the whole block?
        // Standard ResNet: y = Conv(x) + x.
        // ZenithBlock Plan: "Conv1x1_AddResidual(x, proj_weight, proj_bias)".
        // It seems the Conv1x1 is applied to the output of spectral mixing?
        // And then added to 'x' (which 'x'? The input to the block?).
        // Yes, "Residual Add: Input + Output" in ZenithLite.

        const float* in_orig = input.data();
        float* p_ptr = proj.data();

        #pragma omp parallel for
        for(size_t i=0; i<size; ++i) {
            p_ptr[i] += in_orig[i];
        }

        return proj;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override { return grad_output; }

    std::vector<Tensor<float>*> parameters() override {
        std::vector<Tensor<float>*> p = {&gate_h_, &gate_w_};
        auto cp = proj_conv_.parameters();
        p.insert(p.end(), cp.begin(), cp.end());
        return p;
    }

    std::string name() const override { return "ZenithNanoBlock"; }

private:
    int channels_;
    int h_;
    int w_;

    Tensor<float> gate_h_;
    Tensor<float> gate_w_;

    OptimizedConv2D<float> proj_conv_;

    // Persistent scratch buffer
    Tensor<float> scratch_buffer_;
};

} // namespace layers
} // namespace dreidel
