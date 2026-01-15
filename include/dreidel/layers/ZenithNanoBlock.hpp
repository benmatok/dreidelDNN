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
          scratch_buffer_({1, (size_t)res_h, (size_t)res_w, (size_t)channels}),
          scratch_x_({1, (size_t)res_h, (size_t)res_w, (size_t)channels})
    {
        // Initialize gates to 1.0 (Identity)
        gate_h_.fill(1.0f);
        gate_w_.fill(1.0f);
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        // Fallback to alloc version
        Tensor<float> output({1, (size_t)h_, (size_t)w_, (size_t)channels_});
        forward(input, output);
        return output;
    }

    void forward(const Tensor<float>& input, Tensor<float>& output) override {
        // Zero-Copy Path
        // 1. Copy Input to Scratch X (State)
        const float* in_ptr = input.data();
        float* x_ptr = scratch_x_.data();
        size_t size = scratch_x_.size();
        std::memcpy(x_ptr, in_ptr, size * sizeof(float));

        float* buf_ptr = scratch_buffer_.data();

        // 1. Horizontal Spectral Mix (Row-wise)
        // FWHT -> Gate -> IFWHT
        kernels::FWHT_Horizontal_AVX(x_ptr, h_, w_, channels_);
        kernels::ElementWiseMul_Broadcast(x_ptr, gate_h_.data(), {(size_t)1, (size_t)h_, (size_t)w_, (size_t)channels_});
        kernels::FWHT_Horizontal_AVX(x_ptr, h_, w_, channels_); // Inverse (Symmetric)

        // Scale by 1/W
        float scale_w = 1.0f / w_;
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
        // Conv1x1(x) -> Output
        proj_conv_.forward(scratch_x_, output);

        // Residual connection: output += input
        float* out_ptr = output.data();
        const float* in_orig = input.data();

        #pragma omp parallel for
        for(size_t i=0; i<size; ++i) {
            out_ptr[i] += in_orig[i];
        }
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

    // Persistent scratch buffers
    Tensor<float> scratch_buffer_;
    Tensor<float> scratch_x_;
};

} // namespace layers
} // namespace dreidel
