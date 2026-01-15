#pragma once

#include "Layer.hpp"
#include "../core/Tensor.hpp"
#include "../kernels/AvxFwht.hpp"
#include "../kernels/FusedZenith.hpp"
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
          grad_gate_h_({(size_t)res_w}),
          grad_gate_w_({(size_t)res_h}),
          proj_conv_(channels, channels, 1, 1, 0), // 1x1 Conv
          scratch_buffer_({1, (size_t)res_h, (size_t)res_w, (size_t)channels}),
          scratch_x_({1, (size_t)res_h, (size_t)res_w, (size_t)channels}),
          saved_input_({1, (size_t)res_h, (size_t)res_w, (size_t)channels}) // Need to save input
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
        // Save Input for Backward
        size_t size = input.size();
        std::memcpy(saved_input_.data(), input.data(), size * sizeof(float));

        // Zero-Copy Path
        // 1. Copy Input to Scratch X (State)
        const float* in_ptr = input.data();
        float* x_ptr = scratch_x_.data();
        std::memcpy(x_ptr, in_ptr, size * sizeof(float));

        float* buf_ptr = scratch_buffer_.data();

        // 1. Horizontal Spectral Mix (Row-wise)
        // FWHT -> Gate -> IFWHT
        kernels::FWHT_Horizontal_AVX(x_ptr, h_, w_, channels_);

        // Save State After FWHT_H for backward gating?
        // To compute dGateH, we need Input_FWHT.
        // x_ptr is modified in place.
        // We need to save it?
        // Yes.
        // Allocate space for intermediates?
        // This makes ZenithNanoBlock heavy memory-wise during training.
        // 512x512 is 64 blocks of 64x64.
        // Wait, ZenithNano operates on a single 64x64x64 block (after S2).
        // S2 output is 64x64x64. (192->64 channels). Resolution is 64x64.
        // So 64x64x64 floats = 256K floats = 1MB.
        // Saving intermediates is cheap (1MB).

        // We need:
        // 1. Input (Saved)
        // 2. Input_FWHT_H (for GateH grad)
        // 3. Output_GateH_FWHT_H (After scale) -> Transpose input.
        // 4. Input_FWHT_W (for GateW grad)

        // Let's add saved buffers.
        if (saved_fwht_h_.size() == 0) saved_fwht_h_ = Tensor<float>(saved_input_.shape());
        if (saved_fwht_w_.size() == 0) saved_fwht_w_ = Tensor<float>(saved_input_.shape());

        std::memcpy(saved_fwht_h_.data(), x_ptr, size * sizeof(float)); // Save FWHT H state

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

        // Save FWHT W state
        std::memcpy(saved_fwht_w_.data(), buf_ptr, size * sizeof(float));

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

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        // Residual Gradient: dOut passes to dInput and dBranch
        // dInput = grad_output + dBranch
        // dBranch comes from proj_conv backward.

        // 1. Conv Backward
        Tensor<float> d_branch = proj_conv_.backward(grad_output);
        float* dx = d_branch.data(); // Modify in place
        size_t size = d_branch.size();

        // Backprop through Vertical Spectral Mix
        // Transpose Back -> IFWHT -> Gate -> FWHT -> Transpose

        float* buf_ptr = scratch_buffer_.data(); // Re-use scratch

        // Transpose [H,W] -> [W,H] (Since forward ended with Transpose [W,H]->[H,W])
        kernels::Transpose_Block_64x64(dx, buf_ptr, h_, w_, channels_);

        // Scale 1/H (Linear ops commute, but scale is on forward output)
        float scale_h = 1.0f / h_;
        #pragma omp parallel for
        for(size_t i=0; i<size; ++i) buf_ptr[i] *= scale_h;

        // FWHT (Symmetric)
        kernels::FWHT_Horizontal_AVX(buf_ptr, w_, h_, channels_);

        // Gate W Gradient: dGateW += sum(dBuf * Buf_FWHT)
        // Gate is broadcast over W and C.
        // Buf [W, H, C]. Gate [H].
        // dGate[h] = sum_{w, c} (dBuf[w,h,c] * Buf_FWHT[w,h,c])
        // Buf_FWHT is saved in saved_fwht_w_ [W, H, C].

        grad_gate_w_.fill(0);
        float* dgw_ptr = grad_gate_w_.data();
        const float* saved_w = saved_fwht_w_.data();

        #pragma omp parallel for
        for(size_t h_idx=0; h_idx<h_; ++h_idx) {
            float sum = 0.0f;
            for(size_t w_idx=0; w_idx<w_; ++w_idx) {
                const float* d_row = buf_ptr + (w_idx*h_ + h_idx)*channels_;
                const float* s_row = saved_w + (w_idx*h_ + h_idx)*channels_;
                for(size_t c=0; c<channels_; ++c) {
                    sum += d_row[c] * s_row[c];
                }
            }
            dgw_ptr[h_idx] = sum;
        }

        // Propagate dBuf: dBuf *= GateW
        kernels::ElementWiseMul_Broadcast(buf_ptr, gate_w_.data(), {(size_t)1, (size_t)w_, (size_t)h_, (size_t)channels_});

        // FWHT (Symmetric)
        kernels::FWHT_Horizontal_AVX(buf_ptr, w_, h_, channels_);

        // Transpose Back [W,H] -> [H,W]
        kernels::Transpose_Block_64x64(buf_ptr, dx, w_, h_, channels_);

        // Backprop through Horizontal Spectral Mix

        // Scale 1/W
        float scale_w = 1.0f / w_;
        #pragma omp parallel for
        for(size_t i=0; i<size; ++i) dx[i] *= scale_w;

        // FWHT
        kernels::FWHT_Horizontal_AVX(dx, h_, w_, channels_);

        // Gate H Gradient
        // dGateH[w] = sum_{h, c} (dx[h,w,c] * Saved_H[h,w,c])
        // Gate [W]. Broadcast over H, C.
        grad_gate_h_.fill(0);
        float* dgh_ptr = grad_gate_h_.data();
        const float* saved_h = saved_fwht_h_.data();

        #pragma omp parallel for
        for(size_t w_idx=0; w_idx<w_; ++w_idx) {
            float sum = 0.0f;
            for(size_t h_idx=0; h_idx<h_; ++h_idx) {
                const float* d_row = dx + (h_idx*w_ + w_idx)*channels_;
                const float* s_row = saved_h + (h_idx*w_ + w_idx)*channels_;
                for(size_t c=0; c<channels_; ++c) {
                    sum += d_row[c] * s_row[c];
                }
            }
            dgh_ptr[w_idx] = sum;
        }

        // Propagate dx: dx *= GateH
        kernels::ElementWiseMul_Broadcast(dx, gate_h_.data(), {(size_t)1, (size_t)h_, (size_t)w_, (size_t)channels_});

        // FWHT
        kernels::FWHT_Horizontal_AVX(dx, h_, w_, channels_);

        // Total Gradient: dx + grad_output (Residual)
        const float* go_ptr = grad_output.data();
        #pragma omp parallel for
        for(size_t i=0; i<size; ++i) {
            dx[i] += go_ptr[i];
        }

        return d_branch; // dx is in d_branch buffer
    }

    std::vector<Tensor<float>*> parameters() override {
        std::vector<Tensor<float>*> p = {&gate_h_, &gate_w_};
        auto cp = proj_conv_.parameters();
        p.insert(p.end(), cp.begin(), cp.end());
        return p;
    }

    std::vector<Tensor<float>*> gradients() override {
        std::vector<Tensor<float>*> g = {&grad_gate_h_, &grad_gate_w_};
        auto cg = proj_conv_.gradients();
        g.insert(g.end(), cg.begin(), cg.end());
        return g;
    }

    std::string name() const override { return "ZenithNanoBlock"; }

private:
    int channels_;
    int h_;
    int w_;

    Tensor<float> gate_h_;
    Tensor<float> gate_w_;
    Tensor<float> grad_gate_h_;
    Tensor<float> grad_gate_w_;

    OptimizedConv2D<float> proj_conv_;

    // Persistent scratch buffers
    Tensor<float> scratch_buffer_;
    Tensor<float> scratch_x_;

    // Training Buffers
    Tensor<float> saved_input_;
    Tensor<float> saved_fwht_h_;
    Tensor<float> saved_fwht_w_;
};

} // namespace layers
} // namespace dreidel
