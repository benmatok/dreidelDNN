#pragma once

#include "../layers/Layer.hpp"
#include "../core/Tensor.hpp"
#include "../layers/OptimizedConv2D.hpp"
#include "../kernels/AvxFwht.hpp"
#include <vector>
#include <memory>

namespace dreidel {
namespace models {

class ZenithDecoder : public layers::Layer<float> {
public:
    // Zenith Decoder:
    // Input: 64x64x64
    // S4: Conv1x1 64->192
    // S5: DepthToSpace(8) -> 512x512x3
    // Output: 512x512x3

    ZenithDecoder(size_t batch_size = 1) : batch_size_(batch_size) {
        size_t sz_s4 = batch_size * 64 * 64 * 192;
        size_t pad = 64;

        arena_.resize(sz_s4 + pad);
        s4_out_ = Tensor<float>({batch_size, 64, 64, 192}, arena_.data());

        expand_ = std::make_unique<layers::OptimizedConv2D<float>>(64, 192, 1, 1, 0);
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        // Input is 64x64x64 (b3_out from encoder)
        expand_->forward(input, s4_out_);

        // S5: DepthToSpace
        int H = 512;
        int W = 512;
        int C = 3;
        int block = 8;
        size_t s4_stride = (H/block) * (W/block) * (C*block*block);
        size_t final_stride = H * W * C;

        Tensor<float> output({batch_size_, (size_t)H, (size_t)W, (size_t)C});
        float* s4_ptr = s4_out_.data();
        float* final_ptr = output.data();

        #pragma omp parallel for
        for(size_t n=0; n<batch_size_; ++n) {
            kernels::DepthToSpace_Shuffle(s4_ptr + n*s4_stride, final_ptr + n*final_stride, H/block, W/block, C, block);
        }

        return output;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        int H = 512;
        int W = 512;
        int C = 3;
        int block = 8;

        Tensor<float> d_s4_out({batch_size_, 64, 64, 192});

        size_t go_stride = H * W * C;
        size_t ds4_stride = (H/block) * (W/block) * (C*block*block);

        const float* go_ptr = grad_output.data();
        float* ds4_ptr = d_s4_out.data();

        #pragma omp parallel for
        for(size_t n=0; n<batch_size_; ++n) {
            kernels::SpaceToDepth_Shuffle(go_ptr + n*go_stride, ds4_ptr + n*ds4_stride, H, W, C, block);
        }

        return expand_->backward(d_s4_out);
    }

    std::vector<Tensor<float>*> parameters() override {
        return expand_->parameters();
    }

    std::vector<Tensor<float>*> gradients() override {
        return expand_->gradients();
    }

    void set_training(bool training) override {
        Layer::set_training(training);
        expand_->set_training(training);
    }

    std::string name() const override { return "ZenithDecoder"; }

private:
    std::unique_ptr<layers::OptimizedConv2D<float>> expand_;
    Tensor<float> s4_out_;
    size_t batch_size_;
    std::vector<float, core::AlignedAllocator<float>> arena_;
};

} // namespace models
} // namespace dreidel
