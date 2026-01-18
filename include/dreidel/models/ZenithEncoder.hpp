#pragma once

#include "../layers/Layer.hpp"
#include "../core/Tensor.hpp"
#include "../layers/ZenithNanoBlock.hpp"
#include "../layers/OptimizedConv2D.hpp"
#include "../kernels/AvxFwht.hpp"
#include <vector>
#include <memory>
#include <iostream>

namespace dreidel {
namespace models {

class ZenithEncoder : public layers::Layer<float> {
public:
    // Zenith Encoder:
    // Input: 512x512x3
    // S1: SpaceToDepth(8) -> 64x64x192
    // S2: Conv1x1 192->64
    // S3: 3x ZenithNanoBlock
    // Output: 64x64x64

    ZenithEncoder(size_t batch_size = 1) : batch_size_(batch_size) {
        // Arena Allocation
        // S1: 64*64*192
        // S2: 64*64*64
        // B1, B2, B3: 64*64*64
        size_t sz_s1 = batch_size * 64 * 64 * 192;
        size_t sz_s2 = batch_size * 64 * 64 * 64;
        size_t sz_b = batch_size * 64 * 64 * 64;

        size_t pad = 64;

        size_t off_s1 = 0;
        size_t off_s2 = off_s1 + sz_s1 + pad;
        size_t off_b1 = off_s2 + sz_s2 + pad;
        size_t off_b2 = off_b1 + sz_b + pad;
        size_t off_b3 = off_b2 + sz_b + pad; // Output is B3

        size_t total = off_b3 + sz_b + pad;

        arena_.resize(total);
        float* base = arena_.data();

        s1_out_ = Tensor<float>({batch_size, 64, 64, 192}, base + off_s1);
        s2_out_ = Tensor<float>({batch_size, 64, 64, 64}, base + off_s2);
        b1_out_ = Tensor<float>({batch_size, 64, 64, 64}, base + off_b1);
        b2_out_ = Tensor<float>({batch_size, 64, 64, 64}, base + off_b2);
        b3_out_ = Tensor<float>({batch_size, 64, 64, 64}, base + off_b3);

        compress_ = std::make_unique<layers::OptimizedConv2D<float>>(192, 64, 1, 1, 0);
        block1_ = std::make_unique<layers::ZenithNanoBlock>(64, 64, 64);
        block2_ = std::make_unique<layers::ZenithNanoBlock>(64, 64, 64);
        block3_ = std::make_unique<layers::ZenithNanoBlock>(64, 64, 64);
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        // S1: SpaceToDepth
        int H = 512;
        int W = 512;
        int C = 3;
        int block = 8;
        size_t in_stride = H * W * C;
        size_t out_stride = (H/block) * (W/block) * (C*block*block);

        float* in_ptr = const_cast<float*>(input.data());
        float* out_ptr = s1_out_.data();

        #pragma omp parallel for
        for(size_t n=0; n<batch_size_; ++n) {
            kernels::SpaceToDepth_Shuffle(in_ptr + n*in_stride, out_ptr + n*out_stride, H, W, C, block);
        }

        // S2: Compress
        compress_->forward(s1_out_, s2_out_);

        // S3: Blocks
        block1_->forward(s2_out_, b1_out_);
        block2_->forward(b1_out_, b2_out_);
        block3_->forward(b2_out_, b3_out_);

        return b3_out_;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        // grad_output is d_b3_out
        Tensor<float> d_b3_out = const_cast<Tensor<float>&>(grad_output); // Assuming it matches B3 shape

        Tensor<float> d_b2_out = block3_->backward(d_b3_out);
        Tensor<float> d_b1_out = block2_->backward(d_b2_out);
        Tensor<float> d_s2_out = block1_->backward(d_b1_out);
        Tensor<float> d_s1_out = compress_->backward(d_s2_out);

        // S1 Backward (DepthToSpace)
        int H = 512;
        int W = 512;
        int C = 3;
        int block = 8;

        Tensor<float> grad_input({batch_size_, (size_t)H, (size_t)W, (size_t)C});
        size_t ds1_stride = (H/block) * (W/block) * (C*block*block);
        size_t gi_stride = H * W * C;

        float* ds1_ptr = d_s1_out.data();
        float* gi_ptr = grad_input.data();

        #pragma omp parallel for
        for(size_t n=0; n<batch_size_; ++n) {
            kernels::DepthToSpace_Shuffle(ds1_ptr + n*ds1_stride, gi_ptr + n*gi_stride, H/block, W/block, C, block);
        }

        return grad_input;
    }

    std::vector<Tensor<float>*> parameters() override {
        std::vector<Tensor<float>*> p;
        auto add = [&](auto& l) {
            auto pp = l->parameters();
            p.insert(p.end(), pp.begin(), pp.end());
        };
        add(compress_);
        add(block1_); add(block2_); add(block3_);
        return p;
    }

    std::vector<Tensor<float>*> gradients() override {
        std::vector<Tensor<float>*> g;
        auto add = [&](auto& l) {
            auto gg = l->gradients();
            g.insert(g.end(), gg.begin(), gg.end());
        };
        add(compress_);
        add(block1_); add(block2_); add(block3_);
        return g;
    }

    void set_training(bool training) override {
        Layer::set_training(training);
        compress_->set_training(training);
        block1_->set_training(training);
        block2_->set_training(training);
        block3_->set_training(training);
    }

    std::string name() const override { return "ZenithEncoder"; }

private:
    std::unique_ptr<layers::OptimizedConv2D<float>> compress_;
    std::unique_ptr<layers::ZenithNanoBlock> block1_;
    std::unique_ptr<layers::ZenithNanoBlock> block2_;
    std::unique_ptr<layers::ZenithNanoBlock> block3_;

    Tensor<float> s1_out_;
    Tensor<float> s2_out_;
    Tensor<float> b1_out_;
    Tensor<float> b2_out_;
    Tensor<float> b3_out_;

    size_t batch_size_;
    std::vector<float, core::AlignedAllocator<float>> arena_;
};

} // namespace models
} // namespace dreidel
