#pragma once

#include "../layers/Layer.hpp"
#include "../core/Tensor.hpp"
#include "../layers/ZenithNanoBlock.hpp"
#include "../layers/OptimizedConv2D.hpp"
#include "../kernels/AvxFwht.hpp"
#include <vector>
#include <memory>

namespace dreidel {
namespace models {

class ZenithNano : public layers::Layer<float> {
public:
    // Zenith-Nano:
    // Input 512x512x3.
    // S1: SpaceToDepth(8) -> 64x64x192.
    // S2: Conv1x1 192->64.
    // S3: 3x ZenithNanoBlock.
    // S4: Conv1x1 64->192.
    // S5: DepthToSpace(8) -> 512x512x3.

    ZenithNano() {
        // S2
        compress_ = std::make_unique<layers::OptimizedConv2D<float>>(192, 64, 1, 1, 0);

        // S3
        block1_ = std::make_unique<layers::ZenithNanoBlock>(64, 64, 64);
        block2_ = std::make_unique<layers::ZenithNanoBlock>(64, 64, 64);
        block3_ = std::make_unique<layers::ZenithNanoBlock>(64, 64, 64);

        // S4
        expand_ = std::make_unique<layers::OptimizedConv2D<float>>(64, 192, 1, 1, 0);
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        // S1: SpaceToDepth (Zero-Copy View / Shuffle)
        // Input: [1, 512, 512, 3]
        int H = 512;
        int W = 512;
        int C = 3;
        int block = 8;

        Tensor<float> s1_out({1, (size_t)H/block, (size_t)W/block, (size_t)C*block*block}); // 64x64x192
        kernels::SpaceToDepth_Shuffle(input.data(), s1_out.data(), H, W, C, block);

        // S2: Compress
        Tensor<float> s2_out = compress_->forward(s1_out);

        // S3: Zenith Blocks
        Tensor<float> x = block1_->forward(s2_out);
        x = block2_->forward(x);
        x = block3_->forward(x);

        // S4: Expand
        Tensor<float> s4_out = expand_->forward(x);

        // S5: DepthToSpace
        Tensor<float> output({1, (size_t)H, (size_t)W, (size_t)C});
        kernels::DepthToSpace_Shuffle(s4_out.data(), output.data(), H/block, W/block, C, block); // C here is target output channels (3)

        return output;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override { return grad_output; }

    std::vector<Tensor<float>*> parameters() override {
        std::vector<Tensor<float>*> p;
        auto add = [&](auto& l) {
            auto pp = l->parameters();
            p.insert(p.end(), pp.begin(), pp.end());
        };
        add(compress_);
        add(block1_); add(block2_); add(block3_);
        add(expand_);
        return p;
    }

    std::string name() const override { return "ZenithNano"; }

private:
    std::unique_ptr<layers::OptimizedConv2D<float>> compress_;
    std::unique_ptr<layers::ZenithNanoBlock> block1_;
    std::unique_ptr<layers::ZenithNanoBlock> block2_;
    std::unique_ptr<layers::ZenithNanoBlock> block3_;
    std::unique_ptr<layers::OptimizedConv2D<float>> expand_;
};

} // namespace models
} // namespace dreidel
