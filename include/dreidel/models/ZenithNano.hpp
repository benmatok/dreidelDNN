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
        // Implement Arena Allocator to avoid 4K Aliasing
        size_t sz_s1 = 64 * 64 * 192;
        size_t sz_s2 = 64 * 64 * 64;
        size_t sz_b = 64 * 64 * 64;
        size_t sz_s4 = 64 * 64 * 192;

        // Offset 64 floats (256 bytes) to break 4K alignment
        // If block sizes are multiples of 4K, adding 256 bytes ensures offsets differ.
        size_t pad = 64;

        size_t off_s1 = 0;
        size_t off_s2 = off_s1 + sz_s1 + pad;
        size_t off_b1 = off_s2 + sz_s2 + pad;
        size_t off_b2 = off_b1 + sz_b + pad;
        size_t off_b3 = off_b2 + sz_b + pad;
        size_t off_s4 = off_b3 + sz_b + pad;

        size_t total = off_s4 + sz_s4 + pad;

        arena_.resize(total);
        float* base = arena_.data();

        // Initialize Views
        s1_out_ = Tensor<float>({1, 64, 64, 192}, base + off_s1);
        s2_out_ = Tensor<float>({1, 64, 64, 64}, base + off_s2);
        b1_out_ = Tensor<float>({1, 64, 64, 64}, base + off_b1);
        b2_out_ = Tensor<float>({1, 64, 64, 64}, base + off_b2);
        b3_out_ = Tensor<float>({1, 64, 64, 64}, base + off_b3);
        s4_out_ = Tensor<float>({1, 64, 64, 192}, base + off_s4);

        // Check 4K Aliasing (Verify fix)
        auto check = [](const char* name, const Tensor<float>& t) {
            size_t addr = (size_t)t.data();
            std::cout << name << ": " << (void*)addr << " (Offset 4K: " << (addr % 4096) << ")" << std::endl;
        };
        check("s1_out", s1_out_);
        check("s2_out", s2_out_);
        check("b1_out", b1_out_);
        check("b2_out", b2_out_);
        check("b3_out", b3_out_);
        check("s4_out", s4_out_);

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

        // Use persistent buffer s1_out_
        kernels::SpaceToDepth_Shuffle(input.data(), s1_out_.data(), H, W, C, block);

        // S2: Compress
        compress_->forward(s1_out_, s2_out_);

        // S3: Zenith Blocks
        block1_->forward(s2_out_, b1_out_);
        block2_->forward(b1_out_, b2_out_);
        block3_->forward(b2_out_, b3_out_);

        // S4: Expand
        expand_->forward(b3_out_, s4_out_);

        // S5: DepthToSpace
        Tensor<float> output({1, (size_t)H, (size_t)W, (size_t)C});
        kernels::DepthToSpace_Shuffle(s4_out_.data(), output.data(), H/block, W/block, C, block); // C here is target output channels (3)

        return output;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        // S5 Backward: DepthToSpace_Shuffle Backward is SpaceToDepth_Shuffle
        // Output Grad: [N, H, W, 3] -> S4 Grad: [N, H/8, W/8, 192]
        // But s4_out_ is persistent, we need a gradient buffer for it?
        // Ideally we pass gradients back up.
        // Or we use persistent gradient buffers in arena?
        // For now, allocate on stack/heap (slow but works).

        int H = 512;
        int W = 512;
        int C = 3;
        int block = 8;

        Tensor<float> d_s4_out({1, 64, 64, 192});
        kernels::SpaceToDepth_Shuffle(grad_output.data(), d_s4_out.data(), H, W, C, block);

        // S4 Backward: Expand (Conv1x1)
        // d_b3_out = expand_back(d_s4_out)
        // Note: OptimizedConv2D backward returns grad_input.
        Tensor<float> d_b3_out = expand_->backward(d_s4_out);

        // S3 Backward: Blocks
        Tensor<float> d_b2_out = block3_->backward(d_b3_out);
        Tensor<float> d_b1_out = block2_->backward(d_b2_out);
        Tensor<float> d_s2_out = block1_->backward(d_b1_out);

        // S2 Backward: Compress
        Tensor<float> d_s1_out = compress_->backward(d_s2_out);

        // S1 Backward: SpaceToDepth Backward is DepthToSpace
        Tensor<float> grad_input({1, (size_t)H, (size_t)W, (size_t)C});
        kernels::DepthToSpace_Shuffle(d_s1_out.data(), grad_input.data(), H/block, W/block, C, block);

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
        add(expand_);
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
        add(expand_);
        return g;
    }

    std::string name() const override { return "ZenithNano"; }

private:
    std::unique_ptr<layers::OptimizedConv2D<float>> compress_;
    std::unique_ptr<layers::ZenithNanoBlock> block1_;
    std::unique_ptr<layers::ZenithNanoBlock> block2_;
    std::unique_ptr<layers::ZenithNanoBlock> block3_;
    std::unique_ptr<layers::OptimizedConv2D<float>> expand_;

    // Persistent buffers
    Tensor<float> s1_out_;
    Tensor<float> s2_out_;
    Tensor<float> b1_out_;
    Tensor<float> b2_out_;
    Tensor<float> b3_out_;
    Tensor<float> s4_out_;

    // Arena
    std::vector<float, core::AlignedAllocator<float>> arena_;
};

} // namespace models
} // namespace dreidel
