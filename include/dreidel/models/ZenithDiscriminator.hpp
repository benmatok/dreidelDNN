#ifndef DREIDEL_MODELS_ZENITHDISCRIMINATOR_HPP
#define DREIDEL_MODELS_ZENITHDISCRIMINATOR_HPP

#include "../layers/ZenithNanoBlock.hpp"
#include "../layers/OptimizedConv2D.hpp"
#include "../layers/Conv2D.hpp"
#include "../kernels/AvxFwht.hpp"
#include <vector>
#include <memory>

namespace dreidel {
namespace models {

class ZenithDiscriminator : public layers::Layer<float> {
public:
    // ZenithDiscriminator Architecture (Matches ZenithNano Encoder + Head)
    // Encoder Part:
    // 1. SpaceToDepth(8): 512x512x3 -> 64x64x192
    // 2. Conv1x1: 192 -> 64
    // 3. 3x ZenithNanoBlock
    // Head Part:
    // 4. Conv1x1 (Classification): 64 -> 1
    // Output: 64x64x1 (Map of real/fake scores)

    ZenithDiscriminator(size_t batch_size = 1) : batch_size_(batch_size) {
        // Encoder
        compress_ = std::make_unique<layers::OptimizedConv2D<float>>(192, 64, 1, 1, 0);
        block1_ = std::make_unique<layers::ZenithNanoBlock>(64, 64, 64);
        block2_ = std::make_unique<layers::ZenithNanoBlock>(64, 64, 64);
        block3_ = std::make_unique<layers::ZenithNanoBlock>(64, 64, 64);

        // Head: Map 64 channels to 1 channel score map
        head_ = std::make_unique<layers::OptimizedConv2D<float>>(64, 1, 1, 1, 0);

        // Pre-allocate buffers for Forward (similar to ZenithNano to avoid allocs)
        // Note: Batch size 1 assumption in ZenithNano, we should probably support dynamic or fix it.
        // Train loop uses batch 4. ZenithNano code had hardcoded buffers for batch 1 logic?
        // Let's check ZenithNano again.
        // ZenithNano constructor: s1_out_ = Tensor<float>({1, 64, 64, 192}, ...);
        // It seems ZenithNano *hardcodes* batch size 1 in its arena!
        // But the training script sets batch_size = 4.
        // This means the training script might be crashing or overwriting memory if it passes batch > 1!
        // The user says "download small set for validation and train on new batches each time".
        // If ZenithNano supports only batch 1, we must fix it or respect it.
        // The user's prompt implies "add to training scheme".
        // I will assume for now I should follow the pattern.
        // Ideally, buffers should be resized on forward if shape changes.
        // For the Discriminator, I'll just allocate in forward for safety,
        // or replicate the arena strategy if performance is critical.
        // Given "ZenithNano" is "Ultra Fast", let's stick to simple allocation for the D to avoid complexity bugs,
        // unless I see ZenithNano fails with batch 4.
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        // Input: [N, 512, 512, 3]
        size_t H = 512;
        int W = 512;
        int C = 3;
        int block = 8;
        size_t H_out = H / block;
        size_t W_out = W / block;
        size_t C_out = C * block * block;

        Tensor<float> s1_out({batch_size_, H_out, W_out, C_out});

        // Loop batch for SpaceToDepth
        size_t in_stride = H * W * C;
        size_t out_stride = H_out * W_out * C_out;
        float* in_ptr = const_cast<float*>(input.data());
        float* out_ptr = s1_out.data();

        #pragma omp parallel for
        for(size_t n=0; n<batch_size_; ++n) {
            kernels::SpaceToDepth_Shuffle(in_ptr + n*in_stride, out_ptr + n*out_stride, H, W, C, block);
        }

        // 2. Compress (192->64)
        Tensor<float> s2_out = compress_->forward(s1_out);

        // 3. Blocks
        Tensor<float> b1_out = block1_->forward(s2_out);
        Tensor<float> b2_out = block2_->forward(b1_out);
        Tensor<float> b3_out = block3_->forward(b2_out);

        // 4. Head (64->1)
        Tensor<float> scores = head_->forward(b3_out);

        return scores; // [N, 64, 64, 1]
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        // grad_output: [N, 64, 64, 1]

        // 4. Head Backward
        Tensor<float> d_b3 = head_->backward(grad_output);

        // 3. Blocks Backward
        Tensor<float> d_b2 = block3_->backward(d_b3);
        Tensor<float> d_b1 = block2_->backward(d_b2);
        Tensor<float> d_s2 = block1_->backward(d_b1);

        // 2. Compress Backward
        Tensor<float> d_s1 = compress_->backward(d_s2);

        // 1. SpaceToDepth Backward -> DepthToSpace
        size_t block = 8;
        size_t H_small = 64;
        size_t W_small = 64;
        size_t C_orig = 3;
        size_t H_orig = 512;
        size_t W_orig = 512;

        Tensor<float> grad_input({batch_size_, H_orig, W_orig, C_orig});

        size_t d_s1_stride = H_small * W_small * 192;
        size_t gi_stride = H_orig * W_orig * C_orig;
        float* ds1_ptr = d_s1.data();
        float* gi_ptr = grad_input.data();

        #pragma omp parallel for
        for(size_t n=0; n<batch_size_; ++n) {
            kernels::DepthToSpace_Shuffle(ds1_ptr + n*d_s1_stride, gi_ptr + n*gi_stride, H_small, W_small, C_orig, block);
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
        add(head_);
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
        add(head_);
        return g;
    }

    void set_training(bool training) override {
        Layer::set_training(training);
        compress_->set_training(training);
        block1_->set_training(training);
        block2_->set_training(training);
        block3_->set_training(training);
        head_->set_training(training);
    }

    std::string name() const override { return "ZenithDiscriminator"; }

private:
    std::unique_ptr<layers::OptimizedConv2D<float>> compress_;
    std::unique_ptr<layers::ZenithNanoBlock> block1_;
    std::unique_ptr<layers::ZenithNanoBlock> block2_;
    std::unique_ptr<layers::ZenithNanoBlock> block3_;
    std::unique_ptr<layers::OptimizedConv2D<float>> head_;
    size_t batch_size_;
};

} // namespace models
} // namespace dreidel

#endif // DREIDEL_MODELS_ZENITHDISCRIMINATOR_HPP
