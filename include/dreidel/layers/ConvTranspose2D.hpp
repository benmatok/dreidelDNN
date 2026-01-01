#ifndef DREIDEL_LAYERS_CONVTRANSPOSE2D_HPP
#define DREIDEL_LAYERS_CONVTRANSPOSE2D_HPP

#include "Layer.hpp"
#include "Conv2D.hpp"
#include "PixelShuffle.hpp"
#include <vector>
#include <memory>
#include <string>

namespace dreidel {
namespace layers {

// ConvTranspose2D
// A simplified implementation.
// If kernel_size == stride, we can use the efficient Conv2d 1x1 + PixelShuffle approach.
// Otherwise, we might need a full implementation (which is complex).
// For the requested task (k=32, s=32), this optimization works perfectly.
// We will implement the generic interface but throw if k != s for now to save time/complexity,
// as the prompt specifically requests k=32, s=32.

template <typename T>
class ConvTranspose2D : public Layer<T> {
public:
    ConvTranspose2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), stride_(stride), padding_(padding)
    {
        if (kernel_size != stride) {
            throw std::runtime_error("ConvTranspose2D currently only supports kernel_size == stride (no overlap).");
        }
        if (padding != 0) {
            throw std::runtime_error("ConvTranspose2D currently does not support padding.");
        }

        // Implementation Strategy:
        // Input (N, H, W, C_in)
        // 1. Conv2D (1x1): C_in -> C_out * stride * stride
        // 2. PixelShuffle (stride): (N, H, W, C_out * s^2) -> (N, H*s, W*s, C_out)

        size_t intermediate_channels = out_channels * stride * stride;

        // 1x1 Conv
        conv_ = std::make_unique<Conv2D<T>>(in_channels, intermediate_channels, 1, 1, 0);

        // PixelShuffle
        shuffle_ = std::make_unique<PixelShuffle<T>>(stride);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        Tensor<T> inter = conv_->forward(input);
        return shuffle_->forward(inter);
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> grad_inter = shuffle_->backward(grad_output);
        return conv_->backward(grad_inter);
    }

    std::vector<Tensor<T>*> parameters() override {
        return conv_->parameters();
    }

    std::vector<Tensor<T>*> gradients() override {
        return conv_->gradients();
    }

    std::string name() const override { return "ConvTranspose2D"; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;

    std::unique_ptr<Conv2D<T>> conv_;
    std::unique_ptr<PixelShuffle<T>> shuffle_;
};

} // namespace layers
} // namespace dreidel

#endif
