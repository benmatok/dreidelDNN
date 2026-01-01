#ifndef DREIDEL_LAYERS_PIXELSHUFFLE_HPP
#define DREIDEL_LAYERS_PIXELSHUFFLE_HPP

#include "Layer.hpp"
#include <vector>
#include <string>
#include <stdexcept>

namespace dreidel {
namespace layers {

// PixelShuffle: Rearranges elements in a tensor of shape (*, C*r^2, H, W) to (*, C, H*r, W*r)
// But wait, our Tensor layout seems to be NHWC based on Conv2D and ZenithBlock.
// (N, H, W, C).
// Let's verify ZenithBlock: N, H, W, C.
// So PixelShuffle for NHWC:
// Input: (N, H, W, C*r^2) -> Output: (N, H*r, W*r, C)
//
// PyTorch PixelShuffle expects NCHW.
// NCHW: (N, C*r^2, H, W) -> (N, C, H*r, W*r).
//
// Let's stick to NHWC which seems to be the convention here.
// Logic for NHWC PixelShuffle (upscale):
// We have r*r channels per pixel. We want to spread them into an r x r block.
// Input[n, h, w, c_out * r * r + ry * r + rx]  --> Output[n, h*r + ry, w*r + rx, c_out]
// Or some permutation thereof.
// PyTorch: Channel dimension is split.
// C_in = C_out * r * r.
// Reshape to (C_out, r, r).
// Output[n, c, h*r + ry, w*r + rx] = Input[n, c*r*r + ry*r + rx, h, w] (conceptually)
//
// For NHWC:
// Input (N, H, W, C_in). C_in = C_out * r * r.
// We want Output (N, H*r, W*r, C_out).
//
// Mapping:
// For each n, h, w:
//   For each c_out:
//     For each ry in 0..r-1:
//       For each rx in 0..r-1:
//         // Which input channel maps to this spatial offset?
//         // Usually, channels are [c0_00, c0_01, c0_10, c0_11, c1_00...]
//         // c_in = c_out * (r*r) + ry*r + rx
//         // output[n, h*r+ry, w*r+rx, c_out] = input[n, h, w, c_in]

template <typename T>
class PixelShuffle : public Layer<T> {
public:
    PixelShuffle(size_t upscale_factor) : upscale_factor_(upscale_factor) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        // Assume NHWC
        auto shape = input.shape();
        if (shape.size() != 4) throw std::invalid_argument("PixelShuffle expects 4D input (N, H, W, C)");

        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C_in = shape[3];
        size_t r = upscale_factor_;

        if (C_in % (r * r) != 0) {
            throw std::invalid_argument("Input channels must be divisible by upscale_factor^2");
        }
        size_t C_out = C_in / (r * r);
        size_t H_out = H * r;
        size_t W_out = W * r;

        Tensor<T> output({N, H_out, W_out, C_out});
        const T* in_ptr = input.data();
        T* out_ptr = output.data();

        // Parallelize
        #pragma omp parallel for collapse(3)
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    for (size_t c = 0; c < C_out; ++c) {
                        for (size_t ry = 0; ry < r; ++ry) {
                            for (size_t rx = 0; rx < r; ++rx) {
                                size_t c_in = c * (r * r) + ry * r + rx;
                                size_t h_out_idx = h * r + ry;
                                size_t w_out_idx = w * r + rx;

                                T val = in_ptr[((n * H + h) * W + w) * C_in + c_in];
                                out_ptr[((n * H_out + h_out_idx) * W_out + w_out_idx) * C_out + c];
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Just the inverse operation (PixelUnshuffle logic)
        auto g_shape = grad_output.shape();
        size_t N = g_shape[0];
        size_t H_out = g_shape[1];
        size_t W_out = g_shape[2];
        size_t C_out = g_shape[3];
        size_t r = upscale_factor_;

        size_t H = H_out / r;
        size_t W = W_out / r;
        size_t C_in = C_out * r * r;

        Tensor<T> grad_input({N, H, W, C_in});
        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();

        #pragma omp parallel for collapse(3)
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    for (size_t c = 0; c < C_out; ++c) {
                        for (size_t ry = 0; ry < r; ++ry) {
                            for (size_t rx = 0; rx < r; ++rx) {
                                size_t c_in = c * (r * r) + ry * r + rx;
                                size_t h_out_idx = h * r + ry;
                                size_t w_out_idx = w * r + rx;

                                T val = go_ptr[((n * H_out + h_out_idx) * W_out + w_out_idx) * C_out + c];
                                gi_ptr[((n * H + h) * W + w) * C_in + c_in] = val;
                            }
                        }
                    }
                }
            }
        }
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }
    std::string name() const override { return "PixelShuffle"; }

private:
    size_t upscale_factor_;
};

// PixelUnshuffle: Rearranges elements in a tensor of shape (*, C, H*r, W*r) to (*, C*r^2, H, W)
// NHWC Version: (N, H*r, W*r, C) -> (N, H, W, C*r^2)
template <typename T>
class PixelUnshuffle : public Layer<T> {
public:
    PixelUnshuffle(size_t downscale_factor) : downscale_factor_(downscale_factor) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        auto shape = input.shape();
        if (shape.size() != 4) throw std::invalid_argument("PixelUnshuffle expects 4D input (N, H, W, C)");

        size_t N = shape[0];
        size_t H_in = shape[1];
        size_t W_in = shape[2];
        size_t C_in = shape[3];
        size_t r = downscale_factor_;

        if (H_in % r != 0 || W_in % r != 0) {
            throw std::invalid_argument("Input spatial dimensions must be divisible by downscale_factor");
        }

        size_t H_out = H_in / r;
        size_t W_out = W_in / r;
        size_t C_out = C_in * r * r;

        Tensor<T> output({N, H_out, W_out, C_out});
        const T* in_ptr = input.data();
        T* out_ptr = output.data();

        // Map Input (N, H*r, W*r, C) -> Output (N, H, W, C*r^2)
        // input[n, h*r+ry, w*r+rx, c] -> output[n, h, w, c*(r*r) + ry*r + rx]

        #pragma omp parallel for collapse(3)
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H_out; ++h) {
                for (size_t w = 0; w < W_out; ++w) {
                    for (size_t c = 0; c < C_in; ++c) {
                        for (size_t ry = 0; ry < r; ++ry) {
                            for (size_t rx = 0; rx < r; ++rx) {
                                size_t h_in = h * r + ry;
                                size_t w_in = w * r + rx;
                                size_t c_out = c * (r * r) + ry * r + rx;

                                T val = in_ptr[((n * H_in + h_in) * W_in + w_in) * C_in + c];
                                out_ptr[((n * H_out + h) * W_out + w) * C_out + c_out] = val;
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Inverse is PixelShuffle
        auto g_shape = grad_output.shape();
        size_t N = g_shape[0];
        size_t H = g_shape[1];
        size_t W = g_shape[2];
        size_t C_out = g_shape[3]; // C_in * r * r
        size_t r = downscale_factor_;

        size_t H_in = H * r;
        size_t W_in = W * r;
        size_t C_in = C_out / (r*r);

        Tensor<T> grad_input({N, H_in, W_in, C_in});
        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();

        #pragma omp parallel for collapse(3)
        for (size_t n = 0; n < N; ++n) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    for (size_t c = 0; c < C_in; ++c) {
                         for (size_t ry = 0; ry < r; ++ry) {
                            for (size_t rx = 0; rx < r; ++rx) {
                                size_t h_in = h * r + ry;
                                size_t w_in = w * r + rx;
                                size_t c_out_idx = c * (r * r) + ry * r + rx;

                                T val = go_ptr[((n * H + h) * W + w) * C_out + c_out_idx];
                                gi_ptr[((n * H_in + h_in) * W_in + w_in) * C_in + c] = val;
                            }
                        }
                    }
                }
            }
        }
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }
    std::string name() const override { return "PixelUnshuffle"; }

private:
    size_t downscale_factor_;
};

} // namespace layers
} // namespace dreidel

#endif
