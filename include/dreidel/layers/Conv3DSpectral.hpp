#ifndef DREIDEL_LAYERS_CONV3D_SPECTRAL_HPP
#define DREIDEL_LAYERS_CONV3D_SPECTRAL_HPP

#include "Layer.hpp"
#include "LinearWHT.hpp"
#include <vector>
#include <iostream>

namespace dreidel {
namespace layers {

// Conv3DSpectral Layer
// Logic: DepthwiseConv3D (Spatial) -> LinearWHT (Mixing)
// Input: (Batch, D, H, W, C) or (Batch, C, D, H, W) -> Tensor checks for NHWC (Channel last).
// But for 3D, it is NDHWC.
// Our Tensor supports arbitrary dims.
// Let's assume input is (Batch, D, H, W, C).

template <typename T, BackendType B = BackendType::CPU>
class Conv3DSpectral : public Layer<T, B> {
public:
    // spatial_kernel_size: e.g. 3 (3x3x3)
    // channels: number of input/output channels (kept same for depthwise)
    Conv3DSpectral(size_t spatial_kernel_size, size_t channels, size_t d, size_t h, size_t w)
        : kernel_size_(spatial_kernel_size), channels_(channels),
          d_(d), h_(h), w_(w),
          linear_wht_(channels) // LinearWHT operates on the channel dimension?
          // If LinearWHT is "Mixing", it usually mixes channels (1x1 conv replacement).
          // If so, it operates on C.
          // FWHT requires C to be power of 2.
    {
        // Check power of 2 for channels
        if (channels == 0 || (channels & (channels - 1)) != 0) {
            throw std::invalid_argument("Channels must be a power of 2 for Conv3DSpectral (LinearWHT requirement).");
        }

        // Depthwise Kernel: (K, K, K, C) -> One kernel per channel.
        // Or (K, K, K, 1, C) depending on storage.
        // We store as (C, K, K, K) for easier depthwise loop? Or (K, K, K, C).
        // Let's stick to (K, K, K, C).
        size_t k = kernel_size_;
        spatial_weights_ = Tensor<T, B>({k, k, k, channels});

        // Initialize weights (Xavier-ish)
        spatial_weights_.random(0, std::sqrt(2.0 / (k*k*k*channels)));

        grad_spatial_weights_ = Tensor<T, B>({k, k, k, channels});
        grad_spatial_weights_.fill(0);
    }

    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        // Input: (Batch, D, H, W, C)
        input_ = input;

        // 1. Depthwise Conv3D
        // Naive implementation for now.
        // Output shape same as input (assuming padding 'same') or valid?
        // Let's assume 'same' padding to keep dimensions.

        size_t batch = input.shape()[0];
        size_t in_d = input.shape()[1];
        size_t in_h = input.shape()[2];
        size_t in_w = input.shape()[3];
        size_t c = input.shape()[4];

        if (c != channels_) throw std::invalid_argument("Input channels mismatch.");

        // Output of spatial conv
        Tensor<T, B> spatial_out(input.shape());
        spatial_out.fill(0);

        // Naive loop with OpenMP
        int k_half = kernel_size_ / 2;
        int k_size = kernel_size_;

        const T* in_ptr = input.data();
        T* out_ptr = spatial_out.data();
        const T* k_ptr = spatial_weights_.data();

        // This is extremely slow naively.
        // Optimization is mentioned "Fuse Spatial output to Mixing input in L1 cache".
        // That implies we should do Spatial then immediately WHT on the pixel/voxel.

        // Loop over batch
        #pragma omp parallel for
        for (long b = 0; b < (long)batch; ++b) {
            for (long z = 0; z < (long)in_d; ++z) {
                for (long y = 0; y < (long)in_h; ++y) {
                    for (long x = 0; x < (long)in_w; ++x) {

                        // Per pixel (z, y, x), we have C channels.
                        // We apply depthwise kernel.
                        // For each channel c_i, apply 3D kernel [:, :, :, c_i]

                        // To fuse, we compute the C values for this pixel,
                        // and then pass this C-vector to LinearWHT.

                        // Buffer for this pixel's channels
                        // But LinearWHT expects a full Tensor to perform FWHT on last dim.
                        // Since FWHT is element-wise over the outer dimensions,
                        // we can process the whole tensor.

                        // So:
                        // 1. Compute Spatial Conv for this pixel (all channels).
                        for (int c_i = 0; c_i < (int)c; ++c_i) {
                            T sum = 0;

                            // Kernel loop
                            for (int kz = 0; kz < k_size; ++kz) {
                                int in_z = z + kz - k_half;
                                if (in_z < 0 || in_z >= (int)in_d) continue;

                                for (int ky = 0; ky < k_size; ++ky) {
                                    int in_y = y + ky - k_half;
                                    if (in_y < 0 || in_y >= (int)in_h) continue;

                                    for (int kx = 0; kx < k_size; ++kx) {
                                        int in_x = x + kx - k_half;
                                        if (in_x < 0 || in_x >= (int)in_w) continue;

                                        // Indexing
                                        // Input: b, in_z, in_y, in_x, c_i
                                        // Strides: C, W*C, H*W*C, D*H*W*C
                                        size_t idx = ((b * in_d + in_z) * in_h + in_y) * in_w * c + in_x * c + c_i;

                                        // Kernel: kz, ky, kx, c_i
                                        size_t k_idx = ((kz * k_size + ky) * k_size + kx) * c + c_i;

                                        sum += in_ptr[idx] * k_ptr[k_idx];
                                    }
                                }
                            }

                            // Store result
                            size_t out_idx = ((b * in_d + z) * in_h + y) * in_w * c + x * c + c_i;
                            out_ptr[out_idx] = sum;
                        }
                    }
                }
            }
        }

        // 2. LinearWHT (Mixing)
        // LinearWHT forward applies D * x -> FWHT.
        // It operates on the last dimension (C).
        // Since spatial_out is (Batch, D, H, W, C), LinearWHT will treat (Batch*D*H*W) as batch size for itself.
        // But LinearWHT signature takes Tensor.
        // We can just call it.
        // Wait, LinearWHT stores 'input' for backward. We are passing a temporary 'spatial_out'.
        // This is fine, but in backward we need gradients through 'spatial_out'.

        return linear_wht_.forward(spatial_out);
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // grad_output: dL/dY (after WHT)

        // 1. Backward through LinearWHT
        // This gives dL/d(SpatialOut)
        Tensor<T, B> grad_spatial = linear_wht_.backward(grad_output);

        // 2. Backward through Depthwise Conv3D
        size_t batch = input_.shape()[0];
        size_t in_d = input_.shape()[1];
        size_t in_h = input_.shape()[2];
        size_t in_w = input_.shape()[3];
        size_t c = input_.shape()[4];

        Tensor<T, B> grad_input(input_.shape());
        grad_input.fill(0);

        // Naive Backward Loop
        // Very slow, but functional for correctness
        int k_half = kernel_size_ / 2;
        int k_size = kernel_size_;

        const T* grad_s_ptr = grad_spatial.data();
        const T* in_ptr = input_.data();
        const T* k_ptr = spatial_weights_.data();

        T* grad_in_ptr = grad_input.data();
        T* grad_w_ptr = grad_spatial_weights_.data(); // Accumulate here, but reset first?
        // Typically gradients are accumulated in optimizer or cleared before backward.
        // Here we assume cleared or we overwrite if we compute fresh.
        // Since we compute fresh for this batch, we should zero it out first if accumulating over batch in loop.
        // But we are processing whole batch here.
        grad_spatial_weights_.fill(0);

        // We iterate over output spatial locations (grad_spatial) and propagate back
        // Parallelize over batch dimension only to ensure thread-safety for grad_input
        DREIDEL_PARALLEL_LOOP
        for (long b = 0; b < (long)batch; ++b) {
            for (long z = 0; z < (long)in_d; ++z) {
                for (long y = 0; y < (long)in_h; ++y) {
                    for (long x = 0; x < (long)in_w; ++x) {
                        for (int c_i = 0; c_i < (int)c; ++c_i) {

                            size_t out_idx = ((b * in_d + z) * in_h + y) * in_w * c + x * c + c_i;
                            T g = grad_s_ptr[out_idx];

                            // For each weight in kernel that contributed to this output
                            for (int kz = 0; kz < k_size; ++kz) {
                                int in_z = z + kz - k_half;
                                if (in_z < 0 || in_z >= (int)in_d) continue;

                                for (int ky = 0; ky < k_size; ++ky) {
                                    int in_y = y + ky - k_half;
                                    if (in_y < 0 || in_y >= (int)in_h) continue;

                                    for (int kx = 0; kx < k_size; ++kx) {
                                        int in_x = x + kx - k_half;
                                        if (in_x < 0 || in_x >= (int)in_w) continue;

                                        size_t in_idx = ((b * in_d + in_z) * in_h + in_y) * in_w * c + in_x * c + c_i;
                                        size_t k_idx = ((kz * k_size + ky) * k_size + kx) * c + c_i;

                                        // dL/dX += g * W
                                        // Thread-safe because we parallelize over batch 'b', and each batch maps to distinct input range.
                                        grad_in_ptr[in_idx] += g * k_ptr[k_idx];

                                        // dL/dW += g * X
                                        // Weights are shared across batch, so atomic is required.
                                        #pragma omp atomic
                                        grad_w_ptr[k_idx] += g * in_ptr[in_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return grad_input;
    }

    std::vector<Tensor<T, B>*> parameters() override {
        // Combine spatial weights and LinearWHT parameters
        std::vector<Tensor<T, B>*> params = {&spatial_weights_};
        auto wht_params = linear_wht_.parameters();
        params.insert(params.end(), wht_params.begin(), wht_params.end());
        return params;
    }

    std::vector<Tensor<T, B>*> gradients() override {
        std::vector<Tensor<T, B>*> grads = {&grad_spatial_weights_};
        auto wht_grads = linear_wht_.gradients();
        grads.insert(grads.end(), wht_grads.begin(), wht_grads.end());
        return grads;
    }

    std::string name() const override { return "Conv3DSpectral"; }

private:
    size_t kernel_size_;
    size_t channels_;
    size_t d_, h_, w_;

    Tensor<T, B> spatial_weights_;
    Tensor<T, B> grad_spatial_weights_;

    LinearWHT<T, B> linear_wht_;

    Tensor<T, B> input_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_CONV3D_SPECTRAL_HPP
