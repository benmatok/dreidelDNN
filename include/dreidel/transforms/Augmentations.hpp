#pragma once

#include "../core/Tensor.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>

namespace dreidel {
namespace transforms {

class Augmentations {
public:
    // Random Crop: Extracts a 512x512 crop from a larger image (HWC)
    static void RandomCrop(const Tensor<float>& input, Tensor<float>& output, int out_h, int out_w) {
        // Input: [1, H, W, C] (assuming single image for augmentation usually, or handled per image)
        // Output: [1, out_h, out_w, C]

        auto shape = input.shape();
        int in_h = shape[1];
        int in_w = shape[2];
        int c = shape[3];

        static std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dis_h(0, in_h - out_h);
        std::uniform_int_distribution<> dis_w(0, in_w - out_w);

        int start_h = dis_h(gen);
        int start_w = dis_w(gen);

        const float* in_ptr = input.data();
        float* out_ptr = output.data();

        // Parallel copy
        #pragma omp parallel for
        for (int y = 0; y < out_h; ++y) {
            const float* src_row = in_ptr + ((start_h + y) * in_w + start_w) * c;
            float* dst_row = out_ptr + (y * out_w) * c;
            std::memcpy(dst_row, src_row, out_w * c * sizeof(float));
        }
    }

    // PatchMask (MAE Style): Masks 75% of 16x16 patches (or custom block size)
    // Here we use 8x8 to match SpaceToDepth block size if needed, or bigger.
    // MAE usually uses 16x16.
    static void PatchMask(Tensor<float>& input, float mask_ratio = 0.75f, int patch_size = 16) {
        auto shape = input.shape();
        int h = shape[1];
        int w = shape[2];
        int c = shape[3];

        int grid_h = h / patch_size;
        int grid_w = w / patch_size;
        int num_patches = grid_h * grid_w;
        int num_masked = (int)(num_patches * mask_ratio);

        std::vector<int> indices(num_patches);
        for(int i=0; i<num_patches; ++i) indices[i] = i;

        static std::mt19937 gen(std::random_device{}());
        std::shuffle(indices.begin(), indices.end(), gen);

        float* data = input.data();

        // Mask the first num_masked indices
        // Fill with grey (0.5) or noise? Plan says "Grey/Noise". Let's use 0.5 (Grey).
        #pragma omp parallel for
        for(int i=0; i<num_masked; ++i) {
            int idx = indices[i];
            int py = idx / grid_w;
            int px = idx % grid_w;

            int start_y = py * patch_size;
            int start_x = px * patch_size;

            for(int y=0; y<patch_size; ++y) {
                for(int x=0; x<patch_size; ++x) {
                    int pixel_idx = ((start_y + y) * w + (start_x + x)) * c;
                    for(int k=0; k<c; ++k) {
                        data[pixel_idx + k] = 0.5f;
                    }
                }
            }
        }
    }

    // Gaussian Noise
    static void GaussianNoise(Tensor<float>& input, float std_dev = 0.1f) {
        size_t size = input.size();
        float* data = input.data();

        // Generating noise in parallel is tricky with standard RNG.
        // We can use a simple custom RNG or per-thread engine.

        #pragma omp parallel
        {
            // Per thread generator
            std::mt19937 gen(std::random_device{}() + omp_get_thread_num());
            std::normal_distribution<float> dist(0.0f, std_dev);

            #pragma omp for
            for(size_t i=0; i<size; ++i) {
                data[i] += dist(gen);
            }
        }
    }

    // Affine Transform: Rotate 180 (Flip) + Shear (Approx)
    // Plan says "Rotate 180, Shear".
    // 180 rotation is just reading backwards or flipping axes.
    // Shear requires interpolation.
    // Let's implement a generic warp. Or simpler:
    // "View C: Chaos... Rotate 180 + Gaussian Noise".
    // Actually, "Affine(Rotate 180, Shear)".
    // Let's just implement Rotate 180 (Flip H+V) and maybe a shift/shear if easy.
    // Bilinear interpolation in C++ for generic affine is verbose.
    // Let's stick to Rotate 180 (Flip H + Flip V) which is very chaotic for orientation.
    static void Rotate180(const Tensor<float>& input, Tensor<float>& output) {
        auto shape = input.shape();
        int h = shape[1];
        int w = shape[2];
        int c = shape[3];

        const float* in_data = input.data();
        float* out_data = output.data();

        #pragma omp parallel for
        for(int y=0; y<h; ++y) {
            for(int x=0; x<w; ++x) {
                int src_idx = (y * w + x) * c;
                int dst_idx = ((h - 1 - y) * w + (w - 1 - x)) * c;
                for(int k=0; k<c; ++k) {
                    out_data[dst_idx + k] = in_data[src_idx + k];
                }
            }
        }
    }

    // Applying "Chaos" (View C)
    static void ApplyChaos(const Tensor<float>& input, Tensor<float>& output) {
        // Rotate 180
        Rotate180(input, output);
        // Add Noise
        GaussianNoise(output, 0.1f); // Moderate noise
    }

    // Copy helper
    static void Copy(const Tensor<float>& src, Tensor<float>& dst) {
        std::memcpy(dst.data(), src.data(), src.size() * sizeof(float));
    }
};

} // namespace transforms
} // namespace dreidel
