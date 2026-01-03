#ifndef DREIDEL_UTILS_WAVELET_GEN_2D_HPP
#define DREIDEL_UTILS_WAVELET_GEN_2D_HPP

#include "../core/Tensor.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace dreidel {
namespace utils {

// 2D Mixed Wavelet Generator
// Adapts the 1D logic from benchmark_autoencoder.cpp to 2D images.
// Generates parameterized wavelets (Gabor, Mexican Hat, etc.) placed on a 2D grid.

template <typename T>
class WaveletGenerator2D {
public:
    WaveletGenerator2D(size_t height, size_t width) : height_(height), width_(width) {}

    void generate_batch(Tensor<T>& data, size_t batch_size) {
        // data shape: (Batch, Height, Width, 3)
        // We will generate grayscale wavelets and replicate across channels or generate color?
        // Let's generate color by varying parameters slightly per channel.

        static std::random_device rd;
        static std::mt19937 gen(rd());

        // We use local generator copies per thread if possible, or just generate params upfront.
        // For simplicity and thread safety, let's generate params serially then fill parallel.

        std::uniform_real_distribution<T> dist_x(width_ * 0.1, width_ * 0.9);
        std::uniform_real_distribution<T> dist_y(height_ * 0.1, height_ * 0.9);
        std::uniform_real_distribution<T> dist_s(2.0, 30.0); // Size range
        std::uniform_real_distribution<T> dist_w(0.1, 0.8); // Frequency
        std::uniform_real_distribution<T> dist_theta(0.0, 3.14159); // Orientation
        std::uniform_int_distribution<int> dist_type(0, 5); // Subset of nice 2D kernels
        std::uniform_int_distribution<int> dist_count(5, 15); // Number of wavelets per image

        T* ptr = data.data();
        size_t C = 3;
        size_t H = height_;
        size_t W = width_;

        // Clear
        data.fill(0);

        // Pre-generate parameters to avoid random inside parallel loop
        struct WaveletParams {
            int type;
            T cx, cy, s, w, theta;
        };

        std::vector<std::vector<WaveletParams>> batch_wavelets(batch_size);
        for(size_t i=0; i<batch_size; ++i) {
            int count = dist_count(gen);
            for(int k=0; k<count; ++k) {
                batch_wavelets[i].push_back({
                    dist_type(gen), dist_x(gen), dist_y(gen),
                    dist_s(gen), dist_w(gen), dist_theta(gen)
                });
            }
        }

        #pragma omp parallel for
        for (size_t i = 0; i < batch_size; ++i) {
            const auto& wavelets = batch_wavelets[i];

            for (const auto& wp : wavelets) {
                T cos_t = std::cos(wp.theta);
                T sin_t = std::sin(wp.theta);

                // Bounding box optimization: only iterate pixels near center?
                // For now iterate all for correctness.

                for (size_t h = 0; h < H; ++h) {
                    for (size_t w_coord = 0; w_coord < W; ++w_coord) {
                        // Coordinates relative to center
                        T dx = static_cast<T>(w_coord) - wp.cx;
                        T dy = static_cast<T>(h) - wp.cy;

                        // Rotate
                        T rx = dx * cos_t + dy * sin_t;
                        T ry = -dx * sin_t + dy * cos_t;

                        T val = 0;

                        // 2D Function definitions
                        switch(wp.type) {
                            case 0: // Gabor 2D
                                val = std::cos(wp.w * rx) * std::exp(-(rx*rx + ry*ry) / (2 * wp.s * wp.s));
                                break;
                            case 1: // Mexican Hat 2D (Laplacian of Gaussian)
                                {
                                    T r2 = (rx*rx + ry*ry) / (wp.s*wp.s);
                                    val = (1.0 - r2) * std::exp(-r2 / 2.0);
                                }
                                break;
                            case 2: // Gaussian
                                val = std::exp(-(rx*rx + ry*ry) / (2 * wp.s * wp.s));
                                break;
                            case 3: // DoG (Difference of Gaussians)
                                val = std::exp(-(rx*rx + ry*ry) / (2 * wp.s * wp.s)) - 0.5 * std::exp(-(rx*rx + ry*ry) / (2 * wp.s * wp.s * 4));
                                break;
                            case 4: // Ridge
                                 val = std::exp(-rx*rx / (2*wp.s*wp.s)); // Infinite in y direction (local ridge)
                                 // Add Gaussian envelope in y to make it a segment
                                 val *= std::exp(-ry*ry / (2*wp.s*wp.s*9)); // Elongated
                                 break;
                            case 5: // Checkerboard / High Freq
                                if (std::abs(rx) < wp.s && std::abs(ry) < wp.s) {
                                    val = std::cos(wp.w*rx) * std::cos(wp.w*ry);
                                }
                                break;
                        }

                        // Accumulate
                        size_t idx = ((i * H + h) * W + w_coord) * C;
                        // Assuming 3 channels are same (grayscale wavelets)
                        // Add to existing
                        ptr[idx + 0] += val;
                        ptr[idx + 1] += val;
                        ptr[idx + 2] += val;
                    }
                }
            }
        }
    }

private:
    size_t height_;
    size_t width_;
};

} // namespace utils
} // namespace dreidel

#endif
