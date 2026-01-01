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

        std::uniform_real_distribution<T> dist_x(width_ * 0.2, width_ * 0.8);
        std::uniform_real_distribution<T> dist_y(height_ * 0.2, height_ * 0.8);
        std::uniform_real_distribution<T> dist_s(2.0, 10.0); // Size
        std::uniform_real_distribution<T> dist_w(0.1, 0.8); // Frequency
        std::uniform_real_distribution<T> dist_theta(0.0, 3.14159); // Orientation
        std::uniform_int_distribution<int> dist_type(0, 5); // Subset of nice 2D kernels

        T* ptr = data.data();
        size_t C = 3;
        size_t H = height_;
        size_t W = width_;

        // Clear
        data.fill(0);

        #pragma omp parallel for
        for (size_t i = 0; i < batch_size; ++i) {
            int type = dist_type(gen);
            T cx = dist_x(gen);
            T cy = dist_y(gen);
            T s = dist_s(gen);
            T w = dist_w(gen);
            T theta = dist_theta(gen);

            T cos_t = std::cos(theta);
            T sin_t = std::sin(theta);

            for (size_t h = 0; h < H; ++h) {
                for (size_t w_coord = 0; w_coord < W; ++w_coord) {
                    // Coordinates relative to center
                    T dx = static_cast<T>(w_coord) - cx;
                    T dy = static_cast<T>(h) - cy;

                    // Rotate
                    T rx = dx * cos_t + dy * sin_t;
                    T ry = -dx * sin_t + dy * cos_t;

                    T val = 0;

                    // 2D Function definitions
                    switch(type) {
                        case 0: // Gabor 2D
                            val = std::cos(w * rx) * std::exp(-(rx*rx + ry*ry) / (2 * s * s));
                            break;
                        case 1: // Mexican Hat 2D (Laplacian of Gaussian)
                            {
                                T r2 = (rx*rx + ry*ry) / (s*s);
                                val = (1.0 - r2) * std::exp(-r2 / 2.0);
                            }
                            break;
                        case 2: // Gaussian
                            val = std::exp(-(rx*rx + ry*ry) / (2 * s * s));
                            break;
                        case 3: // DoG (Difference of Gaussians)
                            val = std::exp(-(rx*rx + ry*ry) / (2 * s * s)) - 0.5 * std::exp(-(rx*rx + ry*ry) / (2 * s * s * 4));
                            break;
                        case 4: // Ridge
                             val = std::exp(-rx*rx / (2*s*s)); // Infinite in y direction (local ridge)
                             // Mask with bounds? No, let it be a line.
                             // Add Gaussian envelope in y to make it a segment?
                             val *= std::exp(-ry*ry / (2*s*s*9)); // Elongated
                             break;
                        case 5: // Checkerboard / High Freq
                            if (std::abs(rx) < s && std::abs(ry) < s) {
                                val = std::cos(w*rx) * std::cos(w*ry);
                            }
                            break;
                    }

                    // Fill channels (Grayscale for now, or slight tint?)
                    // Let's do grayscale + noise
                    size_t idx = ((i * H + h) * W + w_coord) * C;
                    ptr[idx + 0] = val;
                    ptr[idx + 1] = val;
                    ptr[idx + 2] = val;
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
