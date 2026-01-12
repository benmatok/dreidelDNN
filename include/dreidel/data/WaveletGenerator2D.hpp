#pragma once

#include "../core/Tensor.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include <iostream>

namespace dreidel {
namespace data {

template <typename T>
class WaveletGenerator2D {
public:
    WaveletGenerator2D(size_t height, size_t width, size_t channels)
        : height_(height), width_(width), channels_(channels) {}

    Tensor<T> generate_batch(size_t batch_size) {
        Tensor<T> batch({batch_size, height_, width_, channels_});
        T* ptr = batch.data();

        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd() + omp_get_thread_num());

            // Distributions tailored for HxW
            T scale_base = static_cast<T>(std::min(height_, width_)) / 64.0f;

            std::uniform_int_distribution<int> dist_type(0, 29);
            std::uniform_real_distribution<T> dist_pos_x(0.2f * width_, 0.8f * width_);
            std::uniform_real_distribution<T> dist_pos_y(0.2f * height_, 0.8f * height_);
            std::uniform_real_distribution<T> dist_scale(3.0f * scale_base, 12.0f * scale_base);
            std::uniform_real_distribution<T> dist_angle(0.0, 3.14159);
            std::uniform_real_distribution<T> dist_freq(0.1, 0.5);
            std::uniform_real_distribution<T> dist_phase(0.0, 6.28);
            std::uniform_real_distribution<T> dist_color(0.2, 1.0);

            #pragma omp for
            for (size_t b = 0; b < batch_size; ++b) {
                // For each sample, we might have different patterns per channel or same pattern?
                // Real images have correlated channels.
                // Let's generate a pattern and apply random color mix.

                size_t offset_base = b * height_ * width_ * channels_;

                // Clear
                std::fill(ptr + offset_base, ptr + offset_base + height_ * width_ * channels_, 0.0f);

                int type = dist_type(gen);

                // Generate a base intensity map [H, W]
                std::vector<T> intensity(height_ * width_);

                if (type < 5) { // Gabor
                    T cx = dist_pos_x(gen);
                    T cy = dist_pos_y(gen);
                    T sx = dist_scale(gen);
                    T sy = sx * std::uniform_real_distribution<T>(0.5, 1.5)(gen);
                    T theta = dist_angle(gen);
                    generate_gabor(intensity.data(), width_, height_, cx, cy, sx, sy, theta, dist_freq(gen), dist_phase(gen));
                } else if (type < 10) { // Curvelet
                    T cx = dist_pos_x(gen);
                    T cy = dist_pos_y(gen);
                    T sx = dist_scale(gen) * 1.5;
                    T sy = sx * 0.1;
                    T theta = dist_angle(gen);
                    generate_gabor(intensity.data(), width_, height_, cx, cy, sx, sy, theta, dist_freq(gen), dist_phase(gen));
                } else if (type < 15) { // Mexican Hat
                    T cx = dist_pos_x(gen);
                    T cy = dist_pos_y(gen);
                    T s = dist_scale(gen);
                    generate_mexican_hat(intensity.data(), width_, height_, cx, cy, s);
                } else if (type < 25) { // Texture
                    for(int k=0; k<3; ++k) {
                         T kx = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                         T ky = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                         T ph = dist_phase(gen);
                         for(size_t y=0; y<height_; ++y) {
                            for(size_t x=0; x<width_; ++x) {
                                intensity[y*width_ + x] += std::cos(kx*x + ky*y + ph);
                            }
                         }
                    }
                    for(size_t i=0; i<intensity.size(); ++i) intensity[i] /= 2.0f;
                } else { // Geometric
                    T cx = dist_pos_x(gen);
                    T cy = dist_pos_y(gen);
                    T w = dist_scale(gen);
                    T h = dist_scale(gen);
                    T angle = dist_angle(gen);
                    T ca = std::cos(angle);
                    T sa = std::sin(angle);
                    for(size_t y=0; y<height_; ++y) {
                        for(size_t x=0; x<width_; ++x) {
                            T dx = (T)x - cx;
                            T dy = (T)y - cy;
                            T rx = dx * ca - dy * sa;
                            T ry = dx * sa + dy * ca;
                            if (std::abs(rx) < w && std::abs(ry) < h) {
                                intensity[y*width_ + x] = 1.0f;
                            } else {
                                intensity[y*width_ + x] = -0.5f;
                            }
                        }
                    }
                }

                // Apply to channels with random mixing
                std::vector<T> c_mix(channels_);
                for(size_t c=0; c<channels_; ++c) c_mix[c] = dist_color(gen);

                for(size_t y=0; y<height_; ++y) {
                    for(size_t x=0; x<width_; ++x) {
                        T val = intensity[y*width_ + x];
                        for(size_t c=0; c<channels_; ++c) {
                            ptr[offset_base + (y*width_ + x)*channels_ + c] = val * c_mix[c];
                        }
                    }
                }

                // Normalize per sample
                T max_val = 0;
                for(size_t i=0; i<height_*width_*channels_; ++i) {
                     max_val = std::max(max_val, std::abs(ptr[offset_base + i]));
                }
                if (max_val > 1e-6f) {
                    for(size_t i=0; i<height_*width_*channels_; ++i) {
                         ptr[offset_base + i] /= max_val;
                    }
                }
            }
        }

        return batch;
    }

private:
    size_t height_;
    size_t width_;
    size_t channels_;

    static void generate_gabor(T* buffer, size_t width, size_t height, T cx, T cy, T sx, T sy, T theta, T freq, T psi) {
        T cos_t = std::cos(theta);
        T sin_t = std::sin(theta);
        for(size_t y=0; y<height; ++y) {
            for(size_t x=0; x<width; ++x) {
                T dx = (T)x - cx;
                T dy = (T)y - cy;
                T xp = dx * cos_t + dy * sin_t;
                T yp = -dx * sin_t + dy * cos_t;
                T env = std::exp(-(xp*xp)/(2*sx*sx) - (yp*yp)/(2*sy*sy));
                T carrier = std::cos(2*3.14159 * freq * xp + psi);
                buffer[y*width + x] += env * carrier;
            }
        }
    }

    static void generate_mexican_hat(T* buffer, size_t width, size_t height, T cx, T cy, T sigma) {
        for(size_t y=0; y<height; ++y) {
            for(size_t x=0; x<width; ++x) {
                T dx = (T)x - cx;
                T dy = (T)y - cy;
                T r2 = dx*dx + dy*dy;
                T s2 = sigma*sigma;
                buffer[y*width + x] += (1.0 - r2/s2) * std::exp(-r2/(2*s2));
            }
        }
    }
};

} // namespace data
} // namespace dreidel
