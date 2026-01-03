#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/models/ComparativeAE.hpp" // Contains ZenithHierarchicalAE
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include "../include/dreidel/utils/WaveletGen2D.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <algorithm>

using namespace dreidel;
using namespace dreidel::models;
using namespace dreidel::utils;
using namespace dreidel::optim;

// --- Noise & Augmentation Helpers ---

template <typename T>
void add_gaussian_noise(Tensor<T>& img, float sigma) {
    static std::mt19937 gen(1234);
    std::normal_distribution<T> d(0, sigma);
    T* data = img.data();
    for (size_t i = 0; i < img.size(); ++i) {
        data[i] += d(gen);
    }
}

template <typename T>
void add_correlated_noise(Tensor<T>& img, float sigma) {
    // Generate noise
    Tensor<T> noise = img; // Copy shape
    static std::mt19937 gen(5678);
    std::normal_distribution<T> d(0, sigma);
    T* n_ptr = noise.data();
    for (size_t i = 0; i < noise.size(); ++i) n_ptr[i] = d(gen);

    // Blur noise (Simple 3x3 Box Blur) to correlate it
    Tensor<T> blurred = noise;
    blurred.fill(0);

    // Assuming NHWC
    size_t N = img.shape()[0];
    size_t H = img.shape()[1];
    size_t W = img.shape()[2];
    size_t C = img.shape()[3];

    for (size_t n = 0; n < N; ++n) {
        for (size_t h = 1; h < H - 1; ++h) {
            for (size_t w = 1; w < W - 1; ++w) {
                for (size_t c = 0; c < C; ++c) {
                    T sum = 0;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            size_t idx = ((n * H + (h+dy)) * W + (w+dx)) * C + c;
                            sum += n_ptr[idx];
                        }
                    }
                    size_t out_idx = ((n * H + h) * W + w) * C + c;
                    blurred.data()[out_idx] = sum / 9.0;
                }
            }
        }
    }

    // Add to image
    img = img + blurred;
}

template <typename T>
void add_missing_pixels(Tensor<T>& img, float prob) {
    static std::mt19937 gen(9999);
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    T* data = img.data();
    for (size_t i = 0; i < img.size(); ++i) {
        if (d(gen) < prob) {
            data[i] = 0;
        }
    }
}

// Simulate "Text" Overlay by drawing random lines/rects
template <typename T>
void add_text_overlay(Tensor<T>& img) {
    size_t N = img.shape()[0];
    size_t H = img.shape()[1];
    size_t W = img.shape()[2];
    size_t C = img.shape()[3];

    static std::mt19937 gen(4242);
    std::uniform_int_distribution<int> coord_h(0, H-1);
    std::uniform_int_distribution<int> coord_w(0, W-1);
    std::uniform_int_distribution<int> length_dist(10, 100);

    for (size_t n = 0; n < N; ++n) {
        // Draw 5 random "lines" per image
        for (int k = 0; k < 10; ++k) {
            int h0 = coord_h(gen);
            int w0 = coord_w(gen);
            int len = length_dist(gen);
            int dir = gen() % 2; // 0: horiz, 1: vert

            for (int l = 0; l < len; ++l) {
                int h = h0 + (dir == 1 ? l : 0);
                int w = w0 + (dir == 0 ? l : 0);
                if (h < H && w < W) {
                    for (size_t c = 0; c < C; ++c) {
                        size_t idx = ((n * H + h) * W + w) * C + c;
                        img.data()[idx] = 1.0; // White text
                    }
                }
            }
        }
    }
}

// --- Utils ---

void save_image(const std::string& filename, const Tensor<float>& tensor, int batch_idx) {
    // Tensor is NHWC. Extract batch_idx.
    size_t H = tensor.shape()[1];
    size_t W = tensor.shape()[2];
    size_t C = tensor.shape()[3];

    std::vector<unsigned char> img_data(H * W * C);
    const float* data = tensor.data() + batch_idx * H * W * C;

    for (size_t i = 0; i < H * W * C; ++i) {
        float val = data[i];
        val = std::max(0.0f, std::min(1.0f, val));
        img_data[i] = static_cast<unsigned char>(val * 255.0f);
    }

    stbi_write_png(filename.c_str(), W, H, C, img_data.data(), W * C);
    std::cout << "Saved " << filename << std::endl;
}

int main() {
    std::cout << "Running Zenith Reconstruction Test on 512x512 Images..." << std::endl;

    const size_t H = 512;
    const size_t W = 512;
    const size_t C = 3;
    const size_t BATCH_SIZE = 1; // Keep it small for speed in this test
    const size_t EPOCHS = 100;

    // 1. Initialize Model
    ZenithHierarchicalAE<float> model(2); // Base filters 2 (Stage1=32, Stage2=512)

    // Optimizer
    SimpleAdam<float> optimizer(0.001f);
    optimizer.add_parameters(model.parameters(), model.gradients());

    // 2. Data Generator
    WaveletGenerator2D<float> generator(H, W);
    Tensor<float> inputs({BATCH_SIZE, H, W, C});
    Tensor<float> targets({BATCH_SIZE, H, W, C});

    // 3. Training Loop
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        // Generate Clean Targets
        generator.generate_batch(targets, BATCH_SIZE);

        // Create Noisy Inputs
        inputs = targets; // Copy

        // Apply Augmentations (Rotate through types or apply mix)
        // For this test, let's apply a mix to the single batch item
        if (epoch % 4 == 0) {
            std::cout << "Augmentation: Text Overlay" << std::endl;
            add_text_overlay(inputs);
        } else if (epoch % 4 == 1) {
            std::cout << "Augmentation: Correlated Noise" << std::endl;
            add_correlated_noise(inputs, 0.2f);
        } else if (epoch % 4 == 2) {
            std::cout << "Augmentation: Missing Pixels" << std::endl;
            add_missing_pixels(inputs, 0.3f); // 30% dropout
        } else {
             std::cout << "Augmentation: Mixed (Noise + Text)" << std::endl;
             add_gaussian_noise(inputs, 0.1f);
             add_text_overlay(inputs);
        }

        // Zero gradients
        optimizer.zero_grad();

        // Forward
        Tensor<float> outputs = model.forward(inputs);

        // Loss (MSE)
        Tensor<float> diff = outputs - targets;
        // Manual MSE calculation
        float loss = 0;
        for (size_t i = 0; i < diff.size(); ++i) loss += diff.data()[i] * diff.data()[i];
        loss /= diff.size();

        std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;

        // Backward
        Tensor<float> grad_loss = diff * (2.0f / diff.size()); // dL/dx = 2/N * (x - y)
        model.backward(grad_loss);

        // Step
        optimizer.step();

        // Save images for the last epoch
        if (epoch == EPOCHS - 1) {
            save_image("test_512_input.png", inputs, 0);
            save_image("test_512_target.png", targets, 0);
            save_image("test_512_output.png", outputs, 0);
        }
    }

    std::cout << "Test Complete." << std::endl;
    return 0;
}
