#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <string>

// Include stb_image_write
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/models/ZenithLassoAE.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"

using namespace dreidel;

// Helper to save Tensor as PNG
// Assumes Tensor shape [Batch, H, W, Channels] and saves the first item in batch.
// Normalizes output to 0-255.
template <typename T>
void save_tensor_as_png(const Tensor<T>& tensor, const std::string& filename, size_t batch_idx = 0) {
    auto shape = tensor.shape();
    size_t H = shape[1];
    size_t W = shape[2];
    size_t C = shape[3];

    std::vector<unsigned char> image_data(H * W * (C == 1 ? 1 : 3)); // Support grayscale or RGB (naive)

    const T* data = tensor.data() + batch_idx * (H * W * C);

    // Find min/max for normalization
    T min_val = 1e9;
    T max_val = -1e9;
    for (size_t i = 0; i < H * W * C; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    // Fallback if range is zero
    if (std::abs(max_val - min_val) < 1e-6) {
        max_val = min_val + 1.0f;
    }

    for (size_t i = 0; i < H * W; ++i) {
        if (C == 1) {
            float val = (float)(data[i] - min_val) / (max_val - min_val);
            image_data[i] = (unsigned char)(val * 255.0f);
        } else {
             // Just take first 3 channels if > 3, or replicate if < 3?
             // ZenithLassoAE output is C=1 usually.
             // If C=3, take R,G,B.
             for(size_t c=0; c<3 && c<C; ++c) {
                 float val = (float)(data[i*C + c] - min_val) / (max_val - min_val);
                 image_data[i*3 + c] = (unsigned char)(val * 255.0f);
             }
        }
    }

    int comp = (C == 1) ? 1 : 3;
    stbi_write_png(filename.c_str(), (int)W, (int)H, comp, image_data.data(), (int)W * comp);
}

// Data Gen (Copied from train_zenith_lasso.cpp but updated for consistency)
template <typename T>
void generate_wavelet_batch(Tensor<T>& data, size_t seed_offset) {
    auto shape = data.shape();
    size_t batch = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    static std::mt19937 gen(12345 + seed_offset);
    // We want some variety across calls if needed, but 'train_zenith_lasso' used 'n' index for determinism.
    // To make it vary per epoch if we wanted true training, we should mix in seed_offset.
    // However, the original code used 'n' inside the loop based on loop index.
    // Let's stick to the requested "2d wavelets" which seem to be this procedural generation.
    // To simulate a "long train" on a dataset, we might want the dataset to be static or infinite procedural.
    // Let's assume infinite procedural but consistent within a batch.

    T* ptr = data.data();

    #pragma omp parallel for
    for(size_t n=0; n<batch; ++n) {
        // Mix seed_offset into generation to create different images over time (infinite dataset)
        // Or keep it static to overfit/memorize.
        // "Long train" usually implies learning a distribution.
        // Let's add seed_offset to 'n' for variety.
        size_t eff_n = n + seed_offset;

        T mu_x = 0.5 * W + 0.2 * W * std::sin(eff_n * 0.1);
        T mu_y = 0.5 * H + 0.2 * H * std::cos(eff_n * 0.1);
        T s_x = (W/10.0) * (1.0 + 0.5 * std::sin(eff_n * 0.05));

        for(size_t c=0; c<C; ++c) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w_idx=0; w_idx<W; ++w_idx) {
                     T x = (T)w_idx - mu_x;
                     T y = (T)h - mu_y;
                     T val = std::exp(-(x*x + y*y)/(2*s_x*s_x));
                     ptr[((n*H + h)*W + w_idx)*C + c] = val;
                }
            }
        }
    }
}

int main() {
    std::cout << "=== Zenith-Lasso Long Training (128x128) ===" << std::endl;

    // Config
    size_t batch_size = 8;
    size_t H = 128, W = 128;
    size_t epochs = 1000; // Long train
    float lr = 0.001f;
    float max_lambda = 1e-4f;

    // Model
    models::ZenithLassoAE<float> model;

    // Optimizer
    optim::SimpleAdam<float> optimizer(lr);
    optimizer.set_coordinate_wise_clipping(true, 5.0);

    auto params = model.parameters();
    auto grads = model.gradients();
    optimizer.add_parameters(params, grads);

    Tensor<float> input({batch_size, H, W, 1});
    Tensor<float> target({batch_size, H, W, 1});

    auto start_time = std::chrono::high_resolution_clock::now();

    for(size_t epoch=0; epoch<epochs; ++epoch) {
        // Lambda Schedule
        float current_lambda = 0.0f;
        if (epoch > 100) {
            float progress = (float)(epoch - 100) / 400.0f;
            if (progress > 1.0f) progress = 1.0f;
            current_lambda = max_lambda * progress;
        }

        // Generate data (Varying seed to simulate dataset)
        generate_wavelet_batch(input, epoch * batch_size);
        target = input; // Autoencoder target is input

        // Forward
        Tensor<float> output = model.forward(input);

        // Loss (MSE)
        float loss = 0;
        size_t total_elements = output.size();
        const float* out_ptr = output.data();
        const float* tgt_ptr = target.data();

        Tensor<float> grad_output(output.shape());
        float* go_ptr = grad_output.data();

        #pragma omp parallel for reduction(+:loss)
        for(size_t i=0; i<total_elements; ++i) {
            float diff = out_ptr[i] - tgt_ptr[i];
            loss += diff * diff;
            go_ptr[i] = 2.0f * diff / total_elements;
        }
        loss /= total_elements;

        // Backward
        optimizer.zero_grad();
        model.backward(grad_output);

        // Update
        optimizer.step();

        // Regularizer
        model.apply_lasso(current_lambda, lr);

        // Monitoring
        if (epoch % 50 == 0 || epoch == epochs - 1) {
            size_t total_blocks, zero_blocks;
            model.get_sparsity_stats(total_blocks, zero_blocks);
            float sparsity = (float)zero_blocks / (total_blocks + 1e-9f);

            std::cout << "Epoch " << std::setw(4) << epoch
                      << " | Loss: " << loss
                      << " | Lambda: " << current_lambda
                      << " | Sparsity: " << (sparsity * 100.0f) << "%" << std::endl;

            // Save Ablation Images
            std::string suffix = std::to_string(epoch);
            save_tensor_as_png(target, "ablation_target_" + suffix + ".png", 0);
            save_tensor_as_png(output, "ablation_recon_" + suffix + ".png", 0);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_time = end_time - start_time;
    std::cout << "Training Time: " << train_time.count() << "s" << std::endl;

    return 0;
}
