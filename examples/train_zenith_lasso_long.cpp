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

    std::vector<unsigned char> image_data(H * W * (C == 1 ? 1 : 3));

    const T* data = tensor.data() + batch_idx * (H * W * C);

    // Find min/max for normalization
    T min_val = 1e9;
    T max_val = -1e9;
    for (size_t i = 0; i < H * W * C; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    std::cout << "Saving " << filename << " | Range: [" << min_val << ", " << max_val << "]" << std::endl;

    // Fallback if range is zero or very small
    T range = max_val - min_val;
    if (range < 1e-12f) { // Lowered threshold significantly
        max_val = min_val + 1.0f;
        range = 1.0f;
    }

    for (size_t i = 0; i < H * W; ++i) {
        if (C == 1) {
            float val = (float)(data[i] - min_val) / range;
            // Clamp
            if (val < 0) val = 0;
            if (val > 1) val = 1;
            image_data[i] = (unsigned char)(val * 255.0f);
        } else {
             for(size_t c=0; c<3 && c<C; ++c) {
                 float val = (float)(data[i*C + c] - min_val) / range;
                 if (val < 0) val = 0;
                 if (val > 1) val = 1;
                 image_data[i*3 + c] = (unsigned char)(val * 255.0f);
             }
        }
    }

    int comp = (C == 1) ? 1 : 3;
    stbi_write_png(filename.c_str(), (int)W, (int)H, comp, image_data.data(), (int)W * comp);
}

// Data Gen
template <typename T>
void generate_wavelet_batch(Tensor<T>& data, size_t seed_offset) {
    auto shape = data.shape();
    size_t batch = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    // Use consistent seed + offset
    std::mt19937 gen(12345 + seed_offset);

    T* ptr = data.data();

    #pragma omp parallel for
    for(size_t n=0; n<batch; ++n) {
        size_t eff_n = n + seed_offset;

        // Ensure visible features
        T mu_x = 0.5 * W + 0.25 * W * std::sin(eff_n * 0.3);
        T mu_y = 0.5 * H + 0.25 * H * std::cos(eff_n * 0.2);
        T s_x = (W/8.0) * (1.0 + 0.3 * std::sin(eff_n * 0.5));

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
    size_t epochs = 5000;
    float lr = 0.001f; // Lowered LR for stability and smoothness
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

    // Fixed visualization batch
    Tensor<float> vis_input({batch_size, H, W, 1});
    generate_wavelet_batch(vis_input, 99999); // Fixed seed for visualization

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t dataset_size = 100; // Simulate 100 fixed batches

    for(size_t epoch=0; epoch<epochs; ++epoch) {
        // Lambda Schedule
        float current_lambda = 0.0f;
        if (epoch > 100) {
            float progress = (float)(epoch - 100) / 400.0f;
            if (progress > 1.0f) progress = 1.0f;
            current_lambda = max_lambda * progress;
        }

        // Cycle through a fixed "dataset" of seeds
        generate_wavelet_batch(input, (epoch % dataset_size) * batch_size);
        target = input;

        // Forward
        Tensor<float> output = model.forward(input);

        // Loss (MSE)
        float loss = 0;
        float mae = 0; // MAE accumulator
        size_t total_elements = output.size();
        const float* out_ptr = output.data();
        const float* tgt_ptr = target.data();

        Tensor<float> grad_output(output.shape());
        float* go_ptr = grad_output.data();

        #pragma omp parallel for reduction(+:loss, mae)
        for(size_t i=0; i<total_elements; ++i) {
            float diff = out_ptr[i] - tgt_ptr[i];
            loss += diff * diff;
            mae += std::abs(diff); // Compute MAE
            go_ptr[i] = 2.0f * diff / total_elements;
        }
        loss /= total_elements;
        mae /= total_elements;

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
                      << " | Loss (MSE): " << loss
                      << " | MAE: " << mae
                      << " | Lambda: " << current_lambda
                      << " | Sparsity: " << (sparsity * 100.0f) << "%" << std::endl;

            // Save Ablation Images (Use Fixed Visualization Batch)
            Tensor<float> vis_output = model.forward(vis_input);
            std::string suffix = std::to_string(epoch);
            save_tensor_as_png(vis_input, "ablation_target_" + suffix + ".png", 0);
            save_tensor_as_png(vis_output, "ablation_recon_" + suffix + ".png", 0);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_time = end_time - start_time;
    std::cout << "Training Time: " << train_time.count() << "s" << std::endl;

    return 0;
}
