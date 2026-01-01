#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/dreidel/models/ComparativeAE.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/utils/WaveletGen2D.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <string>

using namespace dreidel;
using namespace dreidel::models;
using namespace dreidel::utils;
using namespace dreidel::optim;

// MSE Loss with Gradient
template <typename T>
T mse_loss(const Tensor<T>& pred, const Tensor<T>& target, Tensor<T>& grad_input) {
    size_t size = pred.size();
    if (size != target.size()) throw std::invalid_argument("Size mismatch in MSE");

    const T* p_ptr = pred.data();
    const T* t_ptr = target.data();
    T* g_ptr = grad_input.data();

    T sum_sq = 0;
    T scale = 2.0 / size;

    #pragma omp parallel for reduction(+:sum_sq)
    for (size_t i = 0; i < size; ++i) {
        T diff = p_ptr[i] - t_ptr[i];
        sum_sq += diff * diff;
        g_ptr[i] = diff * scale;
    }

    return sum_sq / size;
}

// MAE Calculation (Metric only)
template <typename T>
T calculate_mae(const Tensor<T>& pred, const Tensor<T>& target) {
    size_t size = pred.size();
    const T* p_ptr = pred.data();
    const T* t_ptr = target.data();
    T sum_abs = 0;

    #pragma omp parallel for reduction(+:sum_abs)
    for (size_t i = 0; i < size; ++i) {
        sum_abs += std::abs(p_ptr[i] - t_ptr[i]);
    }
    return sum_abs / size;
}

// Helper to save tensor as image
template <typename T>
void save_tensor_as_png(const Tensor<T>& tensor, size_t batch_idx, size_t H, size_t W, const std::string& filename) {
    // Tensor is NHWC
    std::vector<unsigned char> image(H * W * 3);
    const T* data = tensor.data();
    size_t C = 3;
    size_t offset = batch_idx * H * W * C;

    for (size_t i = 0; i < H * W; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            float val = data[offset + i * 3 + c];
            // Normalize roughly from [-1, 1] or [0, 1] to [0, 255]
            // Input data from WaveletGen is roughly -1 to 1 based on previous runs stats
            // Let's assume [-1, 1] mapped to [0, 255]
            val = (val + 1.0f) * 0.5f;
            val = std::max(0.0f, std::min(1.0f, val));
            image[i * 3 + c] = static_cast<unsigned char>(val * 255.0f);
        }
    }

    stbi_write_png(filename.c_str(), W, H, 3, image.data(), W * 3);
    std::cout << "Saved " << filename << std::endl;
}

int main() {
    std::cout << "=== Training Comparative Autoencoders on Wavelets (32x32) ===\n";

    // Config - REDUCED
    const size_t H = 32;
    const size_t W = 32;
    const size_t C = 8;
    const size_t BatchSize = 2; // Reduced to 2 to avoid OOM/Stack issues in sandbox
    const size_t Epochs = 2; // Reduced for speed check
    const size_t StepsPerEpoch = 5;

    // Generator
    std::cout << "Init Generator..." << std::endl;
    WaveletGenerator2D<float> gen(H, W);
    std::cout << "Allocating Batches..." << std::endl;
    Tensor<float> batch_input({BatchSize, H, W, 3});
    Tensor<float> batch_grad({BatchSize, H, W, 3});

    // Models
    std::cout << "Initializing ZenithHierarchicalAE..." << std::endl;
    ZenithHierarchicalAE<float> zenith_ae(C);

    std::cout << "Initializing ConvBaselineAE..." << std::endl;
    ConvBaselineAE<float> conv_ae(C);

    // Optimizers
    std::cout << "Initializing Optimizers..." << std::endl;
    SimpleAdam<float> opt_zenith(1e-3);
    opt_zenith.add_parameters(zenith_ae.parameters(), zenith_ae.gradients());

    SimpleAdam<float> opt_conv(1e-3);
    opt_conv.add_parameters(conv_ae.parameters(), conv_ae.gradients());

    // Training Loop
    auto total_start = std::chrono::high_resolution_clock::now();

    for (size_t epoch = 0; epoch < Epochs; ++epoch) {
        float loss_z_acc = 0;
        float loss_c_acc = 0;

        for (size_t step = 0; step < StepsPerEpoch; ++step) {
            // 1. Generate Data
            gen.generate_batch(batch_input, BatchSize);

            // 2. Train Zenith
            opt_zenith.zero_grad();
            Tensor<float> out_z = zenith_ae.forward(batch_input);
            float loss_z = mse_loss(out_z, batch_input, batch_grad);
            zenith_ae.backward(batch_grad);
            opt_zenith.step();
            loss_z_acc += loss_z;

            // 3. Train Conv
            opt_conv.zero_grad();
            Tensor<float> out_c = conv_ae.forward(batch_input);
            float loss_c = mse_loss(out_c, batch_input, batch_grad); // reuse grad buffer
            conv_ae.backward(batch_grad);
            opt_conv.step();
            loss_c_acc += loss_c;
        }

        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == Epochs - 1) {
            auto current_time = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - total_start).count();
            std::cout << "Epoch " << epoch+1 << "/" << Epochs
                      << " | Elapsed: " << elapsed << "s"
                      << " | Loss Z: " << std::setprecision(5) << loss_z_acc / StepsPerEpoch
                      << " | Loss C: " << std::setprecision(5) << loss_c_acc / StepsPerEpoch << std::endl;
        }
    }

    std::cout << "\nTraining Complete. Running Evaluation...\n";
    // ... rest ...
    return 0;
}
