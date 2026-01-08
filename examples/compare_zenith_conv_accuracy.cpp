#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/dreidel/models/ZenithHierarchicalAE.hpp"
#include "../include/dreidel/models/ConvBaselineAE.hpp"
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

int main() {
    std::cout << "=== Comparative Accuracy & Speed: Zenith vs Conv2D (128x128 Wavelets) ===\n";

    const size_t H = 128;
    const size_t W = 128;
    const size_t C = 16; // Base channels. Zenith expands to 16*16*16 = 4096. Conv matches.
    const size_t BatchSize = 2; // Keep small to avoid OOM
    const size_t Steps = 20;    // Enough to see speed and initial convergence trends

    // Generator
    std::cout << "Init Wavelet Generator (128x128)..." << std::endl;
    WaveletGenerator2D<float> gen(H, W);

    Tensor<float> batch_input({BatchSize, H, W, 3});
    Tensor<float> batch_grad({BatchSize, H, W, 3});

    // --- Zenith Setup ---
    std::cout << "\n[ZenithHierarchicalAE] Initializing (Base C=" << C << ")..." << std::endl;
    // use_slm=true for fair comparison with modern Zenith configs
    ZenithHierarchicalAE<float> zenith_ae(3, C, true, "he", true);

    SimpleAdam<float> opt_zenith(1e-3);
    opt_zenith.add_parameters(zenith_ae.parameters(), zenith_ae.gradients());

    // --- Conv Setup ---
    std::cout << "\n[ConvBaselineAE] Initializing (Base C=" << C << ")..." << std::endl;
    ConvBaselineAE<float> conv_ae(3, C);

    SimpleAdam<float> opt_conv(1e-3);
    opt_conv.add_parameters(conv_ae.parameters(), conv_ae.gradients());

    // --- Warmup ---
    std::cout << "\n[Warmup] Generating data..." << std::endl;
    gen.generate_batch(batch_input, BatchSize);

    // --- Training Loop ---
    std::cout << "\nStarting Training (" << Steps << " steps)...\n" << std::endl;

    double total_time_zenith = 0.0;
    double total_time_conv = 0.0;
    float final_loss_zenith = 0.0;
    float final_loss_conv = 0.0;

    std::cout << "Step | Z-Loss  | Z-Time(s) | C-Loss  | C-Time(s)" << std::endl;
    std::cout << "-----|---------|-----------|---------|-----------" << std::endl;

    for (size_t step = 0; step < Steps; ++step) {
        gen.generate_batch(batch_input, BatchSize);

        // Zenith Step
        auto start_z = std::chrono::high_resolution_clock::now();
        opt_zenith.zero_grad();
        Tensor<float> out_z = zenith_ae.forward(batch_input);
        float loss_z = mse_loss(out_z, batch_input, batch_grad);
        zenith_ae.backward(batch_grad);
        opt_zenith.step();
        auto end_z = std::chrono::high_resolution_clock::now();
        double time_z = std::chrono::duration<double>(end_z - start_z).count();
        total_time_zenith += time_z;
        final_loss_zenith = loss_z;

        // Conv Step
        auto start_c = std::chrono::high_resolution_clock::now();
        opt_conv.zero_grad();
        Tensor<float> out_c = conv_ae.forward(batch_input);
        float loss_c = mse_loss(out_c, batch_input, batch_grad);
        conv_ae.backward(batch_grad);
        opt_conv.step();
        auto end_c = std::chrono::high_resolution_clock::now();
        double time_c = std::chrono::duration<double>(end_c - start_c).count();
        total_time_conv += time_c;
        final_loss_conv = loss_c;

        std::cout << std::setw(4) << step+1 << " | "
                  << std::fixed << std::setprecision(5) << loss_z << " | "
                  << std::setprecision(4) << time_z << "    | "
                  << std::setprecision(5) << loss_c << " | "
                  << std::setprecision(4) << time_c << std::endl;
    }

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Zenith Total Time: " << total_time_zenith << "s (Avg: " << total_time_zenith/Steps << "s/step)" << std::endl;
    std::cout << "Conv   Total Time: " << total_time_conv << "s (Avg: " << total_time_conv/Steps << "s/step)" << std::endl;
    std::cout << "Speedup (Zenith vs Conv): " << total_time_conv / total_time_zenith << "x" << std::endl;

    std::cout << "Zenith Final Loss: " << final_loss_zenith << std::endl;
    std::cout << "Conv   Final Loss: " << final_loss_conv << std::endl;

    return 0;
}
