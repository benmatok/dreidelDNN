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
    std::cout << "=== Comparative Accuracy: Zenith (30x steps) vs Conv2D (1x step) ===\n";
    std::cout << "Goal: Compare convergence given equal wall-clock time.\n";

    const size_t H = 128;
    const size_t W = 128;
    const size_t C = 16;
    const size_t BatchSize = 2;
    const size_t ConvSteps = 20;
    const size_t ZenithMultiplier = 30;

    // Generator
    WaveletGenerator2D<float> gen(H, W);

    Tensor<float> batch_input({BatchSize, H, W, 3});
    Tensor<float> batch_grad({BatchSize, H, W, 3});

    // --- Zenith Setup ---
    std::cout << "\n[ZenithHierarchicalAE] Initializing (Base C=" << C << ")..." << std::endl;
    // Adjusted: use_pe=false, init="he"
    // Rationale: PE might add noise for wavelets.
    ZenithHierarchicalAE<float> zenith_ae(3, C, "he", true);

    // Increased LR to 1e-3 (Standard Adam) to improve convergence speed
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
    std::cout << "\nStarting Training (" << ConvSteps << " Conv steps vs " << ConvSteps * ZenithMultiplier << " Zenith steps)...\n" << std::endl;

    std::cout << "Time Unit | Z-Steps | Z-Loss  | Z-Time(s) | C-Steps | C-Loss  | C-Time(s)" << std::endl;
    std::cout << "----------|---------|---------|-----------|---------|---------|-----------" << std::endl;

    double total_time_z_accum = 0.0;
    double total_time_c_accum = 0.0;
    float current_loss_z = 0.0;

    for (size_t step = 0; step < ConvSteps; ++step) {

        // 1. Train Zenith for 30 steps
        auto start_z = std::chrono::high_resolution_clock::now();
        for(size_t z=0; z < ZenithMultiplier; ++z) {
            gen.generate_batch(batch_input, BatchSize);
            opt_zenith.zero_grad();
            Tensor<float> out_z = zenith_ae.forward(batch_input);
            current_loss_z = mse_loss(out_z, batch_input, batch_grad);
            zenith_ae.backward(batch_grad);
            opt_zenith.step();
        }
        auto end_z = std::chrono::high_resolution_clock::now();
        double time_z_block = std::chrono::duration<double>(end_z - start_z).count();
        total_time_z_accum += time_z_block;

        // 2. Train Conv for 1 step
        gen.generate_batch(batch_input, BatchSize);

        auto start_c = std::chrono::high_resolution_clock::now();
        opt_conv.zero_grad();
        Tensor<float> out_c = conv_ae.forward(batch_input);
        float loss_c = mse_loss(out_c, batch_input, batch_grad);
        conv_ae.backward(batch_grad);
        opt_conv.step();
        auto end_c = std::chrono::high_resolution_clock::now();
        double time_c_block = std::chrono::duration<double>(end_c - start_c).count();
        total_time_c_accum += time_c_block;

        std::cout << std::setw(9) << step+1 << " | "
                  << std::setw(7) << (step+1)*ZenithMultiplier << " | "
                  << std::fixed << std::setprecision(5) << current_loss_z << " | "
                  << std::setprecision(4) << time_z_block << "    | "
                  << std::setw(7) << step+1 << " | "
                  << std::setprecision(5) << loss_c << " | "
                  << std::setprecision(4) << time_c_block << std::endl;
    }

    std::cout << "\n=== Final Results ===" << std::endl;
    std::cout << "Zenith Final Loss (after " << ConvSteps * ZenithMultiplier << " steps): " << current_loss_z << std::endl;

    return 0;
}
