#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric>
#include <iomanip>
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"

using namespace dreidel;

// Synthetic Data: Wavelets or just random data?
// Ideally we want data with some structure that SLM can exploit.
// A sum of sines or sawtooths might be good.
Tensor<float> generate_data(size_t N, size_t H, size_t W, size_t C) {
    Tensor<float> data({N, H, W, C});
    float* ptr = data.data();
    for (size_t n = 0; n < N; ++n) {
        for (size_t i = 0; i < H * W; ++i) {
            for (size_t c = 0; c < C; ++c) {
                // Generate something with frequency components
                float val = 0;
                // Add a few low sequency components
                val += std::sin(c * 0.1f);
                val += (c % 4 == 0) ? 1.0f : 0.0f;
                ptr[(n * H * W + i) * C + c] = val;
            }
        }
    }
    return data;
}

struct BenchmarkResult {
    double forward_ms;
    double backward_ms;
    float initial_loss;
    float final_loss;
};

BenchmarkResult run_benchmark(size_t C, bool use_sequency) {
    size_t N = 4;
    size_t H = 64;
    size_t W = 64;
    size_t steps = 50; // Short training loop

    // Create Layer
    // ZenithBlock(in, out, k, spec, ifwht, dilated, gating, stride, upscale, init, slm, sequency)
    layers::ZenithBlock<float> layer(C, C, 3, C, true, false, false, 1, 1, "he", true, use_sequency);

    // Data
    auto input = generate_data(N, H, W, C);
    auto target = input; // Identity task (Autoencoder-like)

    // Optimizer
    optim::SimpleAdam<float> optimizer(0.01f);
    optimizer.add_parameters(layer.parameters(), layer.gradients());

    auto start_time = std::chrono::high_resolution_clock::now();

    double total_fwd = 0;
    double total_bwd = 0;
    float init_loss = 0;
    float final_loss = 0;

    for (size_t s = 0; s < steps; ++s) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto output = layer.forward(input);
        auto t1 = std::chrono::high_resolution_clock::now();

        // Loss (MSE)
        float loss = 0;
        Tensor<float> grad_out = output; // Placeholder for gradient
        float* g_ptr = grad_out.data();
        const float* o_ptr = output.data();
        const float* t_ptr = target.data();

        for (size_t i = 0; i < output.size(); ++i) {
            float diff = o_ptr[i] - t_ptr[i];
            loss += diff * diff;
            g_ptr[i] = 2.0f * diff / output.size();
        }

        if (s == 0) init_loss = loss;
        final_loss = loss;

        auto t2 = std::chrono::high_resolution_clock::now();
        layer.backward(grad_out);
        auto t3 = std::chrono::high_resolution_clock::now();

        optimizer.step();
        optimizer.zero_grad();

        total_fwd += std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_bwd += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    return {total_fwd / steps, total_bwd / steps, init_loss, final_loss};
}

int main() {
    std::cout << "Benchmarking ZenithBlock Sequency Ordering..." << std::endl;
    std::cout << "Channels: 64, Steps: 50" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "| Mode        | Fwd (ms) | Bwd (ms) | Init Loss | Final Loss |" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    auto res_nat = run_benchmark(64, false);
    std::cout << "| Natural     | "
              << std::fixed << std::setprecision(3) << res_nat.forward_ms << "    | "
              << res_nat.backward_ms << "    | "
              << std::scientific << res_nat.initial_loss << " | " << res_nat.final_loss << " |" << std::endl;

    auto res_seq = run_benchmark(64, true);
    std::cout << "| Sequency    | "
              << std::fixed << std::setprecision(3) << res_seq.forward_ms << "    | "
              << res_seq.backward_ms << "    | "
              << std::scientific << res_seq.initial_loss << " | " << res_seq.final_loss << " |" << std::endl;

    std::cout << "----------------------------------------------------------------" << std::endl;

    // Validate overhead
    double overhead = (res_seq.forward_ms - res_nat.forward_ms) / res_nat.forward_ms * 100.0;
    std::cout << "Sequency Overhead (Fwd): " << overhead << "%" << std::endl;

    // Validate loss improvement
    if (res_seq.final_loss < res_nat.final_loss) {
        std::cout << "Result: Sequency Ordering REDUCED Loss." << std::endl;
    } else {
        std::cout << "Result: Sequency Ordering did NOT reduce Loss (on this synthetic task)." << std::endl;
    }

    return 0;
}
