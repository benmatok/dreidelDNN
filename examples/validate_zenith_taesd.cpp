#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include "dreidel/models/ZenithTAESD.hpp"
#include "dreidel/hal/x86.hpp"
#include "dreidel/core/Tensor.hpp"

using namespace dreidel;

// Helper to check FWHT Invertibility
void check_fwht_invertibility() {
    std::cout << "Checking FWHT Invertibility..." << std::endl;
    size_t N = 128; // Spatial
    size_t C = 64;  // Channels

    std::vector<float> data(N * C);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for(auto& v : data) v = dist(rng);

    std::vector<float> original = data;

    // FWHT
    hal::x86::fwht_1d_vectorized_avx2(data.data(), N, C);

    // IFWHT (Same kernel)
    hal::x86::fwht_1d_vectorized_avx2(data.data(), N, C);

    // Scale by 1/N
    float scale = 1.0f / N;
    float max_err = 0.0f;
    for(size_t i=0; i<data.size(); ++i) {
        float val = data[i] * scale;
        float diff = std::abs(val - original[i]);
        if(diff > max_err) max_err = diff;
    }

    std::cout << "FWHT Max Error: " << max_err << std::endl;
    if (max_err < 1e-4) std::cout << "FWHT Invertibility: PASS" << std::endl;
    else std::cout << "FWHT Invertibility: FAIL" << std::endl;
}

int main() {
    std::cout << "=== Zenith-TAESD Validation & Benchmark ===" << std::endl;

    // 1. Check Kernels
    check_fwht_invertibility();

    // 2. Instantiate Model
    size_t H = 512;
    size_t W = 512;
    size_t C = 3;
    models::ZenithTAESD<float> model(C, 4, 64, H, W);

    std::cout << "Model instantiated. Parameter count: " << std::endl;
    size_t params = 0;
    for(auto* p : model.parameters()) params += p->size();
    std::cout << "Total Parameters: " << params << std::endl;
    std::cout << "Estimated Size (FP32): " << params * 4 / 1024 / 1024 << " MB" << std::endl;

    // 3. Inference Benchmark
    std::cout << "\nBenchmarking Inference (512x512)..." << std::endl;
    Tensor<float> input({1, H, W, C});
    input.random(0.0f, 1.0f);

    // Warmup
    for(int i=0; i<5; ++i) model.forward(input);

    int iterations = 20;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        model.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Avg Latency per Image: " << elapsed.count() / iterations << " ms" << std::endl;

    // Check Output Shape
    Tensor<float> out = model.forward(input);
    auto shape = out.shape();
    std::cout << "Output Shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "]" << std::endl;

    if (shape[1] == H && shape[2] == W && shape[3] == C) {
        std::cout << "Shape Check: PASS" << std::endl;
    } else {
        std::cout << "Shape Check: FAIL" << std::endl;
    }

    return 0;
}
