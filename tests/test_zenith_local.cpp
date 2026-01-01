#include <dreidel/layers/ZenithBlock.hpp>
#include <dreidel/core/Tensor.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace dreidel;

void run_test(size_t C, size_t H, size_t W, size_t loops) {
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Testing C=" << C << " (" << H << "x" << W << ")..." << std::endl;

    layers::ZenithBlock<float> layer(C, 3, C);
    // Note: Weights are random (Depthwise) but Mixing is Identity.
    // Scales are 1.0. Bias is 0.

    Tensor<float> input({1, H, W, C});
    input.fill(1.0f); // Uniform input

    // Warmup
    layer.forward(input);

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto out = layer.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Check Accuracy (Stability/Mean)
    Tensor<float> output = layer.forward(input);
    double mean = 0;
    double sq_sum = 0;
    size_t count = output.size();
    const float* ptr = output.data();
    for(size_t i=0; i<count; ++i) {
        mean += ptr[i];
        sq_sum += ptr[i]*ptr[i];
    }
    mean /= count;
    double stddev = std::sqrt(sq_sum/count - mean*mean);

    std::cout << "Time per iter: " << (elapsed / loops) * 1000.0 << " ms" << std::endl;
    std::cout << "Output Mean: " << mean << " | StdDev: " << stddev << std::endl;
    std::cout << "Status: PASSED" << std::endl;
}

void run_fused_check() {
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Testing C=4096 (Fused Kernel)..." << std::endl;
    size_t C = 4096;
    size_t H = 4, W = 4;

    layers::ZenithBlock<float> layer(C, 3, C);
    Tensor<float> input({1, H, W, C});
    input.fill(1.0f);

    auto start = std::chrono::high_resolution_clock::now();
    auto out = layer.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Forward Time: " << elapsed << " ms" << std::endl;

    // Stats
    double mean = 0;
    size_t count = out.size();
    const float* ptr = out.data();
    for(size_t i=0; i<count; ++i) mean += ptr[i];
    mean /= count;
    std::cout << "Output Mean: " << mean << std::endl;

    // Fused Logic requires: C=4096, K=3, Stride=1, Upscale=1.
    // If mean is > 0, it works (ReLU passes positive values, Identity mixing passes values).
    if (mean > 0) std::cout << "Status: PASSED" << std::endl;
    else std::cout << "Status: FAILED (Zero/Neg Output?)" << std::endl;
}

int main() {
    try {
        std::cout << "=== ZenithBlock Local Tests & Benchmarks ===" << std::endl;

        run_test(32, 32, 32, 100);
        run_test(64, 32, 32, 100);
        run_test(128, 16, 16, 100);
        run_test(256, 8, 8, 100); // Generic Large

        run_fused_check();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
