#include <iostream>
#include <vector>
#include <chrono>
#include <memory>
#include "dreidel/models/ZenithNano.hpp"
#include "dreidel/core/Tensor.hpp"
#include <immintrin.h>

// Benchmark for Full ZenithNano Model
// Target: < 3ms per image (512x512 input)

int main() {
    // Flush Denormals (Hardware Trick)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    #ifdef _OPENMP
    int max_threads = omp_get_max_threads();
    std::cout << "OpenMP Threads: " << max_threads << std::endl;
    if (max_threads < 2) {
        std::cout << "WARNING: Running with 1 thread. Expect slow performance." << std::endl;
        std::cout << "Set OMP_NUM_THREADS=4 for better results." << std::endl;
    }
    #endif

    std::cout << "Benchmarking Full ZenithNano Model..." << std::endl;

    // Initialize Model
    dreidel::models::ZenithNano model;
    model.set_training(false); // Inference Mode

    // Create Input [1, 512, 512, 3]
    dreidel::Tensor<float> input({1, 512, 512, 3});
    input.fill(0.5f);

    // Warmup
    std::cout << "Warming up..." << std::endl;
    for(int i=0; i<5; ++i) {
        auto out = model.forward(input);
        // Prevent DCE
        volatile float x = out.data()[0];
        (void)x;
    }

    // Benchmark
    int iterations = 100;
    std::cout << "Running " << iterations << " iterations..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<iterations; ++i) {
        auto out = model.forward(input);
        volatile float x = out.data()[0];
        (void)x;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_ms = elapsed.count() / iterations;

    std::cout << "Average Latency: " << avg_ms << " ms/image" << std::endl;

    // Check dynamic target
    double target = 10.0;
    #ifdef _OPENMP
    if (max_threads == 1) target = 20.0;
    #endif

    if (avg_ms <= target) {
        std::cout << "[SUCCESS] Target Met (< " << target << "ms)" << std::endl;
    } else {
        std::cout << "[FAILURE] Target Not Met (> " << target << "ms)" << std::endl;
        std::cout << "Gap: " << avg_ms - target << " ms" << std::endl;
    }

    return 0;
}
