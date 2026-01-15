#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "dreidel/models/ZenithNano.hpp"
#include "dreidel/core/Tensor.hpp"

using namespace dreidel;

int main() {
    std::cout << "=== Zenith-Nano Validation ===\n" << std::endl;

    models::ZenithNano model;

    // 1. Grid Test (Sanity)
    // Feed White Image (1.0). If output is smooth white (1.0), mixing is likely working.
    // If gates are zero, output might be zero. Gates init to 1.0.
    // FWHT is linear.
    std::cout << "Running Grid Test (White Image)..." << std::endl;
    Tensor<float> input({1, 512, 512, 3});
    input.fill(1.0f);

    // Set inference mode
    model.set_training(false);

    // Warmup
    for(int i=0; i<5; ++i) model.forward(input);

    Tensor<float> output = model.forward(input);

    // Check center pixel
    float* out_ptr = output.data();
    float center_val = out_ptr[(256*512 + 256)*3];
    std::cout << "Center Pixel Value: " << center_val << " (Expected ~1.0 if Identity)" << std::endl;

    // Check smoothness/checkerboard
    // Sample a few pixels
    float val1 = out_ptr[0];
    float val2 = out_ptr[1];
    float val3 = out_ptr[512*3]; // Row 1
    std::cout << "Sample Pixels: " << val1 << ", " << val2 << ", " << val3 << std::endl;

    // 2. Benchmark
    std::cout << "\nBenchmarking (100 iters)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {
        model.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;

    std::cout << "Avg Inference: " << elapsed << " ms" << std::endl;

    double target = 7.0; // 7ms hard limit
    if (elapsed < target) {
        std::cout << "Target (<" << target << "ms): PASS" << std::endl;
    } else {
        std::cout << "Target (<" << target << "ms): FAIL" << std::endl;
    }

    return 0;
}
