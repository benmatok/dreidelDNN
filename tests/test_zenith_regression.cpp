#include <dreidel/layers/ZenithBlock.hpp>
#include <dreidel/layers/Conv2D.hpp>
#include <dreidel/optim/SGD.hpp>
#include <dreidel/core/Tensor.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <cassert>
#include <cmath>

using namespace dreidel;

// Helper to compute MSE
template<typename T>
T compute_mse(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.size() != b.size()) throw std::runtime_error("Size mismatch in MSE");
    const T* pa = a.data();
    const T* pb = b.data();
    T sum = 0;
    for(size_t i=0; i<a.size(); ++i) {
        T diff = pa[i] - pb[i];
        sum += diff * diff;
    }
    return sum / a.size();
}

void test_approximation() {
    std::cout << "[Test] Regression Approximation (Learning Identity)..." << std::endl;
    // Task: Learn Identity Mapping (Output ~= Input)
    // ZenithBlock initialized randomly will not be identity.
    // We train it to become identity.

    size_t N = 1, H = 8, W = 8, C = 32;
    // Explicit constructor to avoid ambiguity: (in, out, k, spec, ifwht, dilated, gating, stride, upscale)
    layers::ZenithBlock<float> model(C, C, 3, C, true, false, false, 1, 1);

    optim::SGD<float> optimizer(0.1f); // High learning rate for SGD
    optimizer.add_parameters(model.parameters(), model.gradients());

    Tensor<float> input({N, H, W, C});
    input.random(-1.0f, 1.0f);

    Tensor<float> target = input; // Identity target

    float initial_loss = 0;
    for(int i=0; i<200; ++i) {
        Tensor<float> output = model.forward(input);

        float loss = compute_mse(output, target);
        if (i==0) initial_loss = loss;

        // Backward: dL/dOutput = 2 * (Output - Target) / Size
        Tensor<float> grad_output = output;
        float* go = grad_output.data();
        const float* t = target.data();
        float scale = 2.0f / output.size();
        for(size_t j=0; j<output.size(); ++j) {
            go[j] = (go[j] - t[j]) * scale;
        }

        model.backward(grad_output);
        optimizer.step();
    }

    Tensor<float> final_out = model.forward(input);
    float final_loss = compute_mse(final_out, target);
    std::cout << "Initial Loss: " << initial_loss << " -> Final Loss: " << final_loss << std::endl;

    if (final_loss < initial_loss * 0.9f) {
        std::cout << "SUCCESS: Loss decreased." << std::endl;
    } else {
        std::cerr << "FAILURE: Loss did not decrease significantly." << std::endl;
        exit(1);
    }
}

void test_speedup() {
    std::cout << "\n[Test] Speedup vs Conv2D (C=64)..." << std::endl;
    size_t N = 1, H = 64, W = 64, C = 64;

    // Explicit constructor: (in, out, k, spec, ifwht, dilated, gating, stride, upscale)
    layers::ZenithBlock<float> zenith(C, C, 3, C, true, false, false, 1, 1);
    layers::Conv2D<float> conv(C, C, 3, 1, 1); // Same padding behavior approximately

    Tensor<float> input({N, H, W, C});
    input.random(-1.0f, 1.0f);

    // Warmup
    zenith.forward(input);
    conv.forward(input);

    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<10; ++i) zenith.forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    double zenith_time = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<10; ++i) conv.forward(input);
    end = std::chrono::high_resolution_clock::now();
    double conv_time = std::chrono::duration<double>(end - start).count();

    double speedup = conv_time / zenith_time;
    std::cout << "Zenith Time (10 runs): " << zenith_time << "s" << std::endl;
    std::cout << "Conv2D Time (10 runs): " << conv_time << "s" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Algorithmic Speedup baseline for C=64 is approx 64x.
    // We set a conservative threshold to ensure optimization is active.
    const double EXPECTED_SPEEDUP = 60.0;
    const double TOLERANCE = 0.1; // 10%
    const double MIN_SPEEDUP = EXPECTED_SPEEDUP * (1.0 - TOLERANCE);

    if (speedup >= MIN_SPEEDUP) {
        std::cout << "SUCCESS: ZenithBlock speedup (" << speedup << "x) is within expected range (>= " << MIN_SPEEDUP << "x)." << std::endl;
    } else {
        std::cerr << "FAILURE: ZenithBlock speedup (" << speedup << "x) is below expected threshold (" << MIN_SPEEDUP << "x)." << std::endl;
        exit(1);
    }
}

int main() {
    test_approximation();
    test_speedup();
    std::cout << "All regression tests passed." << std::endl;
    return 0;
}
