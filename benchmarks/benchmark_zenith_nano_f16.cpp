#include "../include/dreidel/models/ZenithNano.hpp"
#include "../include/dreidel/models/ZenithNanoInfer.hpp"
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>

using namespace dreidel;

void benchmark(const char* name, int n_iter, std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<n_iter; ++i) func();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / n_iter;
    std::cout << name << ": " << ms << " ms" << std::endl;
}

float max_diff(const Tensor<float>& a, const Tensor<float>& b) {
    float max_d = 0.0f;
    const float* pa = a.data();
    const float* pb = b.data();
    for(size_t i=0; i<a.size(); ++i) {
        float d = std::abs(pa[i] - pb[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

int main() {
    // 1. Setup Data
    Tensor<float> input({1, 512, 512, 3});
    // Random input
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* p = input.data();
    for(size_t i=0; i<input.size(); ++i) p[i] = dist(rng);

    // 2. Setup Models
    models::ZenithNano model_fp32;
    model_fp32.set_training(false);

    models::ZenithNanoInfer model_fp16;
    model_fp16.set_training(false);

    // Transfer weights (Initialize FP32 randomly first if needed, but they are random by default or zero?)
    // Layer parameters are usually initialized.
    // Let's grab parameters from FP32 model and load into FP16 model.
    // We need to access private members or use a helper?
    // ZenithNano doesn't expose components easily, but parameters() returns flat list.
    // Order: Compress (W, B?), Block1 (G,G,W,B?), Block2..., Block3..., Expand...

    // Actually, OptimizedConv2D parameters() returns empty in ZenithNano?
    // Let's check ZenithNano.hpp parameters().
    // It adds compress_, blocks, expand_.
    // OptimizedConv2D parameters() returns weights and bias?
    // Let's check OptimizedConv2D.hpp.
    // It returns {&weights_}. No bias in ZenithNano's usage (constructor 5th arg 0? No, 5th arg is padding, bias is optional).
    // ZenithNano calls OptimizedConv2D(..., 0) -> padding 0. Bias?
    // OptimizedConv2D constructor: (in, out, kh, kw, pad, bool bias=true).
    // ZenithNano: OptimizedConv2D<float>(192, 64, 1, 1, 0) -> bias=true (default).

    // ZenithNanoBlock parameters: {&gate_h, &gate_w, proj_conv params...}

    auto params = model_fp32.parameters();
    // Expected structure:
    // Compress: [W, B] (2)
    // Block1: [Gh, Gw, W, B] (4)
    // Block2: [Gh, Gw, W, B] (4)
    // Block3: [Gh, Gw, W, B] (4)
    // Expand: [W, B] (2)
    // Total: 16 tensors.

    if (params.size() != 16) {
        std::cerr << "Unexpected param count: " << params.size() << std::endl;
        // Maybe some bias is missing?
        // OptimizedConv2D might not have bias if configured so.
        // Let's check if ZenithNano uses bias.
        // OptimizedConv2D(..., padding) -> bias default true.
    }

    // Helper to extract
    int idx = 0;
    auto get_next = [&]() -> Tensor<float>& { return *params[idx++]; };
    auto get_next_ptr = [&]() -> Tensor<float>* { return params[idx++]; };

    // We need to organize them for load_weights
    // load_weights(c_w, c_b, e_w, e_b, b1_p, b2_p, b3_p)

    // Compress
    Tensor<float>* c_w = get_next_ptr();
    Tensor<float>* c_b = get_next_ptr();

    // Block 1
    std::vector<Tensor<float>> b1_p;
    b1_p.push_back(*get_next_ptr()); // Gh
    b1_p.push_back(*get_next_ptr()); // Gw
    b1_p.push_back(*get_next_ptr()); // W
    b1_p.push_back(*get_next_ptr()); // B

    // Block 2
    std::vector<Tensor<float>> b2_p;
    b2_p.push_back(*get_next_ptr());
    b2_p.push_back(*get_next_ptr());
    b2_p.push_back(*get_next_ptr());
    b2_p.push_back(*get_next_ptr());

    // Block 3
    std::vector<Tensor<float>> b3_p;
    b3_p.push_back(*get_next_ptr());
    b3_p.push_back(*get_next_ptr());
    b3_p.push_back(*get_next_ptr());
    b3_p.push_back(*get_next_ptr());

    // Expand
    Tensor<float>* e_w = get_next_ptr();
    Tensor<float>* e_b = get_next_ptr();

    model_fp16.load_weights(*c_w, c_b, *e_w, e_b, b1_p, b2_p, b3_p);

    // 3. Compare Output
    std::cout << "Running Forward FP32..." << std::endl;
    auto out_fp32 = model_fp32.forward(input);

    std::cout << "Running Forward FP16..." << std::endl;
    auto out_fp16 = model_fp16.forward(input);

    float diff = max_diff(out_fp32, out_fp16);
    std::cout << "Max Diff (FP32 vs FP16): " << diff << std::endl;

    // 4. Benchmark
    int n = 10;
    benchmark("ZenithNano FP32", n, [&]() {
        model_fp32.forward(input);
    });

    benchmark("ZenithNano FP16", n, [&]() {
        model_fp16.forward(input);
    });

    return 0;
}
