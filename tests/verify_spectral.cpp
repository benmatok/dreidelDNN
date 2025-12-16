#include <dreidel/dreidel.hpp>
#include <dreidel/layers/Dense.hpp>
#include <dreidel/layers/LinearWHT.hpp>
#include <dreidel/algo/WHT.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

using namespace dreidel;

void verify_parameter_reduction() {
    std::cout << "--- Verifying Spectral Parameter Reduction ---" << std::endl;

    size_t N = 1024; // Must be power of 2 for WHT
    std::cout << "Dimension N = " << N << std::endl;

    // 1. Dense Layer
    layers::Dense<float> dense(N, N);
    auto dense_params = dense.parameters();
    size_t dense_param_count = 0;
    for (auto* p : dense_params) {
        dense_param_count += p->size();
    }
    // W: N*N, b: N => N^2 + N
    size_t expected_dense = N * N + N;
    std::cout << "Dense Layer Parameters: " << dense_param_count
              << " (Expected: " << expected_dense << ")" << std::endl;

    // 2. LinearWHT Layer
    layers::LinearWHT<float> spectral(N);
    auto spectral_params = spectral.parameters();
    size_t spectral_param_count = 0;
    for (auto* p : spectral_params) {
        spectral_param_count += p->size();
    }
    // D: N => N
    size_t expected_spectral = N;
    std::cout << "LinearWHT Layer Parameters: " << spectral_param_count
              << " (Expected: " << expected_spectral << ")" << std::endl;

    // 3. Ratio
    float ratio = (float)dense_param_count / (float)spectral_param_count;
    std::cout << "Reduction Ratio: " << ratio << "x" << std::endl;

    if (dense_param_count == expected_dense && spectral_param_count == expected_spectral && ratio > N) {
        std::cout << "PASS: Significant parameter reduction observed." << std::endl;
    } else {
        std::cout << "FAIL: Parameter counts do not match expectations." << std::endl;
        exit(1);
    }
}

void verify_inverse_fwht() {
    std::cout << "--- Verifying Inverse FWHT ---" << std::endl;
    size_t N = 1024;
    Tensor<float> t({1, N});
    t.random(-1.0, 1.0);
    Tensor<float> original = t;

    // Forward
    algo::WHT::FWHT(t);
    // Inverse
    algo::WHT::InverseFWHT(t);

    // Check difference
    float max_diff = 0;
    for(size_t i=0; i<N; ++i) {
        float diff = std::abs(t.data()[i] - original.data()[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "Max diff: " << max_diff << std::endl;
    if (max_diff < 1e-4) {
        std::cout << "PASS: Inverse FWHT restores original." << std::endl;
    } else {
        std::cout << "FAIL: Inverse FWHT failed." << std::endl;
        exit(1);
    }
}

void verify_throughput() {
    std::cout << "--- Verifying Throughput ---" << std::endl;
    size_t N = 1024 * 1024 * 64; // 64M floats = 256MB
    // Ensure total elements is multiple of N for WHT if dims matter, here we treat as 1D N?
    // WHT is usually on last dim. Let's make last dim 1024.
    size_t dim = 1024;
    size_t batch = N / dim;

    Tensor<float> t({batch, dim});
    // Touch memory
    t.fill(1.0);

    // Benchmark FWHT
    auto start = std::chrono::high_resolution_clock::now();
    algo::WHT::FWHT(t);
    auto end = std::chrono::high_resolution_clock::now();
    double fwht_time = std::chrono::duration<double>(end - start).count();
    double fwht_gb_s = (N * sizeof(float)) / fwht_time / 1e9;

    // Benchmark Memcpy (Reference)
    Tensor<float> t2({batch, dim});
    start = std::chrono::high_resolution_clock::now();
    std::copy(t.data(), t.data() + N, t2.data());
    end = std::chrono::high_resolution_clock::now();
    double copy_time = std::chrono::duration<double>(end - start).count();
    double copy_gb_s = (N * sizeof(float)) / copy_time / 1e9;

    std::cout << "FWHT Throughput: " << fwht_gb_s << " GB/s" << std::endl;
    std::cout << "Memcpy Throughput: " << copy_gb_s << " GB/s" << std::endl;

    double ratio = fwht_gb_s / copy_gb_s;
    std::cout << "Ratio: " << ratio * 100 << "%" << std::endl;

    if (ratio > 0.5) { // Relaxed check
        std::cout << "PASS: Throughput acceptable." << std::endl;
    } else {
        std::cout << "WARNING: Throughput low (" << ratio * 100 << "%)." << std::endl;
    }
}

int main() {
    verify_parameter_reduction();
    verify_inverse_fwht();
    verify_throughput();
    return 0;
}
