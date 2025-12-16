#include <dreidel/dreidel.hpp>
#include <dreidel/layers/Dense.hpp>
#include <dreidel/layers/LinearWHT.hpp>
#include <iostream>
#include <vector>
#include <numeric>

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

int main() {
    verify_parameter_reduction();
    return 0;
}
