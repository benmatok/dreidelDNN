#include "../include/dreidel/models/ZenithHierarchicalAE.hpp"
#include "../include/dreidel/models/ConvBaselineAE.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <chrono>

using namespace dreidel;

// Helper to print vector
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "(";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i < v.size() - 1) os << ", ";
    }
    os << ")";
    return os;
}

template <typename T>
void benchmark_model(layers::Layer<T>* model, const Tensor<T>& input, const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();
    Tensor<T> out = model->forward(input);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Model: " << name << std::endl;
    std::cout << "Input Shape: " << input.shape() << std::endl;
    std::cout << "Output Shape: " << out.shape() << std::endl;
    std::cout << "Forward Time: " << elapsed.count() << "s" << std::endl;

    // Backward
    Tensor<T> grad_out(out.shape());
    grad_out.fill(1.0);

    start = std::chrono::high_resolution_clock::now();
    Tensor<T> grad_in = model->backward(grad_out);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "Backward Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
}

int main() {
    // Input Image: 256x256x3 (Small for quick test)
    // Spec says H/32 bottleneck. 256/32 = 8.
    // Base channels = 32.
    // Stage 1 C = 32*16 = 512.
    // Stage 2 C = 512*16 = 8192.
    // 8x8x8192 floats = 524288 floats = 2MB per item.

    size_t H = 256;
    size_t W = 256;
    size_t C = 3;
    size_t N = 1;

    Tensor<float> input({N, H, W, C});
    input.random(0.0f, 1.0f);

    std::cout << "Initializing ZenithHierarchicalAE..." << std::endl;
    models::ZenithHierarchicalAE<float> zenith_ae;
    benchmark_model(&zenith_ae, input, "ZenithHierarchicalAE");

    std::cout << "Initializing ConvBaselineAE..." << std::endl;
    models::ConvBaselineAE<float> conv_ae;
    benchmark_model(&conv_ae, input, "ConvBaselineAE");

    return 0;
}
