#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/core/Tensor.hpp"

using namespace dreidel;

void test_downsample() {
    std::cout << "Testing Downsample (256 -> 128)..." << std::endl;
    size_t N = 1, H = 8, W = 8, C_in = 256, C_out = 128;
    // Constructor: in, out, kernel, spectral_dim
    layers::ZenithBlock<float> block(C_in, C_out, 3, C_in);

    Tensor<float> input({N, H, W, C_in});
    input.random(0, 1);

    Tensor<float> output = block.forward(input);
    auto shape = output.shape();

    std::cout << "Output shape: " << shape[0] << "," << shape[1] << "," << shape[2] << "," << shape[3] << std::endl;

    assert(shape[0] == N);
    assert(shape[1] == H);
    assert(shape[2] == W);
    assert(shape[3] == C_out);

    // Backward check
    Tensor<float> grad_out({N, H, W, C_out});
    grad_out.random(0, 1);
    Tensor<float> grad_in = block.backward(grad_out);

    auto g_shape = grad_in.shape();
    assert(g_shape[3] == C_in);
    std::cout << "Downsample Passed." << std::endl;
}

void test_upsample() {
    std::cout << "Testing Upsample (128 -> 256)..." << std::endl;
    size_t N = 1, H = 8, W = 8, C_in = 128, C_out = 256;
    layers::ZenithBlock<float> block(C_in, C_out, 3, C_in);

    Tensor<float> input({N, H, W, C_in});
    input.random(0, 1);

    Tensor<float> output = block.forward(input);
    auto shape = output.shape();

    std::cout << "Output shape: " << shape[0] << "," << shape[1] << "," << shape[2] << "," << shape[3] << std::endl;

    assert(shape[3] == C_out);

    // Backward check
    Tensor<float> grad_out({N, H, W, C_out});
    grad_out.random(0, 1);
    Tensor<float> grad_in = block.backward(grad_out);

    auto g_shape = grad_in.shape();
    assert(g_shape[3] == C_in);
    std::cout << "Upsample Passed." << std::endl;
}

int main() {
    try {
        test_downsample();
        test_upsample();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
