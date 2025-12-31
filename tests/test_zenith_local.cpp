#include <dreidel/layers/ZenithBlock.hpp>
#include <dreidel/core/Tensor.hpp>
#include <iostream>
#include <cassert>

using namespace dreidel;

int main() {
    std::cout << "Testing ZenithBlock with Locally Connected Mixing..." << std::endl;

    // Test Case 1: C=32
    {
        std::cout << "Testing C=32..." << std::endl;
        layers::ZenithBlock<float> layer(32, 3, 32);
        Tensor<float> input({1, 32, 32, 32});
        input.fill(1.0f);
        Tensor<float> output = layer.forward(input);
        assert(output.shape() == std::vector<size_t>({1, 32, 32, 32}));
        std::cout << "C=32 Passed." << std::endl;
    }

    // Test Case 2: C=64
    {
        std::cout << "Testing C=64..." << std::endl;
        layers::ZenithBlock<float> layer(64, 3, 64);
        Tensor<float> input({1, 32, 32, 64});
        input.fill(0.5f);
        Tensor<float> output = layer.forward(input);
        assert(output.shape() == std::vector<size_t>({1, 32, 32, 64}));
        std::cout << "C=64 Passed." << std::endl;
    }

    // Test Case 3: C=128
    {
        std::cout << "Testing C=128..." << std::endl;
        layers::ZenithBlock<float> layer(128, 3, 128);
        Tensor<float> input({1, 16, 16, 128});
        input.fill(0.1f);
        Tensor<float> output = layer.forward(input);
        assert(output.shape() == std::vector<size_t>({1, 16, 16, 128}));
        std::cout << "C=128 Passed." << std::endl;
    }

    // Test Case 4: C=256 (Generic Large)
    {
        std::cout << "Testing C=256 (Generic Large)..." << std::endl;
        layers::ZenithBlock<float> layer(256, 3, 256);
        Tensor<float> input({1, 4, 4, 256});
        input.fill(1.0f);
        Tensor<float> output = layer.forward(input);
        assert(output.shape() == std::vector<size_t>({1, 4, 4, 256}));
        std::cout << "C=256 Passed." << std::endl;
    }

    // Test Case 5: C=64 -> 1
    {
        std::cout << "Testing C=64 -> 1..." << std::endl;
        layers::ZenithBlock<float> layer(64, 1, 3, 64, true, false, false, 1, 1);
        Tensor<float> input({1, 8, 8, 64});
        input.fill(1.0f);
        Tensor<float> output = layer.forward(input);
        assert(output.shape() == std::vector<size_t>({1, 8, 8, 1}));
        std::cout << "C=64->1 Passed." << std::endl;
    }

    // Test Case 6: Generic Fallback (Small Power of 2, e.g. 8)
    // C=8 might use optimized eyes but fallback mixer?
    // fwht8_avx2 exists? Yes. But mixer?
    // forward_avx2_generic_large_mixer checks > 128.
    // forward_avx2_c32/64/128 check exact match.
    // So C=8 will fall through to Generic OpenMP loop.
    {
        std::cout << "Testing C=8 (Generic Small)..." << std::endl;
        layers::ZenithBlock<float> layer(8, 3, 8);
        Tensor<float> input({1, 8, 8, 8});
        input.fill(1.0f);
        Tensor<float> output = layer.forward(input);
        assert(output.shape() == std::vector<size_t>({1, 8, 8, 8}));
        std::cout << "C=8 Passed." << std::endl;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
