#include <dreidel/dreidel.hpp>
#include <dreidel/layers/ZenithBlock.hpp>
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace dreidel;

void test_pack_apot() {
    std::cout << "Testing APoT Packing/Unpacking..." << std::endl;

    float values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, 0.1f, -0.1f, 100.0f, -100.0f};

    for(float v : values) {
        int8_t code = hal::AlienOps::pack_apot(v);
        float unpacked = hal::AlienOps::unpack_apot(code);

        // Check strict equality for 0
        if (v == 0.0f) {
            assert(unpacked == 0.0f);
        } else {
            // Check sign
            if (v > 0) assert(unpacked > 0);
            if (v < 0) assert(unpacked < 0);

            // Check power of 2
            // unpacked should be power of 2 close to v
            // log2(unpacked) should be integer
            float l = std::log2(std::abs(unpacked));
            assert(std::abs(l - std::round(l)) < 1e-5);
        }
    }
    std::cout << "APoT Test Passed." << std::endl;
}

void test_branchless_gate() {
    std::cout << "Testing Branchless Gate..." << std::endl;

    std::vector<float> data = {1.0f, -1.0f, 0.5f, -0.5f, 0.0f, -0.001f, 100.0f};
    std::vector<float> expected = {1.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 100.0f};

    hal::AlienOps::branchless_relu(data.data(), data.size());

    for(size_t i=0; i<data.size(); ++i) {
        assert(data[i] == expected[i]);
    }
    std::cout << "Branchless Gate Test Passed." << std::endl;
}

void test_zenith_block_phases() {
    std::cout << "Testing Zenith Block with Phase 5/6..." << std::endl;

    size_t batch = 1;
    size_t H = 16;
    size_t W = 16;
    size_t C = 64;

    Tensor<float> input({batch, H, W, C});
    input.random(-1.0f, 1.0f);

    layers::ZenithBlock<float> block(C, 3, C);

    auto output = block.forward(input);

    // Check output shape
    auto shape = output.shape();
    assert(shape[0] == batch);
    assert(shape[1] == H);
    assert(shape[2] == W);
    assert(shape[3] == C);

    // Check if output contains negative values (Branchless Gate should eliminate them if active)
    bool has_neg = false;
    const float* ptr = output.data();
    for(size_t i=0; i<output.size(); ++i) {
        if(ptr[i] < 0) has_neg = true;
    }

    assert(!has_neg);
    std::cout << "Zenith Block Forward Pass Passed (Output Non-Negative)." << std::endl;
}

int main() {
    test_pack_apot();
    test_branchless_gate();
    test_zenith_block_phases();
    return 0;
}
