#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace dreidel;
using namespace dreidel::layers;

bool check_bias_identity() {
    std::cout << "[Test] Bias Identity Check..." << std::endl;
    // 1 Batch, 4x4 Image, 1 Channel
    size_t C = 1;
    size_t H = 4;
    size_t W = 4;
    ZenithBlock<float> block(C, 1, C, 1024, false, false); // No gating, No APoT

    // Access parameters
    auto params = block.parameters();
    Tensor<float>* weights = params[0];
    Tensor<float>* scales = params[1];
    Tensor<float>* bias = params[2];

    // Set weights to 0
    weights->fill(0.0f);
    scales->fill(1.0f);
    // Set Bias to 5.0
    bias->fill(5.0f);

    Tensor<float> input({1, H, W, C});
    input.fill(1.0f);

    Tensor<float> output = block.forward(input);

    // Verify Output is 5.0 everywhere
    const float* out_ptr = output.data();
    for(size_t i=0; i<output.size(); ++i) {
        if (std::abs(out_ptr[i] - 5.0f) > 1e-5) {
            std::cout << "FAIL: Output " << out_ptr[i] << " != 5.0" << std::endl;
            return false;
        }
    }
    std::cout << "PASS: Positive Bias propagated correctly." << std::endl;

    // Test Negative Bias (ReLU should clip)
    bias->fill(-2.0f);
    output = block.forward(input);
    out_ptr = output.data();
    for(size_t i=0; i<output.size(); ++i) {
        if (std::abs(out_ptr[i] - 0.0f) > 1e-5) {
            std::cout << "FAIL: Output " << out_ptr[i] << " != 0.0 for negative bias" << std::endl;
            return false;
        }
    }
    std::cout << "PASS: Negative Bias clipped correctly." << std::endl;

    return true;
}

bool check_apot_bias() {
    std::cout << "[Test] APoT Bias Check..." << std::endl;
    size_t C = 1;
    ZenithBlock<float> block(C, 1, C, 1024, false, true); // Enable APoT

    auto params = block.parameters();
    params[0]->fill(0.0f); // Weights 0
    params[1]->fill(1.0f); // Scale 1 (will be quantized to 1.0)

    // Set Bias to 3.0. log2(3)=1.585 -> round to 2 -> 2^2 = 4.0.
    params[2]->fill(3.0f);

    Tensor<float> input({1, 1, 1, C});
    input.fill(0.0f);

    Tensor<float> output = block.forward(input);

    float val = output.data()[0];
    std::cout << "Input Bias: 3.0 -> APoT Output: " << val << std::endl;

    if (std::abs(val - 4.0f) > 1e-5) {
        std::cout << "FAIL: Expected 4.0 (Nearest Power of 2 to 3.0)" << std::endl;
        return false;
    }
    std::cout << "PASS: APoT Quantization on Bias works." << std::endl;
    return true;
}

bool check_bias_gradient() {
    std::cout << "[Test] Bias Gradient Check..." << std::endl;
    // 1 Batch, 2x2 Image, 1 Channel
    size_t C = 1;
    size_t H = 2;
    size_t W = 2;
    ZenithBlock<float> block(C, 1, C, 1024, false, false);

    auto params = block.parameters();
    Tensor<float>* bias = params[2];
    bias->fill(1.0f); // Positive bias to ensure ReLU passes

    // Zero weights so output is just bias (which is 1.0)
    params[0]->fill(0.0f);
    params[1]->fill(1.0f);

    Tensor<float> input({1, H, W, C});
    input.fill(0.0f);

    // Forward
    block.forward(input);

    // Backward with all 1s
    Tensor<float> grad_out({1, H, W, C});
    grad_out.fill(1.0f);

    block.backward(grad_out);

    auto grads = block.gradients();
    Tensor<float>* g_bias = grads[2];

    // Grad Bias should be sum of grad_out where output > 0
    // Since output is 1.0 everywhere, it should be sum(grad_out) = 1 * 2 * 2 * 1 = 4.0

    if (std::abs(g_bias->data()[0] - 4.0f) > 1e-5) {
        std::cout << "FAIL: Grad Bias " << g_bias->data()[0] << " != 4.0" << std::endl;
        return false;
    }
    std::cout << "PASS: Bias Gradient accumulated correctly." << std::endl;

    // Test ReLU Masking in Backward
    bias->fill(-1.0f); // Output will be 0
    block.forward(input);
    block.backward(grad_out);

    // Grad bias should be 0 because ReLU blocked it
    if (std::abs(g_bias->data()[0] - 0.0f) > 1e-5) {
        std::cout << "FAIL: Grad Bias " << g_bias->data()[0] << " != 0.0 (ReLU Masking Failed)" << std::endl;
        return false;
    }
    std::cout << "PASS: ReLU Gradient Masking works." << std::endl;

    return true;
}

bool check_training_flag() {
    std::cout << "[Test] Training Flag Check..." << std::endl;
    size_t C = 1;
    ZenithBlock<float> block(C, 1, C, 1024, false, false);

    // Disable training (Inference Mode)
    block.set_training(false);

    Tensor<float> input({1, 1, 1, C});
    input.fill(0.0f);
    block.forward(input);

    // Backward should warn/fail gracefully because input wasn't cached
    // In our implementation, we print to cerr and return 0 gradient
    std::cout << "Calling backward in inference mode (expect error message)..." << std::endl;
    Tensor<float> grad_out({1, 1, 1, C});
    grad_out.fill(1.0f);
    Tensor<float> grad_in = block.backward(grad_out);

    // We can't easily check cerr, but we can check if it didn't crash
    std::cout << "PASS: Backward returned without crash." << std::endl;
    return true;
}

template <size_t C>
bool check_kernel() {
    std::cout << "[Test] AVX Kernel Check C=" << C << "..." << std::endl;
    ZenithBlock<float> block(C, 1, C, 1024*1024, false, false);

    // Identity Test: W=0, Scale=1, Bias=0 -> Output 0
    auto params = block.parameters();
    params[0]->fill(0.0f);
    params[1]->fill(1.0f);
    params[2]->fill(0.0f);

    Tensor<float> input({1, 1, 1, C});
    float* in_ptr = input.data();
    for(size_t i=0; i<C; ++i) in_ptr[i] = (float)(i+1); // Dummy input

    // With weights=0, eyes output is 0. Mixer(0) = 0.
    Tensor<float> output = block.forward(input);

    // Output should be 0
    float sum = 0;
    for(size_t i=0; i<C; ++i) sum += std::abs(output.data()[i]);
    if (sum > 1e-4) {
        std::cout << "FAIL: Identity (Zero) check failed for C=" << C << " sum=" << sum << std::endl;
        return false;
    }

    // Bias pass-through test
    params[2]->fill(10.0f);
    output = block.forward(input);
    // Output should be 10.0 everywhere (ReLU(0+10))
    for(size_t i=0; i<C; ++i) {
        if (std::abs(output.data()[i] - 10.0f) > 1e-4) {
            std::cout << "FAIL: Bias pass-through failed for C=" << C << " val=" << output.data()[i] << std::endl;
            return false;
        }
    }

    std::cout << "PASS: Kernel C=" << C << " functional." << std::endl;
    return true;
}

int main() {
    bool p1 = check_bias_identity();
    bool p2 = check_apot_bias();
    bool p3 = check_bias_gradient();
    bool p4 = check_training_flag();

    // Verify kernels
    bool k16 = check_kernel<16>();
    bool k32 = check_kernel<32>();
    bool k64 = check_kernel<64>();
    bool k128 = check_kernel<128>();

    if (p1 && p2 && p3 && p4 && k16 && k32 && k64 && k128) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "TESTS FAILED" << std::endl;
        return 1;
    }
}
