#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../include/dreidel/layers/Conv3DSpectral.hpp"
#include "../include/dreidel/core/Tensor.hpp"

using namespace dreidel;

void test_conv3d_spectral_forward() {
    std::cout << "Testing Conv3DSpectral Forward..." << std::endl;
    size_t batch = 1;
    size_t d = 3, h = 3, w = 3;
    size_t channels = 2; // Must be power of 2 for FWHT
    size_t kernel_size = 1; // 1x1x1 to simplify check

    layers::Conv3DSpectral<float> layer(kernel_size, channels, d, h, w);

    // Set weights to identity (all 1s)
    auto params = layer.parameters();
    // params[0] is spatial weights (K, K, K, C) -> (1, 1, 1, 2)
    params[0]->fill(1.0f);

    // params[1] is LinearWHT scale (1, C) -> (1, 2)
    params[1]->fill(1.0f);

    Tensor<float> input({batch, d, h, w, channels});
    input.fill(1.0f);

    // Forward logic:
    // 1. Spatial Conv (1x1x1 all 1s): Output = Input * 1 = Input (1.0)
    // 2. LinearWHT: Input * Scale(1.0) = Input (1.0)
    // 3. FWHT([1, 1]) -> [2, 0]

    Tensor<float> output = layer.forward(input);

    assert(output.shape().size() == 5);

    float val0 = output.data()[0]; // Should be 2.0 (channel 0)
    float val1 = output.data()[1]; // Should be 0.0 (channel 1)

    std::cout << "Output[0]: " << val0 << " (Expected 2.0)" << std::endl;
    std::cout << "Output[1]: " << val1 << " (Expected 0.0)" << std::endl;

    if (std::abs(val0 - 2.0f) < 1e-5 && std::abs(val1) < 1e-5) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        exit(1);
    }
}

void test_conv3d_spectral_backward() {
    std::cout << "Testing Conv3DSpectral Backward..." << std::endl;
    size_t batch = 1;
    size_t d = 3, h = 3, w = 3;
    size_t channels = 2;
    size_t kernel_size = 1;

    layers::Conv3DSpectral<float> layer(kernel_size, channels, d, h, w);

    auto params = layer.parameters();
    params[0]->fill(1.0f); // Spatial W = 1
    params[1]->fill(1.0f); // LinearWHT D = 1

    Tensor<float> input({batch, d, h, w, channels});
    input.fill(1.0f);

    Tensor<float> output = layer.forward(input); // [2, 0, 2, 0...]

    // grad_output = 1
    Tensor<float> grad_output(output.shape());
    grad_output.fill(1.0f);

    // Backward:
    // 1. LinearWHT Backward
    //    grad_z = FWHT(grad_output) = FWHT([1, 1]) = [2, 0]
    //    grad_spatial_out = grad_z * scale(1) = [2, 0]
    // 2. Conv3D Backward (1x1 conv with W=1)
    //    dL/dX = grad_spatial_out * W = [2, 0] * 1 = [2, 0]
    //    dL/dW = grad_spatial_out * X = [2, 0] * 1 = [2, 0] (Sum over spatial)

    Tensor<float> grad_input = layer.backward(grad_output);

    float gi0 = grad_input.data()[0]; // 2.0
    float gi1 = grad_input.data()[1]; // 0.0

    std::cout << "Grad Input[0]: " << gi0 << " (Expected 2.0)" << std::endl;
    std::cout << "Grad Input[1]: " << gi1 << " (Expected 0.0)" << std::endl;

    if (std::abs(gi0 - 2.0f) < 1e-5 && std::abs(gi1) < 1e-5) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        exit(1);
    }
}

int main() {
    test_conv3d_spectral_forward();
    test_conv3d_spectral_backward();
    return 0;
}
