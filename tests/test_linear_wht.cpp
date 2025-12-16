#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../include/dreidel/layers/LinearWHT.hpp"
#include "../include/dreidel/core/Tensor.hpp"

using namespace dreidel;

void test_linear_wht_forward() {
    std::cout << "Testing LinearWHT Forward..." << std::endl;
    size_t batch = 2;
    size_t dim = 4;
    layers::LinearWHT<float> layer(dim);

    Tensor<float> input({batch, dim});
    input.fill(1.0f); // Input is all 1s

    // By default scale is random, let's force it for test
    auto params = layer.parameters();
    params[0]->fill(1.0f); // D = I

    // FWHT of [1, 1, 1, 1] -> [4, 0, 0, 0]
    // Unnormalized FWHT:
    // [1 1 1 1]
    // Step 1:
    // [2 2 0 0] (butterfly stride 1)
    // Step 2:
    // [4 0 0 0] (butterfly stride 2)

    Tensor<float> output = layer.forward(input);

    assert(output.shape()[0] == batch);
    assert(output.shape()[1] == dim);

    // Check first element of first batch
    // Should be 4.0
    float val0 = output.data()[0];
    float val1 = output.data()[1];

    std::cout << "Output[0]: " << val0 << " (Expected 4.0)" << std::endl;
    std::cout << "Output[1]: " << val1 << " (Expected 0.0)" << std::endl;

    if (std::abs(val0 - 4.0f) < 1e-5 && std::abs(val1) < 1e-5) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        exit(1);
    }
}

void test_linear_wht_backward() {
    std::cout << "Testing LinearWHT Backward..." << std::endl;
    size_t batch = 1;
    size_t dim = 4;
    layers::LinearWHT<float> layer(dim);

    Tensor<float> input({batch, dim});
    input.fill(1.0f);

    auto params = layer.parameters();
    params[0]->fill(0.5f); // Scale = 0.5

    // Forward
    // x * D = 0.5
    // FWHT(0.5) -> [2, 0, 0, 0]
    Tensor<float> output = layer.forward(input);

    // Backward
    // dL/dy = [1, 0, 0, 0]
    Tensor<float> grad_output({batch, dim});
    grad_output.fill(0.0f);
    grad_output.data()[0] = 1.0f;

    // dL/dScale = sum(FWHT(dL/dy) * x)
    // FWHT([1, 0, 0, 0]) -> [1, 1, 1, 1] (Symmetric property: FWHT of delta at 0 is constant vector 1)
    // Wait, let's check manually.
    // [1 0 0 0]
    // S1: [1 1 0 0]
    // S2: [1 1 1 1]
    // Yes.
    // grad_z = [1, 1, 1, 1]
    // dL/dScale = grad_z * input = [1, 1, 1, 1] * [1, 1, 1, 1] = [1, 1, 1, 1]

    // dL/dx = grad_z * scale = [1, 1, 1, 1] * 0.5 = [0.5, 0.5, 0.5, 0.5]

    Tensor<float> grad_input = layer.backward(grad_output);
    auto grads = layer.gradients();
    Tensor<float>* grad_scale = grads[0];

    std::cout << "Grad Input[0]: " << grad_input.data()[0] << " (Expected 0.5)" << std::endl;
    std::cout << "Grad Scale[0]: " << grad_scale->data()[0] << " (Expected 1.0)" << std::endl;

    bool pass = true;
    for(size_t i=0; i<dim; ++i) {
        if(std::abs(grad_input.data()[i] - 0.5f) > 1e-5) pass = false;
        if(std::abs(grad_scale->data()[i] - 1.0f) > 1e-5) pass = false;
    }

    if(pass) std::cout << "PASS" << std::endl;
    else {
        std::cout << "FAIL" << std::endl;
        exit(1);
    }
}

void test_topk() {
    std::cout << "Testing LinearWHT TopK..." << std::endl;
    size_t batch = 1;
    size_t dim = 4;
    size_t k = 2;
    layers::LinearWHT<float> layer(dim, k);

    // D = 1
    auto params = layer.parameters();
    params[0]->fill(1.0f);

    Tensor<float> input({batch, dim});
    // Set input such that FWHT gives distinct values
    // [1, 0, 0, 0] -> FWHT -> [1, 1, 1, 1] -> Not distinct
    // [2, 0, 0, 0] -> [2, 2, 2, 2]

    // Try random input or constructed
    // We want output (before topk) to be distinct.
    // Inverse FWHT of [4, 3, 2, 1] * (1/N) -> input
    // But FWHT is its own inverse (up to scale).
    // FWHT([4, 3, 2, 1]) -> [10, 2, 0, 4]
    // Let's use input = [4, 3, 2, 1].
    // FWHT([4, 3, 2, 1])
    // S1 (len 1):
    // 4+3=7, 4-3=1, 2+1=3, 2-1=1
    // [7, 1, 3, 1]
    // S2 (len 2):
    // 7+3=10, 1+1=2, 7-3=4, 1-1=0
    // [10, 2, 4, 0]
    // Abs values: 10, 2, 4, 0.
    // Top 2: 10 (idx 0), 4 (idx 2).
    // Result should be [10, 0, 4, 0].

    input.data()[0] = 4;
    input.data()[1] = 3;
    input.data()[2] = 2;
    input.data()[3] = 1;

    Tensor<float> output = layer.forward(input);

    std::cout << "Output: ";
    for(size_t i=0; i<dim; ++i) std::cout << output.data()[i] << " ";
    std::cout << std::endl;

    if (std::abs(output.data()[0] - 10.0f) < 1e-5 &&
        std::abs(output.data()[1]) < 1e-5 &&
        std::abs(output.data()[2] - 4.0f) < 1e-5 &&
        std::abs(output.data()[3]) < 1e-5) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        exit(1);
    }
}

int main() {
    test_linear_wht_forward();
    test_linear_wht_backward();
    test_topk();
    return 0;
}
