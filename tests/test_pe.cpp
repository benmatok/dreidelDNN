#include "../include/dreidel/layers/PositionalEmbedding.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace dreidel;
using namespace dreidel::layers;

int main() {
    std::cout << "Testing SinusoidalPositionalEmbedding2D..." << std::endl;

    // Create 4x4 embedding with 4 channels
    // d_model = 2.
    // pixel(h,w)[0] = sin(h / 10000^0) = sin(h)
    // pixel(h,w)[1] = cos(h / 10000^0) = cos(h)
    // pixel(h,w)[2] = sin(w / 10000^0) = sin(w)
    // pixel(h,w)[3] = cos(w / 10000^0) = cos(w)

    size_t H = 4, W = 4, C = 4;
    SinusoidalPositionalEmbedding2D<float> pe_layer(H, W, C);

    Tensor<float> input({1, H, W, C});
    input.fill(0);

    Tensor<float> output = pe_layer.forward(input);

    // Check (0, 0) -> sin(0)=0, cos(0)=1
    float* d = output.data();
    float p0_0 = d[0]; // sin(0)
    float p0_1 = d[1]; // cos(0)

    std::cout << "PE(0,0)[0] (Expected 0): " << p0_0 << std::endl;
    std::cout << "PE(0,0)[1] (Expected 1): " << p0_1 << std::endl;

    if (std::abs(p0_0) > 1e-5 || std::abs(p0_1 - 1.0f) > 1e-5) {
        std::cout << "FAILURE: PE values incorrect at origin." << std::endl;
        return 1;
    }

    // Check (1, 1) -> sin(1), cos(1)
    float p1_0 = d[(1*W+1)*C + 0]; // sin(1) Y
    float p1_2 = d[(1*W+1)*C + 2]; // sin(1) X

    std::cout << "PE(1,1)[0] (sin(1) ~0.841): " << p1_0 << std::endl;

    if (std::abs(p1_0 - std::sin(1.0f)) > 1e-5) {
        std::cout << "FAILURE: PE values incorrect at (1,1)." << std::endl;
        return 1;
    }

    // Check if X varies with w
    float p0_1_X = d[(0*W+1)*C + 2]; // sin(1)
    float p0_0_X = d[(0*W+0)*C + 2]; // sin(0)
    if (std::abs(p0_1_X - p0_0_X) < 1e-5) {
         std::cout << "FAILURE: X embedding does not vary with width." << std::endl;
         return 1;
    }

    std::cout << "SUCCESS: PE Injection Logic Verified." << std::endl;
    return 0;
}
