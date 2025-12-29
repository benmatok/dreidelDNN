#include <iostream>
#include <vector>
#include <cassert>
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"

using namespace dreidel;

int main() {
    size_t batch = 1;
    size_t H = 4;
    size_t W = 4;
    size_t C_in = 16;
    size_t C_out = 32;

    Tensor<int8_t> input({batch, H, W, C_in});
    input.fill(1);

    std::cout << "Testing ZenithBlock with Cin=" << C_in << " Cout=" << C_out << std::endl;

    // New constructor with Cin and Cout
    layers::ZenithBlock block(C_in, C_out, 3, std::max(C_in, C_out));

    Tensor<int8_t> output = block.forward(input);

    std::cout << "Output shape: " << output.shape()[0] << " " << output.shape()[1] << " " << output.shape()[2] << " " << output.shape()[3] << std::endl;

    if (output.shape()[3] != C_out) {
        std::cout << "FAILURE: Output channels " << output.shape()[3] << " != " << C_out << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: Output channels match Cout." << std::endl;
    return 0;
}
