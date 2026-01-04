#include "../include/dreidel/models/ComparativeAE.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace dreidel;
using namespace dreidel::models;
using namespace dreidel::optim;

int main() {
    std::cout << "Testing Improved ZenithHierarchicalAE..." << std::endl;

    // 1. Instantiate Model
    // 3 channels input.
    ZenithHierarchicalAE<float> model(16); // Base filters = 16

    // 2. Instantiate Optimizer with Coordinate-Wise Clipping
    SimpleAdam<float> optimizer(1e-3);
    optimizer.set_coordinate_wise_clipping(true, 0.1f); // Clip at 0.1
    optimizer.add_parameters(model.parameters(), model.gradients());

    // 3. Create Dummy Data
    // Batch=2, H=64, W=64, C=3
    Tensor<float> input({2, 64, 64, 3});
    input.random(0.0f, 1.0f);

    // 4. Forward Pass
    std::cout << "Running Forward..." << std::endl;
    Tensor<float> output = model.forward(input);
    std::cout << "Output Shape: " << output.shape()[0] << "x" << output.shape()[1] << "x" << output.shape()[2] << "x" << output.shape()[3] << std::endl;

    if (output.shape()[1] != 64 || output.shape()[2] != 64 || output.shape()[3] != 3) {
        std::cerr << "Error: Output shape mismatch!" << std::endl;
        return 1;
    }

    // 5. Backward Pass
    std::cout << "Running Backward..." << std::endl;
    Tensor<float> grad_output = output; // Dummy gradient (MSE loss grad w.r.t 0 target would be 2*output)
    grad_output.fill(0.1f);

    model.backward(grad_output);

    // 6. Optimizer Step
    std::cout << "Optimizer Step..." << std::endl;
    optimizer.step();

    std::cout << "Test Passed Successfully!" << std::endl;
    return 0;
}
