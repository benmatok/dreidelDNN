#include "../include/dreidel/models/ComparativeAE.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace dreidel;
using namespace dreidel::models;
using namespace dreidel::optim;

int main() {
    std::cout << "Benchmark 3: CoordConv (Coordinate Regression Task)" << std::endl;

    // Task: Reproduce Gradient Ramp (X coordinate)
    size_t H = 32, W = 32;
    Tensor<float> input({1, H, W, 3});
    input.fill(0); // Black input

    Tensor<float> target({1, H, W, 3});
    for(size_t h=0; h<H; ++h) {
        for(size_t w=0; w<W; ++w) {
            target.data()[((0*H+h)*W+w)*3 + 0] = (float)w / W;
        }
    }

    auto train = [&](bool use_coord) {
        ZenithHierarchicalAE<float> model(16, use_coord);
        SimpleAdam<float> opt(0.001); // Reduced LR
        opt.add_parameters(model.parameters(), model.gradients());

        float loss = 0;
        for(int i=0; i<20; ++i) {
            opt.zero_grad();
            Tensor<float> out = model.forward(input);
            loss = 0;
            Tensor<float> grad = out;
            for(size_t k=0; k<out.size(); ++k) {
                float diff = out.data()[k] - target.data()[k];
                loss += diff*diff;
                grad.data()[k] = 2*diff/out.size();
            }
            model.backward(grad);
            opt.step();
        }
        return loss;
    };

    float loss_coord = train(true);
    float loss_no_coord = train(false);

    std::cout << "Loss (With CoordConv): " << loss_coord << std::endl;
    std::cout << "Loss (No CoordConv): " << loss_no_coord << std::endl;

    if (loss_coord < loss_no_coord) {
        std::cout << "SUCCESS: CoordConv converged faster/better." << std::endl;
    } else {
        std::cout << "FAILURE: CoordConv provided no benefit." << std::endl;
    }

    return 0;
}
