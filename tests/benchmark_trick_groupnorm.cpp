#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace dreidel;
using namespace dreidel::layers;
using namespace dreidel::optim;

int main() {
    std::cout << "Benchmark 5: GroupNorm vs No Norm (Small Batch Stability)" << std::endl;

    // Small Batch Size N=2
    size_t N = 2;
    size_t C = 64;
    Tensor<float> input({N, 16, 16, C});

    // Normalize Input to Unit Variance so Identity Init doesn't have "unfair" advantage
    // of just passing through high-variance input closer to high-variance target.
    input.random(0, 10);
    float mean = 0, var = 0;
    for(size_t i=0; i<input.size(); ++i) mean += input.data()[i];
    mean /= input.size();
    for(size_t i=0; i<input.size(); ++i) var += (input.data()[i]-mean)*(input.data()[i]-mean);
    var /= input.size();
    float std = std::sqrt(var);
    for(size_t i=0; i<input.size(); ++i) input.data()[i] = (input.data()[i]-mean)/std;

    Tensor<float> target = input;

    auto train = [&](bool use_gn) {
        ZenithBlock<float> model(C, 3, C);
        model.reinit("identity");
        model.set_group_norm(use_gn, 32);

        SimpleAdam<float> opt(0.01);
        opt.add_parameters(model.parameters(), model.gradients());

        float loss = 0;
        for(int i=0; i<10; ++i) {
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

    float loss_gn = train(true);
    float loss_none = train(false);

    std::cout << "Loss (GroupNorm): " << loss_gn << std::endl;
    std::cout << "Loss (No Norm): " << loss_none << std::endl;

    if (loss_gn <= loss_none) {
        std::cout << "SUCCESS: GroupNorm matched or outperformed No Norm." << std::endl;
    } else {
        std::cout << "WARNING: GroupNorm had higher loss (might need tuning)." << std::endl;
    }

    return 0;
}
