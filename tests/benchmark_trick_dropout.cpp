#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace dreidel;
using namespace dreidel::layers;
using namespace dreidel::optim;

int main() {
    std::cout << "Benchmark 2: Spectral Dropout (Regularization Check)" << std::endl;

    // Train a single large block on a tiny dataset (Overfitting scenario)
    // Goal: Dropout should slow down convergence (training loss higher initially) or show stochasticity.

    size_t C = 64;
    size_t H = 8, W = 8;
    Tensor<float> input({1, H, W, C});
    input.random(0, 1);
    Tensor<float> target = input; // Identity task

    auto train_run = [&](float dropout_rate) {
        ZenithBlock<float> model(C, 3, C);
        model.reinit("identity"); // Start from good state
        model.set_spectral_dropout(dropout_rate);

        SimpleAdam<float> opt(0.01);
        opt.add_parameters(model.parameters(), model.gradients());

        float final_loss = 0;
        for(int i=0; i<50; ++i) {
            opt.zero_grad();
            Tensor<float> out = model.forward(input);

            // MSE Loss
            float loss = 0;
            Tensor<float> grad_out = out;
            for(size_t k=0; k<out.size(); ++k) {
                float diff = out.data()[k] - target.data()[k];
                loss += diff * diff;
                grad_out.data()[k] = 2 * diff / out.size();
            }
            if (i==49) final_loss = loss;

            model.backward(grad_out);
            opt.step();
        }
        return final_loss;
    };

    float loss_nodrop = train_run(0.0f);
    float loss_drop = train_run(0.5f);

    std::cout << "Loss (No Dropout): " << loss_nodrop << std::endl;
    std::cout << "Loss (Dropout 0.5): " << loss_drop << std::endl;

    if (loss_drop > loss_nodrop) {
        std::cout << "SUCCESS: Dropout increased training difficulty (Regularization Effect)." << std::endl;
    } else {
        std::cout << "FAILURE: Dropout did not increase loss (Check implementation)." << std::endl;
    }

    return 0;
}
