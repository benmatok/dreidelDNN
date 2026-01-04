#include "../include/dreidel/models/ComparativeAE.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace dreidel;
using namespace dreidel::models;
using namespace dreidel::optim;

int main() {
    std::cout << "Benchmark: 2D Sinusoidal PE in Bottleneck" << std::endl;

    // Use a task that requires spatial awareness: Reconstruct a gradient ramp.
    // However, the bottleneck is small (7x7).
    // The PE helps the decoder know "where" the bottleneck features are from?
    // Actually, in CNNs, spatial info is preserved. But PE makes it explicit/richer.
    // Let's run a short training loop to check stability and loss.

    size_t H = 224, W = 224; // Standard Input size for 7x7 bottleneck
    size_t C = 3;

    // Gradient Input
    Tensor<float> input({1, H, W, C});
    for(size_t h=0; h<H; ++h) {
        for(size_t w=0; w<W; ++w) {
            float v = (float)w/W + (float)h/H;
            for(size_t c=0; c<C; ++c) input.data()[((0*H+h)*W+w)*C + c] = v;
        }
    }

    // Normalize
    float mean = 0, var = 0;
    for(size_t i=0; i<input.size(); ++i) mean += input.data()[i];
    mean /= input.size();
    for(size_t i=0; i<input.size(); ++i) var += (input.data()[i]-mean)*(input.data()[i]-mean);
    var /= input.size();
    float std = std::sqrt(var);
    for(size_t i=0; i<input.size(); ++i) input.data()[i] = (input.data()[i]-mean)/std;

    ZenithHierarchicalAE<float> model(16); // 16 base filters -> 4096 bottleneck
    SimpleAdam<float> opt(0.001);
    opt.add_parameters(model.parameters(), model.gradients());

    float initial_loss = 0;
    float final_loss = 0;

    std::cout << "Training for 5 steps..." << std::endl;
    for(int i=0; i<5; ++i) {
        opt.zero_grad();
        Tensor<float> out = model.forward(input);

        float loss = 0;
        Tensor<float> grad = out;
        // MSE vs Input (Autoencoder)
        for(size_t k=0; k<out.size(); ++k) {
            float diff = out.data()[k] - input.data()[k];
            loss += diff*diff;
            grad.data()[k] = 2*diff/out.size();
        }

        if (i==0) initial_loss = loss;
        final_loss = loss;

        std::cout << "Step " << i << " Loss: " << loss << std::endl;

        model.backward(grad);
        opt.step();
    }

    if (final_loss < initial_loss) {
        std::cout << "SUCCESS: Loss decreased with PE Injection." << std::endl;
    } else {
        std::cout << "FAILURE: Loss did not decrease." << std::endl;
    }

    return 0;
}
