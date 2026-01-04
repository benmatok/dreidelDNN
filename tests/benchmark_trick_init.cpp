#include "../include/dreidel/layers/ZenithBlock.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

using namespace dreidel;
using namespace dreidel::layers;

int main() {
    std::cout << "Benchmark 1: Delta-Orthogonal vs Random Initialization (Depth 20)" << std::endl;

    size_t depth = 20;
    size_t C = 64; // Moderate channel size
    size_t H = 16, W = 16;

    // Create Input (Unit Variance)
    Tensor<float> input({1, H, W, C});
    input.random(0.0f, 1.0f); // Mean 0.5, Var 1/12 approx.

    // Normalize input to 0 mean, 1 var
    float mean = 0, var = 0;
    float* d = input.data();
    for(size_t i=0; i<input.size(); ++i) mean += d[i];
    mean /= input.size();
    for(size_t i=0; i<input.size(); ++i) var += (d[i]-mean)*(d[i]-mean);
    var /= input.size();
    float std = std::sqrt(var);
    for(size_t i=0; i<input.size(); ++i) d[i] = (d[i]-mean)/std;

    auto measure_variance = [&](const std::string& scheme) {
        Tensor<float> x = input; // Copy
        std::vector<std::unique_ptr<ZenithBlock<float>>> layers;

        for(size_t i=0; i<depth; ++i) {
            auto layer = std::make_unique<ZenithBlock<float>>(C, 3, C);
            layer->reinit(scheme);
            // Disable GroupNorm to test raw initialization stability
            // If GroupNorm is on, variance is forced to 1, hiding issues.
            layer->set_group_norm(false);
            // Disable ReLU? No, ReLU cuts variance in half. Init should compensate or just observe decay.
            // Delta-Orthogonal + ReLU: Variance halves each layer?
            // "The Gem: Scale it so the layer acts as a 'Pass-through' at t=0".
            // If we use standard Identity, it preserves variance. ReLU halves it.
            // If we want to verify "Pass-through", maybe we should disable ReLU or expect decay.
            // But the claim is "Allows training 10,000 layer networks".
            // Let's see what happens.
            layers.push_back(std::move(layer));
        }

        for(auto& layer : layers) {
            x = layer->forward(x);
        }

        // Compute output variance
        float m = 0, v = 0;
        const float* out_d = x.data();
        for(size_t i=0; i<x.size(); ++i) m += out_d[i];
        m /= x.size();
        for(size_t i=0; i<x.size(); ++i) v += (out_d[i]-m)*(out_d[i]-m);
        v /= x.size();
        return v;
    };

    float var_identity = measure_variance("identity");
    float var_random = measure_variance("he");

    std::cout << "Variance (Identity Init): " << var_identity << std::endl;
    std::cout << "Variance (Random He Init): " << var_random << std::endl;

    if (var_identity > var_random) {
        std::cout << "SUCCESS: Identity Init preserved more signal variance." << std::endl;
    } else {
        std::cout << "FAILURE: Random Init preserved more/equal variance." << std::endl;
    }

    return 0;
}
