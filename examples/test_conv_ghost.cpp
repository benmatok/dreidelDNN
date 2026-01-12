#include "../include/dreidel/models/ConvGhostAE.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

using namespace dreidel;
using namespace dreidel::models;

int main() {
    std::cout << "Testing ConvGhostAE..." << std::endl;

    // 1. Instantiation
    ConvGhostAE<float> ghost_ae;

    std::cout << "Model instantiated." << std::endl;

    // 2. Forward Inference
    Tensor<float> input({1, 128, 128, 3});
    input.random(0.0f, 1.0f);

    ghost_ae.set_training(false);

    std::cout << "Running Forward..." << std::endl;
    Tensor<float> out = ghost_ae.forward(input);
    std::cout << "Output shape: " << out.shape()[0] << " " << out.shape()[1] << " " << out.shape()[2] << " " << out.shape()[3] << std::endl;

    assert(out.shape()[0] == 1);
    assert(out.shape()[1] == 128);
    assert(out.shape()[2] == 128);
    assert(out.shape()[3] == 3);

    // 3. Test Forward Train
    ghost_ae.set_training(true);
    std::cout << "Testing Forward Train..." << std::endl;
    auto train_out = ghost_ae.forward_train(input);

    std::cout << "Reconstruction Shape: " << train_out.reconstruction.shape()[1] << "x" << train_out.reconstruction.shape()[2] << std::endl;
    std::cout << "Ghost Predictions: " << train_out.ghost_preds.size() << std::endl;
    std::cout << "Encoder Targets: " << train_out.encoder_targets.size() << std::endl;

    assert(train_out.ghost_preds.size() == 7);
    assert(train_out.encoder_targets.size() == 7);

    // Check shapes match between pred and target
    for (size_t i = 0; i < 7; ++i) {
        if (train_out.ghost_preds[i].shape() != train_out.encoder_targets[i].shape()) {
            std::cerr << "Ghost/Target Shape Mismatch at index " << i << std::endl;
            auto s1 = train_out.ghost_preds[i].shape();
            auto s2 = train_out.encoder_targets[i].shape();
            std::cerr << "Pred: "; for(auto s:s1)std::cerr<<s<<" "; std::cerr << " Target: "; for(auto s:s2)std::cerr<<s<<" "; std::cerr<<std::endl;
            return 1;
        }
    }

    // 4. Test Backward Train
    std::cout << "Testing Backward Train..." << std::endl;
    Tensor<float> grad_recon = train_out.reconstruction;
    grad_recon.fill(0.1f);

    std::vector<Tensor<float>> grad_ghosts;
    for (auto& p : train_out.ghost_preds) {
        Tensor<float> g = p;
        g.fill(0.01f);
        grad_ghosts.push_back(g);
    }

    ghost_ae.backward_train(grad_recon, grad_ghosts);
    std::cout << "Backward Train Completed." << std::endl;

    return 0;
}
