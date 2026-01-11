#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/models/ZenithGhostAE.hpp"
#include "../include/dreidel/models/ZenithOverhaulAE.hpp"

using namespace dreidel;

int main(int argc, char** argv) {
    size_t dim = 16384; // 128x128
    size_t batch_size = 4;
    size_t size = 128;
    size_t iterations = 5;

    std::cout << "Benchmarking ZenithOverhaulAE vs ZenithGhostAE" << std::endl;
    std::cout << "Input: " << size << "x" << size << " (" << dim << "), Batch: " << batch_size << std::endl;

    models::ZenithOverhaulAE<float> overhaul;
    models::ZenithGhostAE<float> ghost;

    // Dummy Data
    Tensor<float> input({batch_size, size, size, 3});
    input.fill(0.5f);
    Tensor<float> grad_output({batch_size, size, size, 3});
    grad_output.fill(0.1f);

    // Warmup
    std::cout << "Warmup..." << std::endl;
    overhaul.set_training(false);
    ghost.set_training(false);
    overhaul.forward(input);
    ghost.forward(input);

    // 1. Inference Speed
    std::cout << "\n--- Inference Speed (Forward Only) ---" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<iterations; ++i) {
        volatile auto out = overhaul.forward(input);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double dt_overhaul = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "ZenithOverhaulAE Inference: " << (iterations * batch_size) / dt_overhaul << " samples/sec" << std::endl;

    t1 = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<iterations; ++i) {
        volatile auto out = ghost.forward(input);
    }
    t2 = std::chrono::high_resolution_clock::now();
    double dt_ghost = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "ZenithGhostAE Inference:    " << (iterations * batch_size) / dt_ghost << " samples/sec" << std::endl;

    // 2. Standard Training Speed (Forward + Backward Recon)
    std::cout << "\n--- Standard Training Speed (Recon Loss Only) ---" << std::endl;
    overhaul.set_training(true);
    ghost.set_training(true);

    t1 = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<iterations; ++i) {
        auto out = overhaul.forward(input);
        auto grad = overhaul.backward(grad_output);
    }
    t2 = std::chrono::high_resolution_clock::now();
    dt_overhaul = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "ZenithOverhaulAE Training: " << (iterations * batch_size) / dt_overhaul << " samples/sec" << std::endl;

    // Ghost using standard backward (should be similar/identical)
    t1 = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<iterations; ++i) {
        auto out = ghost.forward(input); // Standard forward
        auto grad = ghost.backward(grad_output); // Standard backward
    }
    t2 = std::chrono::high_resolution_clock::now();
    dt_ghost = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "ZenithGhostAE (Std Train): " << (iterations * batch_size) / dt_ghost << " samples/sec" << std::endl;

    // 3. Ghost Training Speed (Forward Train + Backward Train)
    std::cout << "\n--- Ghost Training Speed (Recon + Ghost Consistency) ---" << std::endl;

    // Prepare dummy ghost grads
    // We need one forward pass to get shapes
    auto train_out = ghost.forward_train(input);
    std::vector<Tensor<float>> ghost_grads;
    for(auto& p : train_out.ghost_preds) {
        Tensor<float> g = p;
        g.fill(0.01f);
        ghost_grads.push_back(g);
    }

    t1 = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<iterations; ++i) {
        auto out = ghost.forward_train(input);
        ghost.backward_train(grad_output, ghost_grads);
    }
    t2 = std::chrono::high_resolution_clock::now();
    double dt_ghost_train = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "ZenithGhostAE (Ghost Train): " << (iterations * batch_size) / dt_ghost_train << " samples/sec" << std::endl;

    double slowdown = dt_ghost_train / dt_overhaul;
    std::cout << "\nRelative Slowdown (Ghost Train vs Overhaul Train): " << slowdown << "x slower" << std::endl;

    return 0;
}
