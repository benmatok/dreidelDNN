#include "../include/dreidel/models/ZenithGhostAE.hpp"
#include "../include/dreidel/models/ZenithOverhaulAE.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

using namespace dreidel;
using namespace dreidel::models;

// Helper to check if two tensors are close
template <typename T>
bool is_close(const Tensor<T>& a, const Tensor<T>& b, T tol = 1e-4) {
    if (a.shape() != b.shape()) return false;
    const T* ap = a.data();
    const T* bp = b.data();
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(ap[i] - bp[i]) > tol) return false;
    }
    return true;
}

int main() {
    std::cout << "Testing ZenithGhostAE..." << std::endl;

    // 1. Instantiation
    // Use float
    ZenithGhostAE<float> ghost_ae;
    ZenithOverhaulAE<float> base_ae;

    std::cout << "Models instantiated." << std::endl;

    // 2. Binary Compatibility Test
    std::cout << "Checking Binary Compatibility..." << std::endl;

    auto ghost_params = ghost_ae.parameters();
    auto base_params = base_ae.parameters();

    std::cout << "Ghost Params: " << ghost_params.size() << std::endl;
    std::cout << "Base Params: " << base_params.size() << std::endl;

    if (ghost_params.size() < base_params.size()) {
        std::cerr << "Error: GhostAE has fewer params than BaseAE!" << std::endl;
        return 1;
    }

    // Copy weights from Base to Ghost
    for (size_t i = 0; i < base_params.size(); ++i) {
        Tensor<float>* dst = ghost_params[i];
        const Tensor<float>* src = base_params[i];

        if (dst->shape() != src->shape()) {
            std::cerr << "Shape mismatch at param " << i << std::endl;
            // Print shapes
            auto s1 = dst->shape();
            auto s2 = src->shape();
            std::cerr << "Ghost: ["; for(auto s:s1) std::cerr<<s<<","; std::cerr<<"] vs Base: ["; for(auto s:s2) std::cerr<<s<<","; std::cerr<<"]" << std::endl;
            return 1;
        }

        // Copy data
        std::copy(src->data(), src->data() + src->size(), dst->data());
    }

    // Run Forward on both
    // Input must match expected dimensions.
    // Stem expects 3 channels.
    // Resolution should be power of 2 for ZenithBlock usually? ZenithBlock works on any if padded or generic, but strict power of 2 channels.
    // Input 64x64 is too small for 7 downsamples (requires 128x128).
    // 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 (Enc7 output)
    Tensor<float> input({1, 128, 128, 3});
    input.random(0.0f, 1.0f);

    // Disable dropout for deterministic output
    base_ae.set_training(false);
    ghost_ae.set_training(false);

    std::cout << "Running Base Forward..." << std::endl;
    Tensor<float> out_base = base_ae.forward(input);

    std::cout << "Running Ghost Forward..." << std::endl;
    Tensor<float> out_ghost = ghost_ae.forward(input);

    if (is_close(out_base, out_ghost)) {
        std::cout << "Binary Compatibility Verified: Outputs match." << std::endl;
    } else {
        std::cerr << "Binary Compatibility Failed: Outputs differ." << std::endl;
        return 1;
    }

    // 3. Test Forward Train
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
