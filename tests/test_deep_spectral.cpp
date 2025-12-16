
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/DeepSpectralLinear.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace dreidel;

void test_forward_backward() {
    std::cout << "Testing DeepSpectralLinear Forward/Backward..." << std::endl;

    size_t batch = 2;
    size_t dim = 16;
    size_t depth = 3;

    layers::DeepSpectralLinear<float> layer(dim, depth);

    Tensor<float> input({batch, dim});
    input.random(0.0f, 1.0f);

    // Forward
    Tensor<float> output = layer.forward(input);

    assert(output.shape().size() == 2);
    assert(output.shape()[0] == batch);
    assert(output.shape()[1] == dim);

    std::cout << "Forward Pass Complete." << std::endl;

    // Backward
    Tensor<float> grad_output({batch, dim});
    grad_output.random(0.0f, 0.1f);

    Tensor<float> grad_input = layer.backward(grad_output);

    assert(grad_input.shape() == input.shape());

    // Check gradients exist
    auto grads = layer.gradients();
    assert(grads.size() == depth);
    for (auto* g : grads) {
        assert(g->size() == dim);
        // Ensure not all zero (random inputs so likely non-zero)
        float sum = 0;
        for (size_t i = 0; i < dim; ++i) sum += std::abs((*g)[i]);
        assert(sum > 0);
    }

    std::cout << "Backward Pass Complete." << std::endl;
}

void test_overfit_random_matrix() {
    std::cout << "Testing Overfitting Random Matrix..." << std::endl;
    // Goal: Learn y = x * W where W is random dense.
    // Use SGD to update scales. P is fixed.

    size_t N = 64;
    size_t depth = 4;
    size_t batch = 32;

    layers::DeepSpectralLinear<float> layer(N, depth);

    // Create random target W
    // We simulate W by just using random inputs and random targets for now,
    // but better to have a consistent function.
    // Let's generate a fixed W
    Tensor<float> W({N, N});
    W.random(0, 1.0f/std::sqrt(N));

    // Training loop
    // FWHT is unnormalized, so gradients scale by N per layer.
    // With depth 4, we need a very small LR or normalization.
    float lr = 0.00001f;
    for (int epoch = 0; epoch < 500; ++epoch) {
        Tensor<float> x({batch, N});
        x.random(0, 1.0f);

        Tensor<float> y_target = x.matmul(W);

        // Forward
        Tensor<float> y_pred = layer.forward(x);

        // Loss = MSE
        Tensor<float> diff = y_pred - y_target;
        float mse = 0;
        for(size_t i=0; i<diff.size(); ++i) mse += diff[i]*diff[i];
        mse /= diff.size();

        if (epoch % 50 == 0) std::cout << "Epoch " << epoch << " MSE: " << mse << std::endl;

        // Backward
        // dL/dy = 2 * (y_pred - y_target) / Batch
        Tensor<float> grad_out = diff * (2.0f / batch);
        layer.backward(grad_out);

        // Update weights (SGD)
        auto params = layer.parameters();
        auto grads = layer.gradients();

        for (size_t k = 0; k < params.size(); ++k) {
            float* p = params[k]->data();
            float* g = grads[k]->data();
            for (size_t i = 0; i < N; ++i) {
                p[i] -= lr * g[i];
            }
        }
    }
    std::cout << "Training Complete." << std::endl;
}

int main() {
    test_forward_backward();
    test_overfit_random_matrix();
    return 0;
}
