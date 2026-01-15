#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/models/ZenithNano.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"

using namespace dreidel;

void generate_batch(Tensor<float>& data) {
    auto shape = data.shape();
    size_t batch = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    size_t C = shape[3]; // 3 for ZenithNano

    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> center_dist(0.3f, 0.7f);
    std::uniform_real_distribution<float> scale_dist(0.05f, 0.2f);

    // Fill with multiple wavelets
    data.fill(0.0f);
    float* ptr = data.data();

    // 4 wavelets per image
    for(size_t n=0; n<batch; ++n) {
        for(int k=0; k<4; ++k) {
            float cx = center_dist(gen) * W;
            float cy = center_dist(gen) * H;
            float s = scale_dist(gen) * W;

            // Random color
            float r = (float)gen() / gen.max();
            float g = (float)gen() / gen.max();
            float b = (float)gen() / gen.max();

            // Add gaussian blob
            #pragma omp parallel for collapse(2)
            for(size_t y=0; y<H; ++y) {
                for(size_t x=0; x<W; ++x) {
                    float dx = x - cx;
                    float dy = y - cy;
                    float val = std::exp(-(dx*dx + dy*dy) / (2*s*s));
                    if (val > 0.01f) {
                        size_t idx = ((n*H + y)*W + x)*C;
                        ptr[idx + 0] += val * r;
                        ptr[idx + 1] += val * g;
                        ptr[idx + 2] += val * b;
                    }
                }
            }
        }
    }
}

int main() {
    std::cout << "=== Training ZenithNano on Wavelets ===" << std::endl;

    // Config
    size_t batch_size = 4; // Use small batch for CPU training
    size_t H = 512, W = 512;
    float lr = 1e-4f;
    double max_time_sec = 150.0;

    // Model
    models::ZenithNano model;

    // Optimizer
    optim::SimpleAdam<float> optimizer(lr);
    optimizer.add_parameters(model.parameters(), model.gradients());

    // Tensors
    Tensor<float> input({batch_size, H, W, 3});
    // Target is input (Autoencoder)

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t epoch = 0;

    std::cout << "Starting training for approx " << max_time_sec << " seconds..." << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;

    while(true) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed > max_time_sec) break;

        // 1. Generate Data
        generate_batch(input);

        // 2. Forward
        // ZenithNano forward currently returns a new tensor.
        // We need to manage memory properly.
        // Layer::forward usually allocates.
        Tensor<float> output = model.forward(input);

        // 3. Loss (MAE)
        float loss = 0;
        size_t size = output.size();
        const float* out_ptr = output.data();
        const float* tgt_ptr = input.data();

        // MAE Loss Gradient: sign(output - target) / size
        Tensor<float> grad_output(output.shape());
        float* go_ptr = grad_output.data();

        // Reduction for loss logging
        #pragma omp parallel for reduction(+:loss)
        for(size_t i=0; i<size; ++i) {
            float diff = out_ptr[i] - tgt_ptr[i];
            loss += std::abs(diff);

            // Gradient
            float sign = (diff > 0) ? 1.0f : ((diff < 0) ? -1.0f : 0.0f);
            go_ptr[i] = sign / size; // Mean reduction
        }
        loss /= size;

        // 4. Backward
        optimizer.zero_grad();
        model.backward(grad_output);

        // 5. Update
        optimizer.step();

        epoch++;

        // Log
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " | Time: " << std::fixed << std::setprecision(2) << elapsed << "s | MAE Loss: " << std::setprecision(6) << loss << std::endl;
        }
    }

    std::cout << "Training Complete." << std::endl;
    std::cout << "Total Epochs: " << epoch << std::endl;

    return 0;
}
