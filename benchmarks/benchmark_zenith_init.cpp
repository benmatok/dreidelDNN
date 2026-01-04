#include "../include/dreidel/models/ComparativeAE.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/utils/WaveletGen2D.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <string>
#include <map>

using namespace dreidel;
using namespace dreidel::models;
using namespace dreidel::utils;
using namespace dreidel::optim;

// MSE Loss
template <typename T>
T mse_loss(const Tensor<T>& pred, const Tensor<T>& target, Tensor<T>& grad_input) {
    size_t size = pred.size();
    const T* p_ptr = pred.data();
    const T* t_ptr = target.data();
    T* g_ptr = grad_input.data();

    T sum_sq = 0;
    T scale = 2.0 / size;

    #pragma omp parallel for reduction(+:sum_sq)
    for (size_t i = 0; i < size; ++i) {
        T diff = p_ptr[i] - t_ptr[i];
        sum_sq += diff * diff;
        g_ptr[i] = diff * scale;
    }
    return sum_sq / size;
}

int main() {
    std::cout << "=== Zenith Meta-Train: Initialization Benchmark ===\n" << std::flush;

    // Disable Fused Kernels to isolate crash
    std::cout << "Disabling Fused Kernels for stability...\n" << std::flush;
    ZenithBlock<float>::use_fused_kernels = false;

    // Config
    const size_t H = 32;
    const size_t W = 32;
    const size_t C = 16;
    const size_t BatchSize = 2; // Keep small for speed
    const size_t Epochs = 5;
    const size_t StepsPerEpoch = 5;

    std::vector<std::string> schemes = {"he", "identity", "scaled_he_0.1", "scaled_he_0.5"};

    // Generator
    std::cout << "Init Generator...\n" << std::flush;
    WaveletGenerator2D<float> gen(H, W);
    std::cout << "Allocating Tensors...\n" << std::flush;
    Tensor<float> batch_input({BatchSize, H, W, 3});
    Tensor<float> batch_grad({BatchSize, H, W, 3});

    std::map<std::string, std::vector<float>> results;

    for (const auto& scheme : schemes) {
        std::cout << "\n--- Testing Initialization: " << scheme << " ---\n" << std::flush;

        // Instantiate fresh model
        std::cout << "Creating model...\n" << std::flush;
        ZenithHierarchicalAE<float> model(C);

        std::cout << "Reinitializing model...\n" << std::flush;
        model.reinit(scheme);

        std::cout << "Creating optimizer...\n" << std::flush;
        SimpleAdam<float> optimizer(1e-3);
        optimizer.add_parameters(model.parameters(), model.gradients());

        std::vector<float> loss_history;
        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << "Starting Training...\n" << std::flush;
        for (size_t epoch = 0; epoch < Epochs; ++epoch) {
            float epoch_loss = 0;
            for (size_t step = 0; step < StepsPerEpoch; ++step) {
                // std::cout << "Step " << step << " Gen batch...\n" << std::flush;
                gen.generate_batch(batch_input, BatchSize);

                // std::cout << "Step " << step << " Zero grad...\n" << std::flush;
                optimizer.zero_grad();
                // std::cout << "Step " << step << " Forward...\n" << std::flush;
                Tensor<float> out = model.forward(batch_input);
                // std::cout << "Step " << step << " Loss...\n" << std::flush;
                float loss = mse_loss(out, batch_input, batch_grad);
                // std::cout << "Step " << step << " Backward...\n" << std::flush;
                model.backward(batch_grad);
                // std::cout << "Step " << step << " Step...\n" << std::flush;
                optimizer.step();

                epoch_loss += loss;
            }
            float avg_loss = epoch_loss / StepsPerEpoch;
            loss_history.push_back(avg_loss);
            std::cout << "Epoch " << epoch+1 << " Loss: " << avg_loss << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Time: " << duration << "s" << std::endl;

        results[scheme] = loss_history;
    }

    std::cout << "\n=== Summary ===\n";
    std::string best_scheme = "";
    float best_final_loss = 1e9;

    for (const auto& [scheme, losses] : results) {
        float final_loss = losses.back();
        std::cout << scheme << ": Final Loss = " << final_loss << std::endl;
        if (final_loss < best_final_loss) {
            best_final_loss = final_loss;
            best_scheme = scheme;
        }
    }

    std::cout << "\nFastest to Converge (Lowest Final Loss): " << best_scheme << std::endl;

    return 0;
}
