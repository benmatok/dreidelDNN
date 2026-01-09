#include "../include/dreidel/models/ZenithHierarchicalAE.hpp"
#include "../include/dreidel/utils/WaveletGen2D.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <cmath>

using namespace dreidel;

// Helper to compute MSE
template <typename T>
T compute_mse(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.size() != b.size()) return -1.0;
    T sum = 0;
    const T* ap = a.data();
    const T* bp = b.data();

    #pragma omp parallel for reduction(+:sum)
    for(size_t i=0; i<a.size(); ++i) {
        T diff = ap[i] - bp[i];
        sum += diff * diff;
    }
    return sum / a.size();
}

template <typename T>
double train_run(const std::vector<Tensor<T>>& dataset, int steps, bool use_slm, size_t base_channels) {
    // Instantiate a fresh model
    // base_channels=8 for speed (vs 32 default)
    // 8 -> 128 -> 2048 channels.
    models::ZenithHierarchicalAE<T> model(3, base_channels, "he", use_slm);

    optim::SimpleAdam<T> optimizer(0.0001); // Stable LR
    optimizer.add_parameters(model.parameters(), model.gradients());

    size_t dataset_size = dataset.size();
    double final_loss = 0;

    for(int step=0; step<steps; ++step) {
        // Annealing Schedule
        // Epoch 0-5 (assume epoch ~ dataset_size steps? or just steps)
        // Spec says: Epoch 0-5 Warmup (Temp 1.0), Epoch 6-30 Squeeze (Decay to 0), Epoch 31+ Lock (Temp 0).
        // Let's assume 1 epoch = dataset_size (200) steps? Or just map steps to schedule.
        // Let's assume total steps=800 corresponds to "full training".
        // 800 steps / 50 epochs = 16 steps/epoch?
        // Let's map steps directly:
        // 0-100: Temp 1.0
        // 100-600: Decay
        // 600+: Temp 0.0
        float temp = 1.0f;
        if (step > 600) temp = 0.0f;
        else if (step > 100) {
            float progress = (float)(step - 100) / 500.0f;
            temp = 1.0f - progress;
        }

        if (use_slm) {
            model.set_gate_training(true, temp);
        } else {
            // Ensure baseline uses hard pass-through (inference mode)
            model.set_gate_training(false);
        }

        const Tensor<T>& input = dataset[step % dataset_size];

        optimizer.zero_grad();
        Tensor<T> output = model.forward(input);

        T mse = compute_mse(input, output);

        // Sparsity Loss (Lambda = 1e-4)
        T sparsity_loss = 0;
        if (use_slm) {
            sparsity_loss = model.get_sparsity_loss() * 1e-4f;
        }

        final_loss = static_cast<double>(mse + sparsity_loss);

        // Backward
        Tensor<T> grad_output(output.shape());
        const T* out_ptr = output.data();
        const T* in_ptr = input.data();
        T* g_ptr = grad_output.data();
        T scale = 2.0f / output.size();

        #pragma omp parallel for
        for(size_t i=0; i<output.size(); ++i) {
            g_ptr[i] = (out_ptr[i] - in_ptr[i]) * scale;
        }

        model.backward(grad_output);
        optimizer.step();
    }
    return final_loss;
}

int main() {
    size_t H = 64;
    size_t W = 64;
    size_t C = 3;
    size_t batch_size = 4;

    int steps = 800;
    int num_runs = 5;
    size_t base_channels = 8; // Reduced for speed

    std::cout << "Pre-generating dataset (200 batches)..." << std::endl;
    utils::WaveletGenerator2D<float> gen(H, W);
    std::vector<Tensor<float>> dataset;
    dataset.reserve(200);
    for(int i=0; i<200; ++i) {
        Tensor<float> t({batch_size, H, W, C});
        gen.generate_batch(t, batch_size);
        dataset.push_back(std::move(t));
    }

    std::cout << "\nStarting Statistical Benchmark (" << num_runs << " runs, " << steps << " steps, base_ch=" << base_channels << ")" << std::endl;

    // 1. Standard Zenith
    std::cout << "\n--- Standard Zenith (No SLM) ---" << std::endl;
    std::vector<double> std_losses;
    for(int i=0; i<num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        double loss = train_run(dataset, steps, false, base_channels);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std_losses.push_back(loss);
        std::cout << "Run " << i+1 << ": Loss = " << loss << " (" << elapsed.count() << "s)" << std::endl;
    }

    // 2. Zenith-SRIG
    std::cout << "\n--- Zenith-SRIG (With SLM) ---" << std::endl;
    std::vector<double> srig_losses;
    for(int i=0; i<num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        double loss = train_run(dataset, steps, true, base_channels);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        srig_losses.push_back(loss);
        std::cout << "Run " << i+1 << ": Loss = " << loss << " (" << elapsed.count() << "s)" << std::endl;
    }

    // Statistics
    auto compute_stats = [](const std::vector<double>& v) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / v.size() - mean * mean);
        return std::make_pair(mean, stdev);
    };

    auto [std_mean, std_dev] = compute_stats(std_losses);
    auto [srig_mean, srig_dev] = compute_stats(srig_losses);

    std::cout << "\n=== Results Summary ===" << std::endl;
    std::cout << "Standard Zenith: Mean Loss = " << std_mean << " (std: " << std_dev << ")" << std::endl;
    std::cout << "Zenith-SRIG    : Mean Loss = " << srig_mean << " (std: " << srig_dev << ")" << std::endl;

    double improvement = (std_mean - srig_mean) / std_mean * 100.0;
    std::cout << "Improvement: " << improvement << "%" << std::endl;

    if (srig_mean < std_mean && (std_mean - srig_mean) > srig_dev) {
        std::cout << "Result: CONSISTENT IMPROVEMENT" << std::endl;
    } else {
        std::cout << "Result: INCONCLUSIVE (High Variance)" << std::endl;
    }

    return 0;
}
