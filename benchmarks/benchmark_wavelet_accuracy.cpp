#include "../include/dreidel/models/ZenithHierarchicalAE.hpp"
#include "../include/dreidel/models/ConvBaselineAE.hpp"
#include "../include/dreidel/utils/WaveletGen2D.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <fstream>
#include <iomanip>

using namespace dreidel;

// Helper to compute MSE
template <typename T>
T compute_mse(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.size() != b.size()) return -1.0;
    T sum = 0;
    const T* ap = a.data();
    const T* bp = b.data();

    // Naive sum (potential overflow for very large tensors if T is small, but float is fine for Loss)
    #pragma omp parallel for reduction(+:sum)
    for(size_t i=0; i<a.size(); ++i) {
        T diff = ap[i] - bp[i];
        sum += diff * diff;
    }
    return sum / a.size();
}

template <typename T>
void train_model(layers::Layer<T>* model, utils::WaveletGenerator2D<T>& gen, const std::string& name,
                 int steps, size_t batch_size, size_t H, size_t W, size_t C, std::vector<double>& loss_history) {

    std::cout << "Training " << name << "..." << std::endl;

    // Optimizer
    // Zenith often needs lower LR. Baseline needs standard LR.
    // Let's pick a conservative one for both, or specific ones.
    // Memory says Zenith uses lower LR (e.g. 0.0001).
    // Baseline Conv usually works with 0.001.
    // To be fair, we should maybe tune or pick a middle ground 0.0005?
    // Or just use 1e-3 and see if Zenith explodes (it has GroupNorm so should be ok-ish).
    // Actually memory says Zenith C=32 converged with 0.0002.
    // Let's use 0.0005.

    optim::SimpleAdam<T> optimizer(0.0005);
    optimizer.add_parameters(model->parameters(), model->gradients());

    // Pre-allocate batch
    Tensor<T> input({batch_size, H, W, C});

    for(int step=0; step<steps; ++step) {
        // Generate Batch
        gen.generate_batch(input, batch_size);

        // Zero Grad
        optimizer.zero_grad();

        // Forward
        Tensor<T> output = model->forward(input);

        // Loss (MSE)
        T mse = compute_mse(input, output);
        loss_history.push_back(static_cast<double>(mse));

        // Backward
        Tensor<T> grad_output(output.shape());
        const T* out_ptr = output.data();
        const T* in_ptr = input.data();
        T* g_ptr = grad_output.data();

        // d(MSE)/dx = 2(y - t) / N
        T scale = 2.0f / output.size();

        #pragma omp parallel for
        for(size_t i=0; i<output.size(); ++i) {
            g_ptr[i] = (out_ptr[i] - in_ptr[i]) * scale;
        }

        model->backward(grad_output);

        // Update
        optimizer.step();

        if (step % 10 == 0) {
            std::cout << "Step " << step << " | Loss: " << mse << std::endl;
        }
    }
}

int main() {
    size_t H = 128;
    size_t W = 128;
    size_t C = 3;
    size_t batch_size = 2; // Small batch to run fast on CPU
    int steps = 20; // Enough to show early convergence curve

    utils::WaveletGenerator2D<float> gen(H, W);

    // Zenith
    models::ZenithHierarchicalAE<float> zenith_ae;
    std::vector<double> zenith_loss;
    train_model(&zenith_ae, gen, "ZenithHierarchicalAE", steps, batch_size, H, W, C, zenith_loss);

    // Baseline
    models::ConvBaselineAE<float> conv_ae;
    std::vector<double> conv_loss;
    train_model(&conv_ae, gen, "ConvBaselineAE", steps, batch_size, H, W, C, conv_loss);

    // Save Results
    std::ofstream csv("benchmark_results_accuracy.csv");
    csv << "Step,ZenithLoss,ConvLoss\n";
    for(size_t i=0; i<zenith_loss.size(); ++i) {
        csv << i << "," << zenith_loss[i] << "," << conv_loss[i] << "\n";
    }
    csv.close();

    std::cout << "Results saved to benchmark_results_accuracy.csv" << std::endl;

    return 0;
}
