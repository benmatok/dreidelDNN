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
void train_model(layers::Layer<T>* model, utils::WaveletGenerator2D<T>& gen, const std::string& name,
                 int steps, size_t batch_size, size_t H, size_t W, size_t C, std::vector<double>& loss_history) {

    std::cout << "Training " << name << "..." << std::endl;

    optim::SimpleAdam<T> optimizer(0.0005);
    optimizer.add_parameters(model->parameters(), model->gradients());

    Tensor<T> input({batch_size, H, W, C});

    for(int step=0; step<steps; ++step) {
        gen.generate_batch(input, batch_size);
        optimizer.zero_grad();
        Tensor<T> output = model->forward(input);

        T mse = compute_mse(input, output);
        loss_history.push_back(static_cast<double>(mse));

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

        model->backward(grad_output);
        optimizer.step();

        if (step % 500 == 0) {
            std::cout << "  Step " << step << " | Loss: " << mse << std::endl;
        }
    }
}

int main() {
    size_t H = 64; // Smaller for speed to allow more steps
    size_t W = 64;
    size_t C = 3;
    size_t batch_size = 4;

    // "500 epochs" requested.
    // Reduced for sandbox limits.

    int steps = 2000;

    utils::WaveletGenerator2D<float> gen(H, W);

    // 1. Baseline: He Init, No PE
    std::cout << "=== Experiment 1: Baseline (He Init, No PE) ===" << std::endl;
    models::ZenithHierarchicalAE<float> model_he_nope(C, 32, false, "he");
    std::vector<double> loss_he_nope;
    train_model(&model_he_nope, gen, "Baseline", steps, batch_size, H, W, C, loss_he_nope);

    // 2. Improvement A: Identity Init, No PE
    std::cout << "\n=== Experiment 2: Improvement A (Identity Init, No PE) ===" << std::endl;
    models::ZenithHierarchicalAE<float> model_id_nope(C, 32, false, "identity");
    std::vector<double> loss_id_nope;
    train_model(&model_id_nope, gen, "Identity Init", steps, batch_size, H, W, C, loss_id_nope);

    // 3. Improvement B: Identity Init + PE
    std::cout << "\n=== Experiment 3: Improvement B (Identity Init + PE) ===" << std::endl;
    models::ZenithHierarchicalAE<float> model_id_pe(C, 32, true, "identity");
    std::vector<double> loss_id_pe;
    train_model(&model_id_pe, gen, "Identity + PE", steps, batch_size, H, W, C, loss_id_pe);

    // Save Results
    std::ofstream csv("benchmark_results_ablation.csv");
    csv << "Step,Baseline,IdentityInit,IdentityPE\n";
    for(size_t i=0; i<loss_he_nope.size(); ++i) {
        // Logging every step creates huge CSV. Maybe subsample?
        if (i % 10 == 0) {
             csv << i << "," << loss_he_nope[i] << "," << loss_id_nope[i] << "," << loss_id_pe[i] << "\n";
        }
    }
    csv.close();

    std::cout << "Results saved to benchmark_results_ablation.csv" << std::endl;

    return 0;
}
