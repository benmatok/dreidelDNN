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
void train_model(layers::Layer<T>* model, const std::vector<Tensor<T>>& dataset, const std::string& name,
                 int steps, size_t batch_size, std::vector<double>& loss_history, bool use_clipping = false) {

    std::cout << "Training " << name << "..." << std::endl;

    optim::SimpleAdam<T> optimizer(0.0005);
    if (use_clipping) {
        optimizer.set_coordinate_wise_clipping(true, 1.0);
    }
    optimizer.add_parameters(model->parameters(), model->gradients());

    size_t dataset_size = dataset.size();

    for(int step=0; step<steps; ++step) {
        // Sample random image from dataset (Simple cycling for benchmark)
        const Tensor<T>& input = dataset[step % dataset_size];

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
    size_t H = 64;
    size_t W = 64;
    size_t C = 3;
    size_t batch_size = 4;

    // Reduced to 1500 to fit sandbox limits
    int steps = 1500;
    int dataset_size = 200; // 200 distinct batches

    std::cout << "Pre-generating dataset (" << dataset_size << " batches)..." << std::endl;
    utils::WaveletGenerator2D<float> gen(H, W);
    std::vector<Tensor<float>> dataset;
    dataset.reserve(dataset_size);
    for(int i=0; i<dataset_size; ++i) {
        Tensor<float> t({batch_size, H, W, C});
        gen.generate_batch(t, batch_size);
        dataset.push_back(std::move(t));
    }

    // 1. Standard Zenith (He, PE=True, SLM=False)
    std::cout << "=== Experiment 1: Standard Zenith (He, PE) ===" << std::endl;
    models::ZenithHierarchicalAE<float> model_std(C, 32, true, "he", false);
    std::vector<double> loss_std;
    train_model(&model_std, dataset, "Standard Zenith", steps, batch_size, loss_std, false);

    // 2. Zenith-SLM (He, PE=True, SLM=True)
    std::cout << "\n=== Experiment 2: Zenith-SLM (He, PE, SLM) ===" << std::endl;
    models::ZenithHierarchicalAE<float> model_slm(C, 32, true, "he", true);
    std::vector<double> loss_slm;
    train_model(&model_slm, dataset, "Zenith-SLM", steps, batch_size, loss_slm, false);

    // Save Results
    std::ofstream csv("benchmark_results_slm.csv");
    csv << "Step,Standard,SLM\n";
    for(size_t i=0; i<loss_std.size(); ++i) {
        if (i % 10 == 0) {
             csv << i << "," << loss_std[i] << "," << loss_slm[i] << "\n";
        }
    }
    csv.close();

    std::cout << "Results saved to benchmark_results_ablation.csv" << std::endl;

    return 0;
}
