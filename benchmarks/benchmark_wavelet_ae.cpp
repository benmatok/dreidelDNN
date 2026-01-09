#include "../include/dreidel/models/ZenithHierarchicalAE.hpp"
#include "../include/dreidel/models/ConvBaselineAE.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <fstream>
#include <iomanip>

using namespace dreidel;

template <typename T>
struct BenchmarkResult {
    std::string name;
    double forward_time_avg;
    double backward_time_avg;
    double total_time_avg;
};

template <typename T>
BenchmarkResult<T> run_benchmark(layers::Layer<T>* model, const Tensor<T>& input, const std::string& name, int iterations) {
    std::vector<double> forward_times;
    std::vector<double> backward_times;

    // Warmup
    for(int i=0; i<5; ++i) {
        Tensor<T> out = model->forward(input);
        Tensor<T> grad_out(out.shape());
        grad_out.fill(1.0);
        model->backward(grad_out);
    }

    // Timed run
    for(int i=0; i<iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor<T> out = model->forward(input);
        auto mid = std::chrono::high_resolution_clock::now();

        Tensor<T> grad_out(out.shape());
        grad_out.fill(1.0);

        model->backward(grad_out);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> fwd = mid - start;
        std::chrono::duration<double> bwd = end - mid;

        forward_times.push_back(fwd.count());
        backward_times.push_back(bwd.count());
    }

    double fwd_sum = std::accumulate(forward_times.begin(), forward_times.end(), 0.0);
    double bwd_sum = std::accumulate(backward_times.begin(), backward_times.end(), 0.0);

    return {name, fwd_sum / iterations, bwd_sum / iterations, (fwd_sum + bwd_sum) / iterations};
}

int main() {
    size_t H = 128;
    size_t W = 128;
    size_t C = 3;
    size_t N = 1;
    int iterations = 20;

    std::cout << "Benchmarking Wavelet AE on " << H << "x" << W << " inputs..." << std::endl;

    Tensor<float> input({N, H, W, C});
    input.random(0.0f, 1.0f);

    models::ZenithHierarchicalAE<float> zenith_ae;
    // Set to inference mode to avoid noise injection overhead and ensure fair comparison
    zenith_ae.set_gate_training(false);
    auto res_zenith = run_benchmark(&zenith_ae, input, "ZenithHierarchicalAE", iterations);

    models::ZenithHierarchicalAE<float> zenith_slm_ae(3, 32, "he", true);
    // Explicitly set to inference mode for benchmarking (disable noise/soft-gating)
    zenith_slm_ae.set_gate_training(false);
    auto res_zenith_slm = run_benchmark(&zenith_slm_ae, input, "Zenith-SLM", iterations);

    models::ConvBaselineAE<float> conv_ae;
    auto res_conv = run_benchmark(&conv_ae, input, "ConvBaselineAE", iterations);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Results (Avg of " << iterations << " runs):" << std::endl;
    std::cout << "Model | Forward (s) | Backward (s) | Total (s)" << std::endl;
    std::cout << "--- | --- | --- | ---" << std::endl;
    std::cout << res_zenith.name << " | " << res_zenith.forward_time_avg << " | " << res_zenith.backward_time_avg << " | " << res_zenith.total_time_avg << std::endl;
    std::cout << res_zenith_slm.name << " | " << res_zenith_slm.forward_time_avg << " | " << res_zenith_slm.backward_time_avg << " | " << res_zenith_slm.total_time_avg << std::endl;
    std::cout << res_conv.name << " | " << res_conv.forward_time_avg << " | " << res_conv.backward_time_avg << " | " << res_conv.total_time_avg << std::endl;

    double fwd_speedup = res_conv.forward_time_avg / res_zenith_slm.forward_time_avg;
    double total_speedup = res_conv.total_time_avg / res_zenith_slm.total_time_avg;

    std::cout << std::endl;
    std::cout << "Speedup (Zenith-SLM vs Baseline):" << std::endl;
    std::cout << "Forward: " << fwd_speedup << "x" << std::endl;
    std::cout << "Total: " << total_speedup << "x" << std::endl;

    std::ofstream csv("benchmark_results_wavelet.csv");
    csv << "Model,Forward,Backward,Total\n";
    csv << res_zenith.name << "," << res_zenith.forward_time_avg << "," << res_zenith.backward_time_avg << "," << res_zenith.total_time_avg << "\n";
    csv << res_zenith_slm.name << "," << res_zenith_slm.forward_time_avg << "," << res_zenith_slm.backward_time_avg << "," << res_zenith_slm.total_time_avg << "\n";
    csv << res_conv.name << "," << res_conv.forward_time_avg << "," << res_conv.backward_time_avg << "," << res_conv.total_time_avg << "\n";
    csv.close();

    return 0;
}
