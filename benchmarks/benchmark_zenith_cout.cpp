#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/Layer.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/layers/Quantization.hpp"
#include "../include/dreidel/layers/QuantizedStandard.hpp"

using namespace dreidel;

void run_benchmark_cout(size_t Cin, size_t Cout) {
    size_t batch_size = 4;
    size_t H = 16, W = 16;
    size_t loops = 5;

    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Benchmarking Cin=" << Cin << " -> Cout=" << Cout << std::endl;

    size_t arena_size = 4 * 1024 * 1024;
    layers::ZenithBlock zenith(Cin, Cout, 3, std::max(Cin, Cout), arena_size);

    Tensor<int8_t> input({batch_size, H, W, Cin});
    input.fill(1); // Dummy data

    // Warmup
    auto out = zenith.forward(input);
    if (out.shape()[3] != Cout) {
        std::cerr << "Error: Output channel mismatch! Expected " << Cout << ", got " << out.shape()[3] << std::endl;
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        zenith.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time_s = std::chrono::duration<double>(end - start).count();

    std::cout << "Time: " << time_s << " s" << std::endl;
}

int main() {
    std::cout << "=== ZenithBlock Cin vs Cout Benchmark ===" << std::endl;
    run_benchmark_cout(32, 32);
    run_benchmark_cout(32, 64);
    run_benchmark_cout(64, 32);
    run_benchmark_cout(128, 256);
    return 0;
}
