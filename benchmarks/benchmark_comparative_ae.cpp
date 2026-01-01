#include "../include/dreidel/models/ComparativeAE.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <chrono>
#include <numeric>
#include <vector>
#include <string>

using namespace dreidel;
using namespace dreidel::models;

template<typename T>
size_t count_parameters(Layer<T>& model) {
    size_t count = 0;
    for (auto* p : model.parameters()) {
        count += p->size();
    }
    return count;
}

int main(int argc, char** argv) {
    // Configuration
    // Defaults match user request: B=4, Iterations=100
    // Optional CLI args for faster local testing
    size_t C = 16;
    size_t B = 4;
    size_t H = 256;
    size_t W = 256;
    size_t Iterations = 100;

    if (argc > 1) Iterations = std::stoul(argv[1]);
    if (argc > 2) B = std::stoul(argv[2]);

    std::cout << "--- Comparative Analysis: Spectral vs Spatial Autoencoders ---\n";
    std::cout << "Configuration: C=" << C << ", Batch=" << B << ", Input=" << H << "x" << W << " (NHWC)\n";
    std::cout << "Iterations: " << Iterations << "\n\n";

    // 1. Instantiate Models
    std::cout << "Instantiating ZenithHierarchicalAE..." << std::endl;
    ZenithHierarchicalAE<float> zenith_ae(C);

    std::cout << "Instantiating ConvBaselineAE..." << std::endl;
    ConvBaselineAE<float> conv_ae(C);

    // 2. Parameter Counts
    size_t zenith_params = count_parameters(zenith_ae);
    size_t conv_params = count_parameters(conv_ae);

    std::cout << "Parameter Counts:\n";
    std::cout << "  Zenith: " << zenith_params << "\n";
    std::cout << "  Conv:   " << conv_params << "\n\n";

    // 3. Generate Dummy Input
    // Note: User spec asked for (Batch, 3, H, W).
    // However, dreidelDNN layers (Conv2D, ZenithBlock) are implemented as NHWC (Batch, Height, Width, Channels).
    // We generate input in NHWC format to ensure correct processing.
    std::cout << "Generating Input Tensor (" << B << ", " << H << ", " << W << ", 3)...\n";
    Tensor<float> input({B, H, W, 3});
    input.random(0.0f, 1.0f);

    // Warmup
    std::cout << "Warming up...\n";
    zenith_ae.forward(input);
    conv_ae.forward(input);

    // 4. Benchmark Zenith
    std::cout << "Benchmarking ZenithHierarchicalAE (" << Iterations << " iterations)...\n";
    auto start_z = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < Iterations; ++i) {
        // Prevent optimization
        auto out = zenith_ae.forward(input);
        if (out.size() == 0) std::cerr << "Error\n";
    }
    auto end_z = std::chrono::high_resolution_clock::now();
    double duration_z = std::chrono::duration_cast<std::chrono::milliseconds>(end_z - start_z).count();
    double avg_z = duration_z / Iterations;

    std::cout << "  Total Time: " << duration_z << " ms\n";
    std::cout << "  Avg Inference: " << avg_z << " ms\n\n";

    // 5. Benchmark Conv
    std::cout << "Benchmarking ConvBaselineAE (" << Iterations << " iterations)...\n";
    auto start_c = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < Iterations; ++i) {
        auto out = conv_ae.forward(input);
        if (out.size() == 0) std::cerr << "Error\n";
    }
    auto end_c = std::chrono::high_resolution_clock::now();
    double duration_c = std::chrono::duration_cast<std::chrono::milliseconds>(end_c - start_c).count();
    double avg_c = duration_c / Iterations;

    std::cout << "  Total Time: " << duration_c << " ms\n";
    std::cout << "  Avg Inference: " << avg_c << " ms\n\n";

    // 6. Compression Ratio
    // Input Size
    size_t input_size = B * H * W * 3;
    // Bottleneck Size:
    // Stage 2 Output. H_out = H/32, W_out = W/32. C_out = C*256.
    size_t H_bot = H / 32;
    size_t W_bot = W / 32;
    size_t C_bot = C * 256;
    size_t bottleneck_size = B * H_bot * W_bot * C_bot;

    float compression_ratio = (float)input_size / (float)bottleneck_size;

    std::cout << "Bottleneck Analysis:\n";
    std::cout << "  Input Size (elements): " << input_size << "\n";
    std::cout << "  Bottleneck Size (elements): " << bottleneck_size << "\n";
    std::cout << "  Compression Ratio (Input/Bottleneck): " << compression_ratio << "\n";
    if (compression_ratio < 1.0) {
        std::cout << "  (Note: Representation is Expanded/Overcomplete)\n";
    } else {
        std::cout << "  (Note: Representation is Compressed)\n";
    }

    // Relative Efficiency
    std::cout << "\nSummary:\n";
    std::cout << "  Speedup (Conv / Zenith): " << avg_c / avg_z << "x\n";
    std::cout << "  Parameter Reduction (Conv / Zenith): " << (float)conv_params / (float)zenith_params << "x\n";

    return 0;
}
