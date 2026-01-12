#include "dreidel/models/TAESD.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <iomanip>

using namespace dreidel::taesd;

int main(int argc, char** argv) {
    std::string encoder_path = "taesd_encoder.bin";
    std::string decoder_path = "taesd_decoder.bin";

    if (argc > 1) encoder_path = argv[1];
    if (argc > 2) decoder_path = argv[2];

    // Load Models
    std::cout << "Loading Models..." << std::endl;
    Encoder encoder;
    Decoder decoder;
    try {
        encoder.load_from_file(encoder_path.c_str());
        decoder.load_from_file(decoder_path.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load models: " << e.what() << std::endl;
        return 1;
    }

    // Input Dimensions (Default 512, can be overridden if we want to test smaller sizes for speed)
    // The "Wavelet Example" (benchmark_wavelet_ae.cpp) uses 128x128.
    // Let's use 128x128 to match the "wavelet example" sizing.

    const int H = 128;
    const int W = 128;
    const int C = 3;
    const int BATCH = 1;

    std::cout << "Running TAESD Auto-Encoding on " << H << "x" << W << " wavelet-like inputs..." << std::endl;

    // Prepare Tensors
    Tensor image(H, W, C);
    Tensor latent(H/8, W/8, 4); // TAESD reduction is 8x
    Tensor recon(H, W, C);

    // Generate Random Data
    // We fill with random data to simulate content.
    #pragma omp parallel for
    for(size_t i=0; i<image.data.size(); ++i) {
        image.data[i] = (float)rand() / RAND_MAX;
    }

    // Warmup
    std::cout << "Warmup..." << std::endl;
    for(int i=0; i<3; ++i) {
        encoder.forward(image, latent);
        decoder.forward(latent, recon);
    }

    // Benchmark
    int iterations = 20;
    std::cout << "Benchmarking (" << iterations << " iterations)..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<iterations; ++i) {
        encoder.forward(image, latent);
        decoder.forward(latent, recon);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end - start;
    double avg_time = total_time.count() / iterations;

    std::cout << "Results for " << H << "x" << W << ":" << std::endl;
    std::cout << "  Total Time: " << total_time.count() << " s" << std::endl;
    std::cout << "  Avg Latency: " << avg_time * 1000.0 << " ms" << std::endl;
    std::cout << "  FPS: " << 1.0 / avg_time << std::endl;

    if (avg_time * 1000.0 < 20.0) {
        std::cout << "[SUCCESS] Speed is sub-20ms!" << std::endl;
    } else {
        std::cout << "[WARNING] Speed is " << avg_time * 1000.0 << "ms (Target: <20ms)" << std::endl;
    }

    return 0;
}
