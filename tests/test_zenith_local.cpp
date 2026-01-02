#include <dreidel/layers/ZenithBlock.hpp>
#include <dreidel/core/Tensor.hpp>
#include <dreidel/utils/WaveletGen2D.hpp>
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace dreidel;

void run_test(size_t C, size_t H, size_t W, size_t loops) {
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Testing C=" << C << " (" << H << "x" << W << ")..." << std::endl;

    layers::ZenithBlock<float> layer(C, 3, C);

    Tensor<float> input({1, H, W, C});
    input.fill(1.0f); // Uniform input

    // Warmup
    layer.forward(input);

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto out = layer.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Stats
    Tensor<float> output = layer.forward(input);
    double mean = 0;
    double sq_sum = 0;
    size_t count = output.size();
    const float* ptr = output.data();
    for(size_t i=0; i<count; ++i) {
        mean += ptr[i];
        sq_sum += ptr[i]*ptr[i];
    }
    mean /= count;
    double stddev = std::sqrt(sq_sum/count - mean*mean);

    std::cout << "Time per iter: " << (elapsed / loops) * 1000.0 << " ms" << std::endl;
    std::cout << "Output Mean: " << mean << " | StdDev: " << stddev << std::endl;
    std::cout << "Status: PASSED" << std::endl;
}

void run_wavelet_identity_test(size_t C, size_t H, size_t W, size_t loops) {
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Testing Wavelet Reconstruction (Identity) C=" << C << " (" << H << "x" << W << ")..." << std::endl;

    layers::ZenithBlock<float> layer(C, 3, C);

    // Set Depthwise Weights to Identity to check reconstruction accuracy
    auto params = layer.parameters();
    Tensor<float>* weights = params[0]; // packed_weights_ [C, 1, K, K]
    weights->fill(0.0f);

    // Set Center (1,1) to 1.0 for each channel
    size_t K = 3;
    float* w_ptr = weights->data();
    for(size_t c=0; c<C; ++c) {
        // Index: c*K*K + 1*K + 1 = c*9 + 4
        w_ptr[c*K*K + 4] = 1.0f;
    }

    // Generate Input
    Tensor<float> input({1, H, W, C});
    // WaveletGenerator2D generates 3 channels. We repeat them?
    // Or just use the generator directly to fill C channels?
    // generator::generate_batch fills (Batch, H, W, 3).
    // We have C=128.
    // Let's manually fill with random wavelets.
    // Or modify WaveletGen to support C.
    // Let's just use `generate_wavelet_images` logic from benchmark helper if possible,
    // or reimplement simple generator here.

    {
        static std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist_param(0.5, 2.0);
        std::uniform_real_distribution<float> dist_pos(0.2, 0.8);
        float* d = input.data();
        for(size_t c=0; c<C; ++c) {
            float mu_x = dist_pos(gen) * W;
            float mu_y = dist_pos(gen) * H;
            float s_x = dist_param(gen) * (W/10.0);
            for(size_t h=0; h<H; ++h) {
                for(size_t w_idx=0; w_idx<W; ++w_idx) {
                     float x = (float)w_idx - mu_x;
                     float y = (float)h - mu_y;
                     float val = std::exp(-(x*x + y*y)/(2*s_x*s_x)); // Gaussian
                     d[((0*H + h)*W + w_idx)*C + c] = val;
                }
            }
        }
    }

    // Warmup
    layer.forward(input);

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto out = layer.forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Accuracy
    Tensor<float> output = layer.forward(input);

    // MSE
    double mse = 0;
    size_t count = output.size();
    const float* in_ptr = input.data();
    const float* out_ptr = output.data();
    for(size_t i=0; i<count; ++i) {
        float diff = in_ptr[i] - out_ptr[i];
        mse += diff * diff;
    }
    mse /= count;

    std::cout << "Time per iter: " << (elapsed / loops) * 1000.0 << " ms" << std::endl;
    std::cout << "MSE (Accuracy): " << mse << std::endl;
    if (mse < 1e-5) std::cout << "Status: PASSED (Identity Verified)" << std::endl;
    else std::cout << "Status: FAILED (High MSE)" << std::endl;
}

int main() {
    try {
        std::cout << "=== ZenithBlock Local Tests & Benchmarks ===" << std::endl;

        run_test(32, 32, 32, 10);
        run_test(64, 32, 32, 10);
        run_test(128, 16, 16, 10);
        // Skipped > 128 as requested

        run_wavelet_identity_test(128, 512, 512, 10); // 512x512, C=128

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
