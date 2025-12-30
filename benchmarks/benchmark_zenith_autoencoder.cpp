#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>

// Include Dreidel headers
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/Layer.hpp"
#include "../include/dreidel/layers/Conv2D.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"

using namespace dreidel;

// Helpers
template <typename T>
void generate_wavelet_images(Tensor<T>& data) {
    auto shape = data.shape();
    size_t batch = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    static std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist_type(0, 19);
    std::uniform_real_distribution<T> dist_param(0.5, 2.0);
    std::uniform_real_distribution<T> dist_pos(0.2, 0.8);
    T* ptr = data.data();
    auto get_wavelet_val = [&](int type, T t, T mu, T s, T w) -> T {
        T x = t - mu; T val = 0;
        switch(type) {
             case 0: val = std::cos(w*x) * std::exp(-x*x/(2*s*s)); break;
             default: val = std::exp(-x*x/(2*s*s)); break;
        }
        return val;
    };
    for(size_t n=0; n<batch; ++n) {
        for(size_t c=0; c<C; ++c) {
            T mu_x = dist_pos(gen) * W; T mu_y = dist_pos(gen) * H;
            T s_x = dist_param(gen) * (W/10.0);
            for(size_t h=0; h<H; ++h) {
                for(size_t w_idx=0; w_idx<W; ++w_idx) {
                     T wx = get_wavelet_val(0, (T)w_idx, mu_x, s_x, 1.0);
                     ptr[((n*H + h)*W + w_idx)*C + c] = wx;
                }
            }
        }
    }
}

// Simple Autoencoder Benchmark
// Compares:
// 1. Zenith AE (Downsample in Spectral domain)
// 2. Conv2D AE (Not easily implemented with standard Conv2D without stride/transpose support in Conv2D.hpp, so we just run Zenith AE as proof of concept)

void run_autoencoder_benchmark() {
    size_t C_in = 64;
    size_t C_mid = 32;
    size_t batch_size = 8;
    size_t H = 64, W = 64;
    size_t loops = 10;

    std::cout << "=== Zenith Autoencoder Benchmark ===" << std::endl;
    std::cout << "Input: " << H << "x" << W << "x" << C_in << std::endl;
    std::cout << "Bottleneck: " << H << "x" << W << "x" << C_mid << std::endl;

    Tensor<float> input({batch_size, H, W, C_in});
    generate_wavelet_images(input);

    // Encoder: Zenith (C -> C/2)
    layers::ZenithBlock<float> encoder(C_in, C_mid, 3, C_in);

    // Decoder: Zenith (C/2 -> C)
    layers::ZenithBlock<float> decoder(C_mid, C_in, 3, C_mid);

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto latent = encoder.forward(input);
        auto recon = decoder.forward(latent);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();

    std::cout << "Zenith Autoencoder Time: " << time << " s" << std::endl;
    std::cout << "Throughput: " << (batch_size * loops) / time << " images/s" << std::endl;
}

int main() {
    run_autoencoder_benchmark();
    return 0;
}
