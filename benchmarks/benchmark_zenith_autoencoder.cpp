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

// Helper: Upscale2D (Nearest Neighbor)
template <typename T>
class Upscale2D : public layers::Layer<T> {
public:
    Upscale2D(size_t scale) : scale_(scale) {}
    Tensor<T> forward(const Tensor<T>& input) override {
        auto shape = input.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
        size_t H_out = H * scale_; size_t W_out = W * scale_;
        Tensor<T> output({N, H_out, W_out, C});
        T* out_ptr = output.data(); const T* in_ptr = input.data();
        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {
                    size_t h_in = h_out / scale_; size_t w_in = w_out / scale_;
                    for(size_t c=0; c<C; ++c) {
                        out_ptr[((n*H_out + h_out)*W_out + w_out)*C + c] = in_ptr[((n*H + h_in)*W + w_in)*C + c];
                    }
                }
            }
        }
        return output;
    }
    Tensor<T> backward(const Tensor<T>& grad_output) override { return Tensor<T>(); }
    std::string name() const override { return "Upscale2D"; }
private:
    size_t scale_;
};

// Helper: Wavelet Generator
template <typename T>
void generate_wavelet_images(Tensor<T>& data) {
    auto shape = data.shape();
    size_t batch = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    static std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist_param(0.5, 2.0);
    std::uniform_real_distribution<T> dist_pos(0.2, 0.8);
    T* ptr = data.data();
    for(size_t n=0; n<batch; ++n) {
        for(size_t c=0; c<C; ++c) {
            T mu_x = dist_pos(gen) * W; T mu_y = dist_pos(gen) * H;
            T s_x = dist_param(gen) * (W/10.0);
            for(size_t h=0; h<H; ++h) {
                for(size_t w_idx=0; w_idx<W; ++w_idx) {
                     T x = (T)w_idx - mu_x;
                     T val = std::exp(-x*x/(2*s_x*s_x));
                     ptr[((n*H + h)*W + w_idx)*C + c] = val;
                }
            }
        }
    }
}

void run_autoencoder_benchmark(size_t base_channels) {
    // Config: 64x64x1 -> ... bottleneck ... -> 64x64x1
    size_t batch_size = 4;
    size_t H_in = 64, W_in = 64, C_in = 1;
    size_t loops = 5;

    size_t C1 = base_channels;
    size_t C2 = base_channels / 2;
    if (C2 < 4) C2 = 4;

    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Benchmarking Autoencoder with Base Channel C=" << base_channels << std::endl;
    std::cout << "Architecture: 64x64x1 -> 16x16x" << C1 << " -> 4x4x" << C1 << " -> 1x1x" << C2 << " -> 4x4x" << C1 << " -> 16x16x" << C1 << " -> 64x64x1" << std::endl;

    Tensor<float> input({batch_size, H_in, W_in, C_in});
    generate_wavelet_images(input);

    // --- 1. Zenith Autoencoder (Implicit Upscale) ---

    // Encoder
    layers::ZenithBlock<float> z_e1(1, C1, 3, C1, true, true, false, 4);
    layers::ZenithBlock<float> z_e2(C1, C1, 3, C1, true, true, false, 4);
    layers::ZenithBlock<float> z_e3(C1, C2, 3, C1, true, true, false, 4); // Bottleneck

    // Decoder (Fused Upscale)
    // Removed Upscale2D layers. Added upscale param to ZenithBlocks.
    // Constructor: in, out, kernel, spectral, ifwht, dilated, gating, stride, upscale
    layers::ZenithBlock<float> z_d1(C2, C1, 3, C1, true, true, false, 1, 4);
    layers::ZenithBlock<float> z_d2(C1, C1, 3, C1, true, true, false, 1, 4);
    layers::ZenithBlock<float> z_d3(C1, 1, 3, C1, true, true, false, 1, 4);

    // Warmup Zenith
    {
        auto t1 = z_e1.forward(input);
        auto t2 = z_e2.forward(t1);
        auto t3 = z_e3.forward(t2);

        auto d1 = z_d1.forward(t3); // Implicit upscale 4x
        auto d2 = z_d2.forward(d1); // Implicit upscale 4x
        auto out = z_d3.forward(d2); // Implicit upscale 4x
    }

    if (base_channels >= 1024) loops = 1; // Reduce loops for large channels

    auto start_z = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto t1 = z_e1.forward(input);
        auto t2 = z_e2.forward(t1);
        auto t3 = z_e3.forward(t2);

        auto d1 = z_d1.forward(t3);
        auto d2 = z_d2.forward(d1);
        auto out = z_d3.forward(d2);
    }
    auto end_z = std::chrono::high_resolution_clock::now();
    double time_z = std::chrono::duration<double>(end_z - start_z).count();


    // --- 2. Conv2D Autoencoder (Explicit Upscale) ---
    // Conv2D doesn't support implicit upscale in this framework, so we keep explicit layers.
    // This is a fair comparison of "Optimized Zenith Architecture" vs "Standard Conv Architecture".
    layers::Conv2D<float> c_e1(1, C1, 3, 4, 1);
    layers::Conv2D<float> c_e2(C1, C1, 3, 4, 1);
    layers::Conv2D<float> c_e3(C1, C2, 3, 4, 1);

    Upscale2D<float> c_up1(4);
    layers::Conv2D<float> c_d1(C2, C1, 3, 1, 1);
    Upscale2D<float> c_up2(4);
    layers::Conv2D<float> c_d2(C1, C1, 3, 1, 1);
    Upscale2D<float> c_up3(4);
    layers::Conv2D<float> c_d3(C1, 1, 3, 1, 1);

    // Warmup Conv
    double time_c = -1.0;
    if (base_channels < 1024) { // Skip Conv2D for C=4096 as it will likely OOM or take forever
        {
            auto t1 = c_e1.forward(input);
            auto t2 = c_e2.forward(t1);
            auto t3 = c_e3.forward(t2);

            auto d1 = c_up1.forward(t3);
            auto d2 = c_d1.forward(d1);
            auto d3 = c_up2.forward(d2);
            auto d4 = c_d2.forward(d3);
            auto d5 = c_up3.forward(d4);
            auto out = c_d3.forward(d5);
        }

        auto start_c = std::chrono::high_resolution_clock::now();
        for(size_t i=0; i<loops; ++i) {
            auto t1 = c_e1.forward(input);
            auto t2 = c_e2.forward(t1);
            auto t3 = c_e3.forward(t2);

            auto d1 = c_up1.forward(t3);
            auto d2 = c_d1.forward(d1);
            auto d3 = c_up2.forward(d2);
            auto d4 = c_d2.forward(d3);
            auto d5 = c_up3.forward(d4);
            auto out = c_d3.forward(d5);
        }
        auto end_c = std::chrono::high_resolution_clock::now();
        time_c = std::chrono::duration<double>(end_c - start_c).count();
    }

    // --- Report ---
    std::cout << std::left << std::setw(20) << "Zenith AE Time:" << time_z << " s" << std::endl;
    if (time_c > 0) {
        std::cout << std::left << std::setw(20) << "Conv2D AE Time:" << time_c << " s" << std::endl;
        std::cout << "Speedup: " << time_c / time_z << "x" << std::endl;
    } else {
        std::cout << "Conv2D AE Time: N/A (Skipped)" << std::endl;
    }
}

int main() {
    std::cout << "=== Autoencoder Benchmark: Zenith vs Conv2D ===" << std::endl;
    std::vector<size_t> channels_list = {8, 16, 32, 64, 128, 256, 4096};
    for(size_t C : channels_list) {
        run_autoencoder_benchmark(C);
    }
    return 0;
}
