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

void run_autoencoder_benchmark() {
    // Requested Architecture:
    // 64x64x1 -> 32x32x256 -> 16x16x128 -> 8x8x64 -> 1x1x64
    // -> 8x8x64 -> 16x16x128 -> 32x32x256 -> 64x64x1

    size_t batch_size = 4;
    size_t H_in = 64, W_in = 64, C_in = 1;
    size_t loops = 2;

    std::cout << "=== Autoencoder Benchmark: Zenith vs Conv2D ===" << std::endl;
    std::cout << "Architecture: 64x64x1 -> 32x32x256 -> 16x16x128 -> 8x8x64 -> 1x1x64 -> 8x8x64 -> 16x16x128 -> 32x32x256 -> 64x64x1" << std::endl;
    std::cout << "Batch: " << batch_size << ", Loops: " << loops << std::endl;

    Tensor<float> input({batch_size, H_in, W_in, C_in});
    generate_wavelet_images(input);

    // --- 1. Zenith Autoencoder ---

    // Encoder
    // 64x64x1 -> 32x32x256 (Stride 2)
    layers::ZenithBlock<float> z_e1(1, 256, 3, 256, true, true, false, 2);
    // 32x32x256 -> 16x16x128 (Stride 2)
    layers::ZenithBlock<float> z_e2(256, 128, 3, 256, true, true, false, 2);
    // 16x16x128 -> 8x8x64 (Stride 2)
    layers::ZenithBlock<float> z_e3(128, 64, 3, 128, true, true, false, 2);
    // 8x8x64 -> 1x1x64 (Stride 8)
    layers::ZenithBlock<float> z_e4(64, 64, 3, 64, true, true, false, 8);

    // Decoder
    // 1x1x64 -> 8x8x64 (Upscale 8)
    Upscale2D<float> z_up1(8);
    layers::ZenithBlock<float> z_d1(64, 64, 3, 64, true, true, false, 1);

    // 8x8x64 -> 16x16x128 (Upscale 2)
    Upscale2D<float> z_up2(2);
    layers::ZenithBlock<float> z_d2(64, 128, 3, 128, true, true, false, 1);

    // 16x16x128 -> 32x32x256 (Upscale 2)
    Upscale2D<float> z_up3(2);
    layers::ZenithBlock<float> z_d3(128, 256, 3, 256, true, true, false, 1);

    // 32x32x256 -> 64x64x1 (Upscale 2)
    Upscale2D<float> z_up4(2);
    layers::ZenithBlock<float> z_d4(256, 1, 3, 256, true, true, false, 1);

    // Warmup Zenith
    {
        auto t1 = z_e1.forward(input);
        auto t2 = z_e2.forward(t1);
        auto t3 = z_e3.forward(t2);
        auto t4 = z_e4.forward(t3); // Bottleneck 1x1x64

        auto d1 = z_up1.forward(t4);
        auto d2 = z_d1.forward(d1);
        auto d3 = z_up2.forward(d2);
        auto d4 = z_d2.forward(d3);
        auto d5 = z_up3.forward(d4);
        auto d6 = z_d3.forward(d5);
        auto d7 = z_up4.forward(d6);
        auto out = z_d4.forward(d7);
    }

    auto start_z = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto t1 = z_e1.forward(input);
        auto t2 = z_e2.forward(t1);
        auto t3 = z_e3.forward(t2);
        auto t4 = z_e4.forward(t3);

        auto d1 = z_up1.forward(t4);
        auto d2 = z_d1.forward(d1);
        auto d3 = z_up2.forward(d2);
        auto d4 = z_d2.forward(d3);
        auto d5 = z_up3.forward(d4);
        auto d6 = z_d3.forward(d5);
        auto d7 = z_up4.forward(d6);
        auto out = z_d4.forward(d7);
    }
    auto end_z = std::chrono::high_resolution_clock::now();
    double time_z = std::chrono::duration<double>(end_z - start_z).count();


    // --- 2. Conv2D Autoencoder ---
    layers::Conv2D<float> c_e1(1, 256, 3, 2, 1);
    layers::Conv2D<float> c_e2(256, 128, 3, 2, 1);
    layers::Conv2D<float> c_e3(128, 64, 3, 2, 1);
    layers::Conv2D<float> c_e4(64, 64, 3, 8, 1); // Stride 8 needs proper padding in naive conv?
    // Naive Conv2D logic: (H+2p-k)/s + 1. (8+2-3)/8 + 1 = 7/8 + 1 = 1. Matches.

    Upscale2D<float> c_up1(8);
    layers::Conv2D<float> c_d1(64, 64, 3, 1, 1);
    Upscale2D<float> c_up2(2);
    layers::Conv2D<float> c_d2(64, 128, 3, 1, 1);
    Upscale2D<float> c_up3(2);
    layers::Conv2D<float> c_d3(128, 256, 3, 1, 1);
    Upscale2D<float> c_up4(2);
    layers::Conv2D<float> c_d4(256, 1, 3, 1, 1);

    // Warmup Conv
    {
        auto t1 = c_e1.forward(input);
        auto t2 = c_e2.forward(t1);
        auto t3 = c_e3.forward(t2);
        auto t4 = c_e4.forward(t3);

        auto d1 = c_up1.forward(t4);
        auto d2 = c_d1.forward(d1);
        auto d3 = c_up2.forward(d2);
        auto d4 = c_d2.forward(d3);
        auto d5 = c_up3.forward(d4);
        auto d6 = c_d3.forward(d5);
        auto d7 = c_up4.forward(d6);
        auto out = c_d4.forward(d7);
    }

    auto start_c = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto t1 = c_e1.forward(input);
        auto t2 = c_e2.forward(t1);
        auto t3 = c_e3.forward(t2);
        auto t4 = c_e4.forward(t3);

        auto d1 = c_up1.forward(t4);
        auto d2 = c_d1.forward(d1);
        auto d3 = c_up2.forward(d2);
        auto d4 = c_d2.forward(d3);
        auto d5 = c_up3.forward(d4);
        auto d6 = c_d3.forward(d5);
        auto d7 = c_up4.forward(d6);
        auto out = c_d4.forward(d7);
    }
    auto end_c = std::chrono::high_resolution_clock::now();
    double time_c = std::chrono::duration<double>(end_c - start_c).count();

    // --- Report ---
    std::cout << "\nResults:" << std::endl;
    std::cout << std::left << std::setw(20) << "Model" << "Time (s)" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    std::cout << std::left << std::setw(20) << "Zenith AE" << time_z << std::endl;
    std::cout << std::left << std::setw(20) << "Conv2D AE" << time_c << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    std::cout << "Speedup: " << time_c / time_z << "x" << std::endl;
}

int main() {
    run_autoencoder_benchmark();
    return 0;
}
