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
    // Config: 64x64x1 -> 1x1x256 -> 64x64x1
    size_t batch_size = 4; // Smaller batch for deep model
    size_t H_in = 64, W_in = 64, C_in = 1;
    size_t C_bottleneck = 256;
    size_t loops = 2; // Even fewer loops due to depth

    std::cout << "=== Autoencoder Benchmark: Zenith vs Conv2D ===" << std::endl;
    std::cout << "Structure: 64x64x1 -> 1x1x256 -> 64x64x1" << std::endl;
    std::cout << "Batch: " << batch_size << ", Loops: " << loops << std::endl;

    Tensor<float> input({batch_size, H_in, W_in, C_in});
    generate_wavelet_images(input);

    // --- 1. Zenith Autoencoder ---
    // Encoder: 3 layers, Stride 4 each (64->16->4->1). Channels 1->16->64->256
    // Decoder: 3 layers, Upscale 4 each. Channels 256->64->16->1

    // Encoder
    layers::ZenithBlock<float> z_e1(1, 16, 3, 1, true, true, false, 4);
    layers::ZenithBlock<float> z_e2(16, 64, 3, 16, true, true, false, 4);
    layers::ZenithBlock<float> z_e3(64, 256, 3, 64, true, true, false, 4);

    // Decoder
    Upscale2D<float> z_up1(4);
    layers::ZenithBlock<float> z_d1(256, 64, 3, 64, true, true, false, 1);
    Upscale2D<float> z_up2(4);
    layers::ZenithBlock<float> z_d2(64, 16, 3, 16, true, true, false, 1);
    Upscale2D<float> z_up3(4);
    layers::ZenithBlock<float> z_d3(16, 1, 3, 1, true, true, false, 1);

    // Warmup Zenith
    {
        auto t1 = z_e1.forward(input);
        auto t2 = z_e2.forward(t1);
        auto bottleneck = z_e3.forward(t2);

        // Check bottleneck shape
        // auto sh = bottleneck.shape();
        // std::cout << "Zenith Bottleneck: " << sh[1] << "x" << sh[2] << "x" << sh[3] << std::endl;

        auto t4 = z_up1.forward(bottleneck);
        auto t5 = z_d1.forward(t4);
        auto t6 = z_up2.forward(t5);
        auto t7 = z_d2.forward(t6);
        auto t8 = z_up3.forward(t7);
        auto out = z_d3.forward(t8);
    }

    auto start_z = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto t1 = z_e1.forward(input);
        auto t2 = z_e2.forward(t1);
        auto t3 = z_e3.forward(t2);
        auto t4 = z_up1.forward(t3);
        auto t5 = z_d1.forward(t4);
        auto t6 = z_up2.forward(t5);
        auto t7 = z_d2.forward(t6);
        auto t8 = z_up3.forward(t7);
        auto out = z_d3.forward(t8);
    }
    auto end_z = std::chrono::high_resolution_clock::now();
    double time_z = std::chrono::duration<double>(end_z - start_z).count();


    // --- 2. Conv2D Autoencoder ---
    // Same structure
    layers::Conv2D<float> c_e1(1, 16, 3, 4, 1);
    layers::Conv2D<float> c_e2(16, 64, 3, 4, 1);
    layers::Conv2D<float> c_e3(64, 256, 3, 4, 1);

    Upscale2D<float> c_up1(4);
    layers::Conv2D<float> c_d1(256, 64, 3, 1, 1);
    Upscale2D<float> c_up2(4);
    layers::Conv2D<float> c_d2(64, 16, 3, 1, 1);
    Upscale2D<float> c_up3(4);
    layers::Conv2D<float> c_d3(16, 1, 3, 1, 1);

    // Warmup Conv
    {
        auto t1 = c_e1.forward(input);
        auto t2 = c_e2.forward(t1);
        auto bottleneck = c_e3.forward(t2);

        auto t4 = c_up1.forward(bottleneck);
        auto t5 = c_d1.forward(t4);
        auto t6 = c_up2.forward(t5);
        auto t7 = c_d2.forward(t6);
        auto t8 = c_up3.forward(t7);
        auto out = c_d3.forward(t8);
    }

    auto start_c = std::chrono::high_resolution_clock::now();
    for(size_t i=0; i<loops; ++i) {
        auto t1 = c_e1.forward(input);
        auto t2 = c_e2.forward(t1);
        auto t3 = c_e3.forward(t2);
        auto t4 = c_up1.forward(t3);
        auto t5 = c_d1.forward(t4);
        auto t6 = c_up2.forward(t5);
        auto t7 = c_d2.forward(t6);
        auto t8 = c_up3.forward(t7);
        auto out = c_d3.forward(t8);
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
