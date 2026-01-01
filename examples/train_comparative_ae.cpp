#include "../include/dreidel/models/ComparativeAE.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/utils/WaveletGen2D.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace dreidel;
using namespace dreidel::models;
using namespace dreidel::utils;
using namespace dreidel::optim;

// MSE Loss
template <typename T>
T mse_loss(const Tensor<T>& pred, const Tensor<T>& target, Tensor<T>& grad_input) {
    size_t size = pred.size();
    if (size != target.size()) throw std::invalid_argument("Size mismatch in MSE");

    const T* p_ptr = pred.data();
    const T* t_ptr = target.data();
    T* g_ptr = grad_input.data();

    T sum_sq = 0;
    T scale = 2.0 / size;

    #pragma omp parallel for reduction(+:sum_sq)
    for (size_t i = 0; i < size; ++i) {
        T diff = p_ptr[i] - t_ptr[i];
        sum_sq += diff * diff;
        g_ptr[i] = diff * scale;
    }

    return sum_sq / size;
}

int main() {
    std::cout << "=== Training Comparative Autoencoders on Wavelets (32x32) ===\n";

    // Config
    const size_t H = 32;
    const size_t W = 32;
    const size_t C = 8;
    const size_t BatchSize = 8;
    const size_t Epochs = 5;
    const size_t StepsPerEpoch = 20;

    // Generator
    WaveletGenerator2D<float> gen(H, W);
    Tensor<float> batch_input({BatchSize, H, W, 3});
    Tensor<float> batch_grad({BatchSize, H, W, 3});

    // Models
    std::cout << "Initializing ZenithHierarchicalAE..." << std::endl;
    ZenithHierarchicalAE<float> zenith_ae(C);

    std::cout << "Initializing ConvBaselineAE..." << std::endl;
    ConvBaselineAE<float> conv_ae(C);

    // Optimizers
    SimpleAdam<float> opt_zenith(1e-3);
    opt_zenith.add_parameters(zenith_ae.parameters(), zenith_ae.gradients());

    SimpleAdam<float> opt_conv(1e-3);
    opt_conv.add_parameters(conv_ae.parameters(), conv_ae.gradients());

    // Training Loop
    for (size_t epoch = 0; epoch < Epochs; ++epoch) {
        float loss_z_acc = 0;
        float loss_c_acc = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t step = 0; step < StepsPerEpoch; ++step) {
            // 1. Generate Data
            gen.generate_batch(batch_input, BatchSize);

            // 2. Train Zenith
            opt_zenith.zero_grad();
            Tensor<float> out_z = zenith_ae.forward(batch_input);
            float loss_z = mse_loss(out_z, batch_input, batch_grad);
            zenith_ae.backward(batch_grad);
            opt_zenith.step();
            loss_z_acc += loss_z;

            // 3. Train Conv
            opt_conv.zero_grad();
            Tensor<float> out_c = conv_ae.forward(batch_input);
            float loss_c = mse_loss(out_c, batch_input, batch_grad); // reuse grad buffer
            conv_ae.backward(batch_grad);
            opt_conv.step();
            loss_c_acc += loss_c;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

        std::cout << "Epoch " << epoch+1 << "/" << Epochs
                  << " | Time: " << std::fixed << std::setprecision(2) << duration << "s"
                  << " | Loss Z: " << std::setprecision(5) << loss_z_acc / StepsPerEpoch
                  << " | Loss C: " << std::setprecision(5) << loss_c_acc / StepsPerEpoch << std::endl;
    }

    std::cout << "\nTraining Complete.\n";

    // Final Validation on one batch
    std::cout << "Final Validation (One Batch):\n";
    gen.generate_batch(batch_input, BatchSize);
    Tensor<float> val_z = zenith_ae.forward(batch_input);
    Tensor<float> val_c = conv_ae.forward(batch_input);
    float lz = mse_loss(val_z, batch_input, batch_grad);
    float lc = mse_loss(val_c, batch_input, batch_grad);

    std::cout << "  Zenith MSE: " << lz << "\n";
    std::cout << "  Conv   MSE: " << lc << "\n";

    return 0;
}
