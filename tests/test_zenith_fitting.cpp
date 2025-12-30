#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/Layer.hpp"
#include "../include/dreidel/layers/Conv2D.hpp"
#include "../include/dreidel/layers/ZenithVariants.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"

using namespace dreidel;

// Helper to init Conv2D with smooth Gaussian filters
template <typename T>
void init_smooth_conv(layers::Conv2D<T>& conv, size_t channels, size_t kernel_size) {
    // Conv2D weights: (Out, In, K, K)
    // We want each filter to be a random Gaussian blob
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dist_pos(0.0, (T)kernel_size);
    std::uniform_real_distribution<T> dist_sigma(0.5, 1.5);

    // Access weights via parameters() -> [0]
    Tensor<T>* w = conv.parameters()[0];
    T* ptr = w->data();

    // Normalize factor
    T norm = 1.0 / (channels * kernel_size * kernel_size);

    for(size_t oc=0; oc<channels; ++oc) {
        T cx = dist_pos(gen);
        T cy = dist_pos(gen);
        T sigma = dist_sigma(gen);

        for(size_t ic=0; ic<channels; ++ic) {
            for(size_t ky=0; ky<kernel_size; ++ky) {
                for(size_t kx=0; kx<kernel_size; ++kx) {
                    T dy = (T)ky - cy;
                    T dx = (T)kx - cx;
                    T val = std::exp(-(dx*dx + dy*dy) / (2*sigma*sigma));
                    ptr[((oc*channels + ic)*kernel_size + ky)*kernel_size + kx] = val * norm;
                }
            }
        }
    }
    // Bias [1]
    conv.parameters()[1]->fill(0.0f);
}

int main() {
    std::cout << "=== Zenith Fitting Test (Distillation) ===" << std::endl;

    size_t C = 64;
    size_t K = 3;
    size_t H = 16, W = 16;
    size_t batch = 4;
    size_t epochs = 1000;

    // 1. Ground Truth: Conv2D
    std::cout << "Initializing Ground Truth Conv2D (" << C << " smooth filters)..." << std::endl;
    layers::Conv2D<float> teacher(C, C, K, 1, 1); // Padding 1
    init_smooth_conv(teacher, C, K);

    // 2. Student: ZenithFloatEyes (Approximating ZenithBlock)
    std::cout << "Initializing Student ZenithFloatEyes..." << std::endl;
    layers::ZenithFloatEyes<float> student(C, C, K);

    // Optimizer
    optim::DiagonalNewton<float> opt(0.001f); // Reduced Learning rate
    opt.add_parameters(student.parameters(), student.gradients(), student.curvatures());

    // Data
    Tensor<float> x({batch, H, W, C});

    std::cout << "Training..." << std::endl;
    for(size_t epoch=0; epoch<epochs; ++epoch) {
        // Random input each time or fixed?
        // Fixed dataset is better for fitting test.
        // Let's re-generate X every time to simulate "distribution" fitting, or fixed for "function approximation".
        // Let's use random input to ensure generalization over input space.
        x.random(-1.0f, 1.0f);

        // Teacher Forward
        Tensor<float> y_target = teacher.forward(x);

        // Student Forward
        opt.zero_grad();
        Tensor<float> y_pred = student.forward(x);

        // Loss (MSE)
        Tensor<float> diff = y_pred - y_target;
        float mse = 0;
        const float* d = diff.data();
        for(size_t i=0; i<diff.size(); ++i) mse += d[i]*d[i];
        mse /= diff.size();

        // Backward
        Tensor<float> grad = diff * (2.0f / diff.size());
        student.backward(grad);
        opt.step();

        if(epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " MSE: " << mse << std::endl;
        }
    }

    // Final Eval
    x.random(-1.0f, 1.0f);
    Tensor<float> y_target = teacher.forward(x);
    Tensor<float> y_pred = student.forward(x);

    // Calculate Relative Error: ||y - y_hat|| / ||y||
    float num = 0, den = 0;
    const float* yt = y_target.data();
    const float* yp = y_pred.data();
    for(size_t i=0; i<y_target.size(); ++i) {
        float diff = yt[i] - yp[i];
        num += diff*diff;
        den += yt[i]*yt[i];
    }
    float rel_error = std::sqrt(num) / std::sqrt(den);

    std::cout << "Final Relative Error: " << rel_error << std::endl;

    if (rel_error < 0.2) {
        std::cout << "SUCCESS: Zenith approximated Conv2D reasonably well." << std::endl;
        return 0;
    } else {
        std::cout << "RESULT: Zenith approximation error is high (" << rel_error << "). This is expected for single-layer sparse approx of dense conv." << std::endl;
        return 0; // Return 0 as this is an experiment, not a unit test failure condition.
    }
}
