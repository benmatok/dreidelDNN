#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/layers/Conv2D.hpp"
#include "../include/dreidel/optim/SGD.hpp"
#include "../include/dreidel/optim/RMSProp.hpp"
#include "../include/dreidel/optim/Adam.hpp"

using namespace dreidel;

// --- Helper Layers ---
template <typename T>
class Upscale2D : public layers::Layer<T> {
public:
    Upscale2D(size_t scale) : scale_(scale) {}
    Tensor<T> forward(const Tensor<T>& input) override {
        input_ = input;
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

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        auto shape = input_.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
        Tensor<T> grad_input(shape);
        grad_input.fill(0);

        auto g_shape = grad_output.shape();
        size_t H_out = g_shape[1]; size_t W_out = g_shape[2];

        const T* go_ptr = grad_output.data();
        T* gi_ptr = grad_input.data();

        #pragma omp parallel for collapse(3)
        for(size_t n=0; n<N; ++n) {
            for(size_t h_out=0; h_out<H_out; ++h_out) {
                for(size_t w_out=0; w_out<W_out; ++w_out) {
                    size_t h_in = h_out / scale_; size_t w_in = w_out / scale_;
                    for(size_t c=0; c<C; ++c) {
                        T val = go_ptr[((n*H_out + h_out)*W_out + w_out)*C + c];
                        #pragma omp atomic
                        gi_ptr[((n*H + h_in)*W + w_in)*C + c] += val;
                    }
                }
            }
        }
        return grad_input;
    }

    std::string name() const override { return "Upscale2D"; }
    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }

private:
    size_t scale_;
    Tensor<T> input_;
};

// --- Data Gen ---
template <typename T>
void generate_wavelet_batch(Tensor<T>& data) {
    auto shape = data.shape();
    size_t batch = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    static std::mt19937 gen(12345);
    T* ptr = data.data();

    #pragma omp parallel for
    for(size_t n=0; n<batch; ++n) {
        T mu_x = 0.5 * W + 0.2 * W * std::sin(n);
        T mu_y = 0.5 * H + 0.2 * H * std::cos(n);
        T s_x = (W/10.0) * (1.0 + 0.5 * std::sin(n*0.5));

        for(size_t c=0; c<C; ++c) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w_idx=0; w_idx<W; ++w_idx) {
                     T x = (T)w_idx - mu_x;
                     T y = (T)h - mu_y;
                     T val = std::exp(-(x*x + y*y)/(2*s_x*s_x));
                     ptr[((n*H + h)*W + w_idx)*C + c] = val;
                }
            }
        }
    }
}

// Build Optimized Zenith Autoencoder (Implicit Upscale) - Scaled Up C=64
std::vector<layers::Layer<float>*> build_zenith_autoencoder() {
    std::vector<layers::Layer<float>*> model;
    // Encoder: Stride=4, Upscale=1 (Downsample)
    // 64x64x1 -> 16x16x64
    model.push_back(new layers::ZenithBlock<float>(1, 64, 3, 1, true, true, false, 4, 1));
    // 16x16x64 -> 4x4x64
    model.push_back(new layers::ZenithBlock<float>(64, 64, 3, 64, true, true, false, 4, 1));
    // 4x4x64 -> 1x1x32
    model.push_back(new layers::ZenithBlock<float>(64, 32, 3, 64, true, true, false, 4, 1));

    // Decoder: Stride=1, Upscale=4 (Upsample)
    // 1x1x32 -> 4x4x64
    model.push_back(new layers::ZenithBlock<float>(32, 64, 3, 32, true, true, false, 1, 4));
    // 4x4x64 -> 16x16x64
    model.push_back(new layers::ZenithBlock<float>(64, 64, 3, 64, true, true, false, 1, 4));
    // 16x16x64 -> 64x64x1
    model.push_back(new layers::ZenithBlock<float>(64, 1, 3, 64, true, true, false, 1, 4));
    return model;
}

// Build Conv2D Autoencoder (Baseline with Explicit Upscale) - Scaled Up C=64
std::vector<layers::Layer<float>*> build_conv_autoencoder() {
    std::vector<layers::Layer<float>*> model;
    // Encoder
    model.push_back(new layers::Conv2D<float>(1, 64, 3, 4, 1));
    model.push_back(new layers::Conv2D<float>(64, 64, 3, 4, 1));
    model.push_back(new layers::Conv2D<float>(64, 32, 3, 4, 1));

    // Decoder
    model.push_back(new Upscale2D<float>(4));
    model.push_back(new layers::Conv2D<float>(32, 64, 3, 1, 1));

    model.push_back(new Upscale2D<float>(4));
    model.push_back(new layers::Conv2D<float>(64, 64, 3, 1, 1));

    model.push_back(new Upscale2D<float>(4));
    model.push_back(new layers::Conv2D<float>(64, 1, 3, 1, 1));

    return model;
}

// Training Loop Helper
float train_loop(std::string name, std::vector<layers::Layer<float>*> model, size_t epochs, size_t batch_size, float lr = 0.001f) {
    std::cout << "\n--- Starting Training: " << name << " ---" << std::endl;
    size_t H = 64, W = 64;

    optim::Adam<float> optimizer(lr);

    // Register params
    for(auto* layer : model) {
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (!params.empty() && params.size() == grads.size()) {
            optimizer.add_parameters(params, grads);
        }
    }

    Tensor<float> input({batch_size, H, W, 1});
    Tensor<float> target({batch_size, H, W, 1});
    float final_loss = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for(size_t epoch=0; epoch<epochs; ++epoch) {
        generate_wavelet_batch(input);
        target = input; // Copy input to target

        // Forward
        Tensor<float> x = input;
        for(auto* layer : model) {
            x = layer->forward(x);
        }
        Tensor<float> output = x;

        // Loss (MSE)
        float loss = 0;
        size_t total_elements = output.size();
        const float* out_ptr = output.data();
        const float* tgt_ptr = target.data();

        Tensor<float> grad_output(output.shape());
        float* go_ptr = grad_output.data();

        #pragma omp parallel for reduction(+:loss)
        for(size_t i=0; i<total_elements; ++i) {
            float diff = out_ptr[i] - tgt_ptr[i];
            loss += diff * diff;
            go_ptr[i] = 2.0f * diff / total_elements;
        }
        loss /= total_elements;
        final_loss = loss;

        // Backward
        optimizer.zero_grad();
        Tensor<float> grad = grad_output;
        for(int i = model.size() - 1; i >= 0; --i) {
            grad = model[i]->backward(grad);
        }

        // Update
        optimizer.step();

        if (epoch % 50 == 0 || epoch == epochs-1) {
            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time_sec = std::chrono::duration<double>(end - start).count();

    std::cout << name << " | Final Loss: " << final_loss << " | Time: " << time_sec << "s" << std::endl;

    for(auto* l : model) delete l;
    return final_loss;
}

int main() {
    std::cout << "=== Model Comparison Benchmark: Optimized Zenith (C=64) vs Conv2D (Adam, 500 Epochs) ===" << std::endl;

    size_t batch_size = 4;
    size_t epochs = 500;

    // 1. Zenith (Implicit Upscale)
    // Reduce LR slightly for stability
    train_loop("Optimized Zenith Autoencoder", build_zenith_autoencoder(), epochs, batch_size, 0.0001f);

    // 2. Conv2D
    train_loop("Conv2D Autoencoder", build_conv_autoencoder(), epochs, batch_size, 0.001f);

    return 0;
}
