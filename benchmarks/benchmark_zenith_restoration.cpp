#include "../include/dreidel/models/ZenithHierarchicalAE.hpp"
#include "../include/dreidel/utils/WaveletGen2D.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <algorithm>

using namespace dreidel;

// --- Noise Functions ---

template <typename T>
void add_gaussian_noise(Tensor<T>& t, float stddev, std::mt19937& rng) {
    std::normal_distribution<float> d(0.0f, stddev);
    T* ptr = t.data();
    size_t sz = t.size();
    #pragma omp parallel for
    for(size_t i=0; i<sz; ++i) {
        ptr[i] += d(rng);
    }
}

template <typename T>
void add_shot_noise(Tensor<T>& t, float prob, float magnitude, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    std::bernoulli_distribution b(prob);
    T* ptr = t.data();
    size_t sz = t.size();

    // Serial loop for random generation usually, or thread-local RNGs.
    // For simplicity in benchmark, we accept serial overhead or use simple rand in parallel (not safe).
    // Let's use serial for safety here.
    for(size_t i=0; i<sz; ++i) {
        if (b(rng)) {
            // Add spike
            ptr[i] += (d(rng) > 0.5f ? magnitude : -magnitude);
        }
    }
}

template <typename T>
void apply_blur(Tensor<T>& t) {
    // Simple 3x3 Box Blur
    auto shape = t.shape(); // N, H, W, C
    size_t N = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    size_t C = shape[3];

    Tensor<T> copy = t; // Deep copy
    const T* src = copy.data();
    T* dst = t.data();

    #pragma omp parallel for collapse(3)
    for(size_t n=0; n<N; ++n) {
        for(size_t h=0; h<H; ++h) {
            for(size_t w=0; w<W; ++w) {
                for(size_t c=0; c<C; ++c) {
                    float sum = 0;
                    int count = 0;
                    for(int ky=-1; ky<=1; ++ky) {
                        for(int kx=-1; kx<=1; ++kx) {
                            int ih = h + ky;
                            int iw = w + kx;
                            if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
                                sum += src[((n*H + ih)*W + iw)*C + c];
                                count++;
                            }
                        }
                    }
                    dst[((n*H + h)*W + w)*C + c] = sum / count;
                }
            }
        }
    }
}

// --- Metrics ---

template <typename T>
T compute_mse(const Tensor<T>& a, const Tensor<T>& b) {
    if (a.size() != b.size()) return -1.0;
    T sum = 0;
    const T* ap = a.data();
    const T* bp = b.data();

    #pragma omp parallel for reduction(+:sum)
    for(size_t i=0; i<a.size(); ++i) {
        T diff = ap[i] - bp[i];
        sum += diff * diff;
    }
    return sum / a.size();
}

// --- Denoising Trainer ---

void train_denoising(models::ZenithHierarchicalAE<float>& model,
                     const std::vector<Tensor<float>>& clean_data,
                     int epochs, float lr) {

    optim::SimpleAdam<float> optimizer(lr);
    optimizer.add_parameters(model.parameters(), model.gradients());

    std::mt19937 rng(42);
    size_t N = clean_data.size();

    std::cout << "Training Denoising AE (" << epochs << " steps)..." << std::endl;

    for(int step=0; step<epochs; ++step) {
        // Sample batch
        Tensor<float> target = clean_data[step % N]; // Copy
        Tensor<float> input = target; // Copy

        // Mix of noises during training for robustness
        int noise_type = step % 3;
        if (noise_type == 0) add_gaussian_noise(input, 0.2f, rng);
        else if (noise_type == 1) add_shot_noise(input, 0.05f, 2.0f, rng);
        else apply_blur(input);

        optimizer.zero_grad();
        Tensor<float> output = model.forward(input);

        // MSE Loss against CLEAN target
        float mse = compute_mse(target, output);

        // Backward
        Tensor<float> grad_output = output;
        float* g = grad_output.data();
        const float* o = output.data();
        const float* t = target.data();
        float scale = 2.0f / output.size();

        #pragma omp parallel for
        for(size_t i=0; i<output.size(); ++i) {
            g[i] = (o[i] - t[i]) * scale;
        }

        model.backward(grad_output);
        optimizer.step();

        if (step % 50 == 0) {
            std::cout << "Step " << step << " Loss: " << mse << std::endl;
        }
    }
}

// --- Benchmark Main ---

int main() {
    size_t H = 64;
    size_t W = 64;
    size_t C = 3;
    size_t batch_size = 4;
    size_t base_channels = 16;
    int train_steps = 200; // Brief training

    // 1. Data Gen
    std::cout << "Generating Clean Wavelets..." << std::endl;
    utils::WaveletGenerator2D<float> gen(H, W);
    std::vector<Tensor<float>> train_data;
    train_data.reserve(50);
    for(int i=0; i<50; ++i) {
        Tensor<float> t({batch_size, H, W, C});
        gen.generate_batch(t, batch_size);
        train_data.push_back(std::move(t));
    }

    // Test Batches (Fixed)
    Tensor<float> clean_test({batch_size, H, W, C});
    gen.generate_batch(clean_test, batch_size);

    // 2. Model Setup (Zenith-SRIG)
    models::ZenithHierarchicalAE<float> model(C, base_channels, true, "he", true); // use_slm=true

    // 3. Train
    auto start_train = std::chrono::high_resolution_clock::now();
    train_denoising(model, train_data, train_steps, 0.001f);
    auto end_train = std::chrono::high_resolution_clock::now();
    std::cout << "Training Time: " << std::chrono::duration<double>(end_train - start_train).count() << "s" << std::endl;

    // 4. Evaluation
    std::cout << "\n=== Restoration Benchmark Results ===" << std::endl;
    std::mt19937 rng(123);

    auto evaluate = [&](const std::string& name, std::function<void(Tensor<float>&)> noise_fn) {
        Tensor<float> input = clean_test; // Copy
        noise_fn(input);

        float input_mse = compute_mse(clean_test, input);

        auto start = std::chrono::high_resolution_clock::now();
        Tensor<float> output = model.forward(input);
        auto end = std::chrono::high_resolution_clock::now();

        float output_mse = compute_mse(clean_test, output);

        std::cout << "[" << name << "]" << std::endl;
        std::cout << "  Input MSE (Noise Level): " << input_mse << std::endl;
        std::cout << "  Restored MSE:            " << output_mse << std::endl;
        std::cout << "  Noise Reduction Ratio:   " << (input_mse / output_mse) << "x" << std::endl;
        std::cout << "  Inference Time:          " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

        if (output_mse < input_mse) std::cout << "  RESULT: PASS (Restored)" << std::endl;
        else std::cout << "  RESULT: FAIL (Degraded)" << std::endl;
    };

    // Test 1: Gaussian
    evaluate("Gaussian Noise (std=0.3)", [&](Tensor<float>& t) {
        add_gaussian_noise(t, 0.3f, rng);
    });

    // Test 2: Shot Noise
    evaluate("Shot Noise (5% Spikes)", [&](Tensor<float>& t) {
        add_shot_noise(t, 0.05f, 2.0f, rng);
    });

    // Test 3: Blur
    evaluate("Box Blur (3x3)", [&](Tensor<float>& t) {
        apply_blur(t);
    });

    return 0;
}
