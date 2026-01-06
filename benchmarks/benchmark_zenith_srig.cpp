#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace dreidel;

// Helper to compute MSE
template <typename T>
T compute_mse(const Tensor<T>& a, const Tensor<T>& b) {
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

template <typename T>
void add_noise(Tensor<T>& t, float noise_level) {
    static std::mt19937 gen(42);
    std::normal_distribution<T> d(0.0f, noise_level);
    T* ptr = t.data();
    for(size_t i=0; i<t.size(); ++i) {
        ptr[i] += d(gen);
    }
}

int main() {
    size_t H = 64;
    size_t W = 64;
    size_t C = 64;
    size_t batch_size = 4;
    int steps = 500;

    std::cout << "Generating dataset (Sine patterns with noise)..." << std::endl;
    std::vector<Tensor<float>> targets;
    std::vector<Tensor<float>> inputs;

    for(int i=0; i<100; ++i) {
        Tensor<float> target({batch_size, H, W, C});
        Tensor<float> input({batch_size, H, W, C});

        float* tgt_ptr = target.data();
        float* inp_ptr = input.data();

        for(size_t b=0; b<batch_size; ++b) {
            float phase = (float)i * 0.1f + (float)b;
            for(size_t h=0; h<H; ++h) {
                for(size_t w=0; w<W; ++w) {
                    float val = std::sin(h * 0.2f + phase) * std::cos(w * 0.2f);
                    for(size_t c=0; c<C; ++c) {
                        size_t idx = ((b*H + h)*W + w)*C + c;
                        tgt_ptr[idx] = val;
                        inp_ptr[idx] = val;
                    }
                }
            }
        }

        add_noise(input, 0.2f); // Significant noise

        targets.push_back(std::move(target));
        inputs.push_back(std::move(input));
    }

    auto run_benchmark = [&](bool use_slm, const std::string& name) {
        std::cout << "\n=== Training " << name << " ===" << std::endl;
        layers::ZenithBlock<float> model(C, C, 3, C, true, false, false, 1, 1, "he", use_slm);

        optim::SimpleAdam<float> opt(0.001);
        opt.add_parameters(model.parameters(), model.gradients());

        auto start_time = std::chrono::high_resolution_clock::now();
        double total_loss = 0;

        for(int s=0; s<steps; ++s) {
            const auto& in = inputs[s % 100];
            const auto& tgt = targets[s % 100];

            opt.zero_grad();
            auto out = model.forward(in);
            float loss = compute_mse(out, tgt);
            total_loss += loss;

            Tensor<float> grad_out(out.shape());
            float* go = grad_out.data();
            const float* o = out.data();
            const float* t = tgt.data();
            float scale = 2.0f / out.size();
            #pragma omp parallel for
            for(size_t i=0; i<out.size(); ++i) go[i] = (o[i] - t[i]) * scale;

            model.backward(grad_out);
            opt.step();

            if (s % 100 == 0) std::cout << "Step " << s << " Loss: " << loss << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << name << " Time: " << elapsed.count() << "s | Avg Loss: " << total_loss / steps << std::endl;
    };

    run_benchmark(false, "Standard Zenith (No SLM)");
    run_benchmark(true, "Zenith-SRIG (With SLM)");

    return 0;
}
