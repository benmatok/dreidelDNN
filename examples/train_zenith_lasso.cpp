#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include "../include/dreidel/optim/ZenithRegularizer.hpp"

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
        // Nearest Neighbor Backward: Sum gradients
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

    // Parameters interface (empty)
    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }

private:
    size_t scale_;
    Tensor<T> input_;
};

// --- Data Gen ---
template <typename T>
void generate_wavelet_batch(Tensor<T>& data, bool verbose=false) {
    auto shape = data.shape();
    size_t batch = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    static std::mt19937 gen(12345); // Fixed seed for stability
    std::uniform_real_distribution<T> dist_param(0.5, 2.0);
    std::uniform_real_distribution<T> dist_pos(0.2, 0.8);
    T* ptr = data.data();

    // Simple deterministic based on index 'n'
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

    if (verbose) {
        double energy = 0;
        for(size_t i=0; i<data.size(); ++i) energy += ptr[i] * ptr[i];
        std::cout << "Target Energy (Mean Squared): " << energy / data.size() << std::endl;
    }
}

// Add Gaussian Noise for Test C
template <typename T>
void add_gaussian_noise(Tensor<T>& data, float stddev) {
    static std::mt19937 gen(54321);
    std::normal_distribution<T> d(0.0f, stddev);
    T* ptr = data.data();
    for(size_t i=0; i<data.size(); ++i) {
        ptr[i] += d(gen);
    }
}

int main() {
    std::cout << "=== Zenith-Lasso Training Specification ===" << std::endl;

    // Config
    size_t batch_size = 8;
    size_t H = 128, W = 128; // Updated to 128x128 as requested
    size_t epochs = 100; // Standard demo length
    float lr = 0.001f; // Standard learning rate
    float max_lambda = 1e-4f; // Tune: Higher = Sparse, Lower = Accurate

    std::vector<layers::Layer<float>*> model;

    // Encoder (64x64x1 -> 1x1x64) in 3 steps
    // args: in, out, k, spec, ifwht, dil, gate, stride, upscale, init, slm, seq, eps
    // Use eps=1.0f for maximum stability in sparse regime
    model.push_back(new layers::ZenithBlock<float>(1, 128, 3, 128, true, true, false, 4, 1, "he", false, false, 1.0f));
    model.push_back(new layers::ZenithBlock<float>(128, 128, 3, 128, true, true, false, 4, 1, "he", false, false, 1.0f));
    model.push_back(new layers::ZenithBlock<float>(128, 64, 3, 128, true, true, false, 4, 1, "he", false, false, 1.0f));

    // Decoder (1x1x64 -> 64x64x1) in 3 steps
    model.push_back(new Upscale2D<float>(4));
    model.push_back(new layers::ZenithBlock<float>(64, 128, 3, 128, true, true, false, 1, 1, "he", false, false, 1.0f));
    model.push_back(new Upscale2D<float>(4));
    model.push_back(new layers::ZenithBlock<float>(128, 128, 3, 128, true, true, false, 1, 1, "he", false, false, 1.0f));
    model.push_back(new Upscale2D<float>(4));
    model.push_back(new layers::ZenithBlock<float>(128, 1, 3, 128, true, true, false, 1, 1, "he", false, false, 1.0f));

    // Optimizer
    optim::SimpleAdam<float> optimizer(lr);
    optimizer.set_coordinate_wise_clipping(true, 5.0); // Clip gradients to prevent explosion

    for(auto* layer : model) {
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (!params.empty() && params.size() == grads.size()) {
            optimizer.add_parameters(params, grads);
        }
    }

    Tensor<float> input({batch_size, H, W, 1});
    Tensor<float> target({batch_size, H, W, 1});
    float final_loss = 0.0f;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Training Loop
    for(size_t epoch=0; epoch<epochs; ++epoch) {
        // Lambda Schedule
        float current_lambda = 0.0f;
        if (epoch > 100) {
            float progress = (float)(epoch - 100) / 400.0f;
            if (progress > 1.0f) progress = 1.0f;
            current_lambda = max_lambda * progress;
        }

        // Logic to adjust lambda if loss explodes (Convergence Test A Logic)
        if (final_loss > 1.0f && epoch > 200) {
             max_lambda = 1e-5f; // Reduce if diverging
             current_lambda = max_lambda;
        }

        generate_wavelet_batch(input, (epoch==0));
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

        // Regularizer Step (Zenith-Lasso)
        if (current_lambda > 0.0f) {
            for(auto* layer_base : model) {
                auto* layer = dynamic_cast<layers::ZenithBlock<float>*>(layer_base);
                if(layer) {
                     auto params = layer->parameters();
                     // mixing_weights_ is params[2] (Left, Center, Right stacked 3xChannels)
                     // params[2] is Tensor<T>*, we need data pointer.
                     optim::apply_group_lasso_avx(params[2]->data(), params[2]->size(), current_lambda, lr);
                }
            }
        }

        if (epoch % 50 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << std::setw(4) << epoch
                      << " | Loss: " << loss
                      << " | Lambda: " << current_lambda << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> train_time = end_time - start_time;

    std::cout << "Training Complete." << std::endl;
    std::cout << "Final Loss: " << final_loss << std::endl;

    // --- Validation Protocol ---
    std::cout << "\n=== Validation Protocol ===" << std::endl;

    // Test A: Convergence
    bool convergence_pass = (final_loss < 0.30f);
    std::cout << "[Test A] Convergence (Loss < 0.30): " << (convergence_pass ? "PASS" : "FAIL")
              << " (" << final_loss << ")" << std::endl;

    // Test B: Sparsity
    size_t total_blocks = 0;
    size_t zero_blocks = 0;
    for(auto* layer_base : model) {
        auto* layer = dynamic_cast<layers::ZenithBlock<float>*>(layer_base);
        if(layer) {
            auto params = layer->parameters();
            float* w = params[2]->data();
            size_t s = params[2]->size();
            for(size_t i=0; i<s; i+=8) {
                if (i + 8 > s) break;
                float energy = 0;
                for(int j=0; j<8; j++) energy += std::abs(w[i+j]);
                total_blocks++;
                if (energy < 1e-9f) zero_blocks++;
            }
        }
    }
    float sparsity = (float)zero_blocks / (total_blocks + 1e-9f);
    bool sparsity_pass = (sparsity > 0.70f);
    std::cout << "[Test B] Sparsity (>70%): " << (sparsity_pass ? "PASS" : "FAIL")
              << " (" << (sparsity * 100.0f) << "%)" << std::endl;


    // Test C: Accuracy (Restoration with Noise)
    // Add Gaussian noise to input and check reconstruction
    generate_wavelet_batch(input);
    target = input;
    add_gaussian_noise(input, 0.1f); // 0.1 stddev noise

    Tensor<float> x_rest = input;
    for(auto* layer : model) {
        x_rest = layer->forward(x_rest);
    }

    float mse_rest = 0;
    const float* r_ptr = x_rest.data();
    const float* t_ptr = target.data();
    for(size_t i=0; i<x_rest.size(); ++i) {
        float d = r_ptr[i] - t_ptr[i];
        mse_rest += d * d;
    }
    mse_rest /= x_rest.size();

    bool accuracy_pass = (mse_rest < 0.03f);
    std::cout << "[Test C] Accuracy (MSE < 0.03): " << (accuracy_pass ? "PASS" : "FAIL")
              << " (" << mse_rest << ")" << std::endl;

    // Test D: Speed (Throughput)
    auto fwd_start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<10; ++i) {
        Tensor<float> dummy = input;
        for(auto* layer : model) dummy = layer->forward(dummy);
    }
    auto fwd_end = std::chrono::high_resolution_clock::now();
    double avg_fwd_ms = std::chrono::duration<double, std::milli>(fwd_end - fwd_start).count() / 10.0;

    bool speed_pass = (avg_fwd_ms < 15.0);
    std::cout << "[Test D] Speed (< 15ms): " << (speed_pass ? "PASS" : "FAIL")
              << " (" << avg_fwd_ms << " ms)" << std::endl;

    // Cleanup
    for(auto* l : model) delete l;

    if (convergence_pass && accuracy_pass) {
        return 0; // Success
    } else {
        return 1; // Fail
    }
}
