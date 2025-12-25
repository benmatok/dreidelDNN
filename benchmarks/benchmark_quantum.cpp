#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <chrono>

// Include Dreidel headers
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/Layer.hpp"
#include "../include/dreidel/layers/DeepSpectralLinear.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"
#include "../include/dreidel/algo/WHT.hpp"

// Use C++17 special functions if available, or implement Hermite manually
#if __cplusplus >= 201703L
#include <cmath> // std::hermite
#else
// Simple recurrence for Hermite polynomials H_n(x)
double hermite_poly(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return 2.0 * x;
    double h0 = 1.0;
    double h1 = 2.0 * x;
    double h2 = 0.0;
    for (int i = 2; i <= n; ++i) {
        h2 = 2.0 * x * h1 - 2.0 * (i - 1) * h0;
        h0 = h1;
        h1 = h2;
    }
    return h2;
}
#endif

using namespace dreidel;

// Custom Dense Layer with Curvature Support (Diagonal Approximation)
template <typename T>
class DenseWithCurvature : public layers::Layer<T> {
public:
    DenseWithCurvature(size_t input_dim, size_t output_dim, T bias_init = 0.0)
        : input_dim_(input_dim), output_dim_(output_dim),
          weights_({input_dim, output_dim}), bias_({1, output_dim}),
          grad_weights_({input_dim, output_dim}), grad_bias_({1, output_dim}),
          curv_weights_({input_dim, output_dim}), curv_bias_({1, output_dim})
    {
        // Xavier Init
        T stddev = std::sqrt(2.0 / (input_dim + output_dim));
        weights_.random(0, stddev);
        bias_.fill(bias_init);
        grad_weights_.fill(0);
        grad_bias_.fill(0);
        curv_weights_.fill(0);
        curv_bias_.fill(0);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        input_ = input; // Cache input
        Tensor<T> out = input.matmul(weights_);
        return out + bias_;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // dL/dW = X^T * G
        grad_weights_ = input_.transpose().matmul(grad_output);
        // dL/db = sum(G, 0)
        grad_bias_ = grad_output.sum(0);

        // Curvature Approximation: diag(H) approx sum(X^2)
        Tensor<T> x_sq = input_ * input_;
        Tensor<T> x_sum_sq = x_sq.sum(0); // (1, In)

        T* c_ptr = curv_weights_.data();
        const T* x_ptr = x_sum_sq.data();

        for (size_t r = 0; r < input_dim_; ++r) {
            T val = x_ptr[r];
            for (size_t c = 0; c < output_dim_; ++c) {
                c_ptr[r * output_dim_ + c] += val;
            }
        }

        T batch_size = static_cast<T>(input_.shape()[0]);
        T* cb_ptr = curv_bias_.data();
        for (size_t i=0; i<curv_bias_.size(); ++i) {
            cb_ptr[i] += batch_size;
        }

        return grad_output.matmul(weights_.transpose());
    }

    std::vector<Tensor<T>*> parameters() override { return {&weights_, &bias_}; }
    std::vector<Tensor<T>*> gradients() override { return {&grad_weights_, &grad_bias_}; }
    std::vector<Tensor<T>*> curvatures() override { return {&curv_weights_, &curv_bias_}; }

    std::string name() const override { return "DenseWithCurvature"; }

    void zero_grad() {
        grad_weights_.fill(0);
        grad_bias_.fill(0);
        curv_weights_.fill(0);
        curv_bias_.fill(0);
    }

private:
    size_t input_dim_, output_dim_;
    Tensor<T> weights_, bias_;
    Tensor<T> grad_weights_, grad_bias_;
    Tensor<T> curv_weights_, curv_bias_;
    Tensor<T> input_;
};

// Tanh Activation
template <typename T>
class Tanh : public layers::Layer<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) override {
        output_ = input.apply([](T x) { return std::tanh(x); });
        return output_;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> grad_input = grad_output * output_.apply([](T y) { return 1.0 - y * y; });
        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }
    std::vector<Tensor<T>*> curvatures() override { return {}; }
    std::string name() const override { return "Tanh"; }

private:
    Tensor<T> output_;
};

// Quantum PINN Model
template <typename T>
class QuantumPINN {
public:
    QuantumPINN(bool use_custom_layer, size_t hidden_dim = 128) {
        if (use_custom_layer) {
            layers_.push_back(new layers::DeepSpectralLinear<T>(hidden_dim, 4));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new layers::DeepSpectralLinear<T>(hidden_dim, 4));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new layers::DeepSpectralLinear<T>(hidden_dim, 4));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new DenseWithCurvature<T>(hidden_dim, 1, 0.1));
        } else {
            layers_.push_back(new DenseWithCurvature<T>(1, hidden_dim));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new DenseWithCurvature<T>(hidden_dim, hidden_dim));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new DenseWithCurvature<T>(hidden_dim, hidden_dim));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new DenseWithCurvature<T>(hidden_dim, 1, 0.1));
        }
    }

    ~QuantumPINN() {
        for (auto l : layers_) delete l;
    }

    Tensor<T> forward(const Tensor<T>& x) {
        Tensor<T> out = x;
        for (auto l : layers_) {
            out = l->forward(out);
        }
        return out;
    }

    void backward(const Tensor<T>& grad_output) {
        Tensor<T> grad = grad_output;
        for (int i = layers_.size() - 1; i >= 0; --i) {
            grad = layers_[i]->backward(grad);
        }
    }

    std::vector<Tensor<T>*> parameters() {
        std::vector<Tensor<T>*> params;
        for (auto l : layers_) {
            auto p = l->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }

    std::vector<Tensor<T>*> gradients() {
        std::vector<Tensor<T>*> grads;
        for (auto l : layers_) {
            auto g = l->gradients();
            grads.insert(grads.end(), g.begin(), g.end());
        }
        return grads;
    }

    std::vector<Tensor<T>*> curvatures() {
        std::vector<Tensor<T>*> curvs;
        for (auto l : layers_) {
            auto c = l->curvatures();
            curvs.insert(curvs.end(), c.begin(), c.end());
        }
        return curvs;
    }

private:
    std::vector<layers::Layer<T>*> layers_;
};

// Analytical Solution
double get_analytical_psi(double x, int n) {
    double log_fact = 0;
    for(int i=1; i<=n; ++i) log_fact += std::log((double)i);
    double log_Cn = -0.5 * (n * std::log(2.0) + log_fact + 0.5 * std::log(M_PI));
    double Cn = std::exp(log_Cn);

#if __cplusplus >= 201703L
    double Hn = std::hermite(n, x);
#else
    double Hn = hermite_poly(n, x);
#endif

    return Cn * std::exp(-0.5 * x * x) * Hn;
}

// Single Training Step
// Note: We implement a manual training step because PINNs require the Laplacian (psi''),
// which involves second-order derivatives w.r.t Input.
// Standard backprop only gives gradients w.r.t Weights.
// We use Finite Differences to approximate psi'' and the "Concat Trick"
// to backpropagate through the 3-point stencil efficiently in one pass.
template <typename T>
T train_step(QuantumPINN<T>& model, optim::DiagonalNewton<T>& optimizer,
             const Tensor<T>& x, const Tensor<T>& x_l, const Tensor<T>& x_r,
             const std::vector<T>& x_data, T epsilon, T E, T domain_L) {

    optimizer.zero_grad();
    size_t batch_size = x.shape()[0];

    // Forward
    Tensor<T> psi_c = model.forward(x);
    Tensor<T> psi_l = model.forward(x_l);
    Tensor<T> psi_r = model.forward(x_r);

    // Finite Diff
    Tensor<T> term1 = psi_c * 2.0;
    Tensor<T> diff1 = psi_r - term1;
    Tensor<T> sum1 = diff1 + psi_l;
    Tensor<T> d2psi = sum1 * (1.0 / (epsilon * epsilon));

    // Residual
    Tensor<T> V = x.apply([](T v){ return 0.5 * v * v; });
    Tensor<T> V_minus_E = V.apply([E](T v){ return v - E; });
    Tensor<T> potential_term = V_minus_E * psi_c;
    Tensor<T> res = d2psi * (-0.5) + potential_term;

    // Loss Calculation
    T norm_sq = 0;
    const T* psi_ptr = psi_c.data();
    for(size_t i=0; i<batch_size; ++i) norm_sq += psi_ptr[i] * psi_ptr[i];
    norm_sq = (norm_sq / batch_size) * domain_L;

    // Gradients via Concat trick
    // 1. Concat inputs
    std::vector<T> all_data;
    all_data.reserve(3*batch_size);
    const T* l_ptr = x_l.data();
    const T* c_ptr = x.data();
    const T* r_ptr = x_r.data();
    for(size_t i=0; i<batch_size; ++i) all_data.push_back(l_ptr[i]);
    for(size_t i=0; i<batch_size; ++i) all_data.push_back(c_ptr[i]);
    for(size_t i=0; i<batch_size; ++i) all_data.push_back(r_ptr[i]);

    Tensor<T> x_all({3*batch_size, 1}, all_data);

    // 2. Forward Full Batch
    Tensor<T> psi_all = model.forward(x_all);
    const T* p_ptr = psi_all.data();

    // 3. Compute Gradients per sample
    std::vector<T> g_all(3*batch_size);
    T total_loss = 0;

    // Norm gradient factor
    T norm_grad_factor = 2.0 * (norm_sq - 1.0) * 2.0 * domain_L / batch_size;

    for(size_t i=0; i<batch_size; ++i) {
        T pl = p_ptr[i];
        T pc = p_ptr[batch_size + i];
        T pr = p_ptr[2*batch_size + i];

        T d2 = (pr - 2*pc + pl) / (epsilon * epsilon);
        T xv = x_data[i];
        T pot = 0.5 * xv * xv - E;
        T r = -0.5 * d2 + pot * pc;

        total_loss += r * r;

        // dL/dr = 2r / B
        T gr = 2.0 * r / batch_size;

        g_all[i] = gr * (-0.5 / (epsilon * epsilon)); // Left
        g_all[batch_size + i] = gr * (1.0/(epsilon * epsilon) + pot) + norm_grad_factor * pc; // Center
        g_all[2*batch_size + i] = gr * (-0.5 / (epsilon * epsilon)); // Right
    }

    total_loss = total_loss / batch_size;
    total_loss += (norm_sq - 1.0)*(norm_sq - 1.0);

    Tensor<T> grad_all({3*batch_size, 1}, g_all);

    // 4. Backward
    model.backward(grad_all);

    // 5. Optimizer
    optimizer.step();

    return total_loss;
}

// Main Training Function
void train_benchmark_all() {
    int quantum_n = 1;
    float E = static_cast<float>(quantum_n) + 0.5f;
    int epochs = 2000;
    size_t batch_size = 500;
    float epsilon = 1e-3f;
    float domain_L = 20.0f;

    std::cout << "Starting Benchmark (n=" << quantum_n << ", E=" << E << ", " << epochs << " epochs)..." << std::endl;

    // Models
    std::cout << "Initializing Models..." << std::endl;
    QuantumPINN<float> model_std(false);
    QuantumPINN<float> model_dsl(true);

    // Optimizers
    optim::DiagonalNewton<float> opt_std(1e-3f);
    optim::DiagonalNewton<float> opt_dsl(1e-5f); // Slower learning rate for DSL

    opt_std.add_parameters(model_std.parameters(), model_std.gradients(), model_std.curvatures());
    opt_dsl.add_parameters(model_dsl.parameters(), model_dsl.gradients(), model_dsl.curvatures());

    // DSL Initialization Fix
    auto params_dsl = model_dsl.parameters();
    for (auto* p : params_dsl) {
        if (p->size() > 1) {
            float* ptr = p->data();
            for(size_t i=0; i<p->size(); ++i) ptr[i] *= 0.1f;
        }
    }

    // Random Generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    // Debug shapes once
    {
        std::vector<float> x_data(batch_size);
        Tensor<float> x({batch_size, 1}, x_data);
        std::cout << "Input Shape: [" << x.shape()[0] << ", " << x.shape()[1] << "]" << std::endl;
        Tensor<float> out = model_dsl.forward(x);
        std::cout << "DSL Output Shape: [" << out.shape()[0] << ", " << out.shape()[1] << "]" << std::endl;
    }

    double time_std = 0;
    double time_dsl = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Generate Batch (Shared)
        std::vector<float> x_data(batch_size);
        for(size_t i=0; i<batch_size; ++i) x_data[i] = dist(gen);
        Tensor<float> x({batch_size, 1}, x_data);

        Tensor<float> x_l = x.apply([&](float v){ return v - epsilon; });
        Tensor<float> x_r = x.apply([&](float v){ return v + epsilon; });

        // Standard MLP Step
        auto t1 = std::chrono::high_resolution_clock::now();
        float loss_std = train_step(model_std, opt_std, x, x_l, x_r, x_data, epsilon, E, domain_L);
        auto t2 = std::chrono::high_resolution_clock::now();
        time_std += std::chrono::duration<double>(t2 - t1).count();

        // DSL Step
        auto t3 = std::chrono::high_resolution_clock::now();
        float loss_dsl = train_step(model_dsl, opt_dsl, x, x_l, x_r, x_data, epsilon, E, domain_L);
        auto t4 = std::chrono::high_resolution_clock::now();
        time_dsl += std::chrono::duration<double>(t4 - t3).count();

        if (std::isnan(loss_dsl)) {
             if ((epoch+1) % 100 == 0) std::cout << "Epoch " << epoch+1 << " | Std Loss: " << loss_std << " | DSL Loss: NaN (Diverged)" << std::endl;
        } else {
             if ((epoch+1) % 100 == 0) std::cout << "Epoch " << epoch+1 << " | Std Loss: " << loss_std << " | DSL Loss: " << loss_dsl << std::endl;
        }
    }

    std::cout << "Training Complete." << std::endl;
    std::cout << "Standard MLP Time: " << time_std << " s (" << (epochs * batch_size / time_std) << " samples/s)" << std::endl;
    std::cout << "DeepSpectralLinear Time: " << time_dsl << " s (" << (epochs * batch_size / time_dsl) << " samples/s)" << std::endl;

    // Evaluate
    std::cout << "Evaluating..." << std::endl;
    std::ofstream out("quantum_benchmark_results.csv");
    out << "x,psi_true,psi_std,psi_dsl\n";

    std::vector<float> x_eval;
    for(float val = -10.0f; val <= 10.0f; val += 0.1f) x_eval.push_back(val);

    Tensor<float> x_t({x_eval.size(), 1}, x_eval);
    Tensor<float> psi_std = model_std.forward(x_t);
    Tensor<float> psi_dsl = model_dsl.forward(x_t);

    const float* ptr_std = psi_std.data();
    const float* ptr_dsl = psi_dsl.data();

    for(size_t i=0; i<x_eval.size(); ++i) {
        float xv = x_eval[i];
        float tv = (float)get_analytical_psi(xv, quantum_n);
        out << xv << "," << tv << "," << ptr_std[i] << "," << ptr_dsl[i] << "\n";
    }
    out.close();
}

int main() {
    train_benchmark_all();
    return 0;
}
