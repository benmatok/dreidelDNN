#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>

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
    DenseWithCurvature(size_t input_dim, size_t output_dim)
        : input_dim_(input_dim), output_dim_(output_dim),
          weights_({input_dim, output_dim}), bias_({1, output_dim}),
          grad_weights_({input_dim, output_dim}), grad_bias_({1, output_dim}),
          curv_weights_({input_dim, output_dim}), curv_bias_({1, output_dim})
    {
        // Xavier Init
        T stddev = std::sqrt(2.0 / (input_dim + output_dim));
        weights_.random(0, stddev);
        bias_.fill(0);
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
        // For Dense layer with Least Squares / CrossEntropy, the Gauss-Newton approx
        // for weights W_ij is related to sum_batch (X_bi^2).
        // This is a rough diagonal approximation often used.
        // C_W = sum(X^2, axis=0)^T broadcasted?
        // Actually, diagonal element for W_ij depends on X_i.
        // H_{ij, ij} = sum_k (X_ki^2 * (activation_deriv)^2 ...)
        // We approximate simply by input energy for the benchmark to enable Newton steps.

        // Sum of squares of input along batch
        Tensor<T> x_sq = input_ * input_;
        Tensor<T> x_sum_sq = x_sq.sum(0); // (1, In)

        // We replicate this for each output neuron?
        // W is (In, Out). Each column j uses same input x.
        // So curvature is same for all j.
        // curv_weights_ (In, Out)

        // Create a tensor of shape (In, Out) by repeating x_sum_sq
        // Since we don't have broadcast-copy, we do it manually or loop.

        T* c_ptr = curv_weights_.data();
        const T* x_ptr = x_sum_sq.data();

        for (size_t r = 0; r < input_dim_; ++r) {
            T val = x_ptr[r]; // + epsilon done in optimizer
            for (size_t c = 0; c < output_dim_; ++c) {
                c_ptr[r * output_dim_ + c] += val; // Accumulate
            }
        }

        // Bias curvature = Batch Size (sum of 1s)
        T batch_size = static_cast<T>(input_.shape()[0]);
        T* cb_ptr = curv_bias_.data();
        for (size_t i=0; i<curv_bias_.size(); ++i) {
            cb_ptr[i] += batch_size;
        }

        // dL/dX = G * W^T
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
        // d/dx = (1 - y^2) * grad
        Tensor<T> grad_input = grad_output * output_.apply([](T y) { return 1.0 - y * y; });
        return grad_input;
    }

    // Tanh has no params, returns empty
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
            std::cout << "Using DeepSpectralLinear" << std::endl;
            // Input -> Hidden
            // Input is 1D. DSL requires Power of 2 usually, or padding.
            // 1 -> 128 (Power of 2)
            layers_.push_back(new layers::DeepSpectralLinear<T>(hidden_dim, 4));
            layers_.push_back(new Tanh<T>());

            // Hidden -> Hidden
            layers_.push_back(new layers::DeepSpectralLinear<T>(hidden_dim, 4));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new layers::DeepSpectralLinear<T>(hidden_dim, 4));
            layers_.push_back(new Tanh<T>());

            // Hidden -> Output
            // 128 -> 1
            // DSL outputs same dim, so we need a slice or a final Dense?
            // "Custom Layers against Standard MLPs".
            // Usually the final projection is Dense.
            // Or we use DSL and slice.
            // Let's use Dense for final projection to 1.
            layers_.push_back(new DenseWithCurvature<T>(hidden_dim, 1));
        } else {
            std::cout << "Using Standard MLP (Dense)" << std::endl;
            layers_.push_back(new DenseWithCurvature<T>(1, hidden_dim));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new DenseWithCurvature<T>(hidden_dim, hidden_dim));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new DenseWithCurvature<T>(hidden_dim, hidden_dim));
            layers_.push_back(new Tanh<T>());

            layers_.push_back(new DenseWithCurvature<T>(hidden_dim, 1));
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

    // Manual Backprop for Accumulation
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
double get_analytical_psi(double x, int n=30) {
    // Cn = 1 / sqrt(2^n * n! * sqrt(pi))
    // We compute log first
    double log_fact = 0;
    for(int i=1; i<=n; ++i) log_fact += std::log((double)i);
    double log_Cn = -0.5 * (n * std::log(2.0) + log_fact + 0.5 * std::log(M_PI));
    double Cn = std::exp(log_Cn);

    // Hermite
#if __cplusplus >= 201703L
    double Hn = std::hermite(n, x);
#else
    double Hn = hermite_poly(n, x);
#endif

    return Cn * std::exp(-0.5 * x * x) * Hn;
}

// Training Loop
template <typename T>
void train_benchmark(bool use_custom, int epochs=2000) {
    QuantumPINN<T> model(use_custom);
    optim::DiagonalNewton<T> optimizer(1e-3); // Learning rate

    optimizer.add_parameters(model.parameters(), model.gradients(), model.curvatures());

    size_t batch_size = 500;
    T epsilon = 1e-3; // Finite Difference step
    T E = 30.5;

    std::cout << "Starting Training (" << epochs << " epochs)..." << std::endl;

    // Domain [-10, 10]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-10.0, 10.0);

    std::vector<T> loss_history;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();

        // 1. Generate Batch
        std::vector<T> x_data(batch_size);
        for(size_t i=0; i<batch_size; ++i) x_data[i] = dist(gen);

        Tensor<T> x({batch_size, 1}, x_data);

        // 2. Finite Differences for Laplacian
        // x_l, x_c, x_r
        Tensor<T> x_l = x.apply([&](T v){ return v - epsilon; });
        Tensor<T> x_r = x.apply([&](T v){ return v + epsilon; });

        Tensor<T> psi_c = model.forward(x);
        Tensor<T> psi_l = model.forward(x_l);
        Tensor<T> psi_r = model.forward(x_r);

        if (psi_r.shape() != psi_c.shape()) {
            std::cerr << "Shape Mismatch! psi_r: ";
            for(auto s : psi_r.shape()) std::cerr << s << " ";
            std::cerr << " psi_c: ";
            for(auto s : psi_c.shape()) std::cerr << s << " ";
            std::cerr << std::endl;
        }

        // psi'' approx (psi_r - 2psi_c + psi_l) / eps^2
        std::cerr << "Computing d2psi..." << std::endl;
        Tensor<T> term1 = psi_c * 2.0;
        Tensor<T> diff1 = psi_r - term1;
        Tensor<T> sum1 = diff1 + psi_l;
        Tensor<T> d2psi = sum1 * (1.0 / (epsilon * epsilon));

        // Residual: -0.5 psi'' + (0.5 x^2 - E) psi
        Tensor<T> V = x.apply([](T v){ return 0.5 * v * v; });
        Tensor<T> V_minus_E = V.apply([E](T v){ return v - E; });
        Tensor<T> potential_term = V_minus_E * psi_c;

        // Recalculate residual properly
        Tensor<T> res = d2psi * (-0.5) + potential_term;

        // Loss = mean(res^2)
        // Also Normalization penalty: mean(psi^2) * L approx 1
        T domain_L = 20.0;
        T norm_sq = 0;
        const T* psi_ptr = psi_c.data();
        for(size_t i=0; i<batch_size; ++i) norm_sq += psi_ptr[i] * psi_ptr[i];
        norm_sq = (norm_sq / batch_size) * domain_L;

        T loss_norm_val = std::pow(norm_sq - 1.0, 2);

        // Backprop
        // dL/d(res) = 2 * res / batch_size
        Tensor<T> grad_res = res * (2.0 / batch_size);

        // dL/d(psi_c) from PDE part:
        // res = -0.5/eps^2 (r - 2c + l) + (V-E) c
        // d(res)/dc = 1.0/eps^2 + (V-E)
        // grad_c_pde = grad_res * (1/eps^2 + V-E)

        Tensor<T> grad_c = grad_res * V_minus_E;
        grad_c = grad_c + grad_res * (1.0/(epsilon*epsilon));

        // dL/d(psi_l) = grad_res * (-0.5 / eps^2)
        Tensor<T> grad_l = grad_res * (-0.5 / (epsilon*epsilon));

        // dL/d(psi_r) = grad_res * (-0.5 / eps^2)
        Tensor<T> grad_r = grad_res * (-0.5 / (epsilon*epsilon));

        // Add Normalization Gradient to center
        // L_norm = (S - 1)^2, S = mean(c^2)*L
        // dL_norm/dS = 2(S-1)
        // dS/dc = 2c * L / B
        // dL_norm/dc = 2(S-1) * 2c * L / B
        T grad_norm_scale = 2.0 * (norm_sq - 1.0) * 2.0 * domain_L / batch_size;
        Tensor<T> grad_c_norm = psi_c * grad_norm_scale;

        // Combine gradients for center
        grad_c = grad_c + grad_c_norm; // Actually this applies 0.1 weight if we want?
        // Let's stick to 1.0 weight for now or add small weight.

        // We need to run backward 3 times?
        // Yes, but we need to accumulate gradients in the optimizer/layers.
        // Does Layer::backward accumulate?
        // Dense::backward overwrites grad_weights_! "grad_weights_ = ..."
        // This is a problem.
        // We cannot simply call backward 3 times sequentially if it overwrites.
        // We must modify layers to accumulate, or manually sum.
        // Or, since Dense backprop is linear in grad_output,
        // we can compute the effective grad_output on x_c?
        // Wait, the backward pass depends on the stored input "input_".
        // Forward was called 3 times: l, c, r.
        // The last call was 'r'. So layers currently hold state for 'r'.
        // So we can only backprop for 'r' immediately.
        // For 'c' and 'l', the state is lost!
        // This is the limitation of the stateful Layer design.

        // WORKAROUND:
        // We must do Forward-Backward interleaved for each point, or save/restore state.
        // But we want to do it in one batch update.
        // Since we are doing FD, we effectively treat the network as processing 3 independent batches.
        // Or one big batch of size 3*B.

        // Let's concat inputs!
        // X_all = [x_l; x_c; x_r] (Size 3B)
        // Forward(X_all) -> Psi_all
        // Split Psi_all -> Psi_l, Psi_c, Psi_r
        // Compute Gradients dL/dPsi_all
        // Backward(dL/dPsi_all)
        // This works perfectly and handles state correctly.

        // Concatenate
        std::vector<T> all_data;
        all_data.reserve(3*batch_size);
        const T* l_ptr = x_l.data();
        const T* c_ptr = x.data();
        const T* r_ptr = x_r.data();
        for(size_t i=0; i<batch_size; ++i) all_data.push_back(l_ptr[i]);
        for(size_t i=0; i<batch_size; ++i) all_data.push_back(c_ptr[i]);
        for(size_t i=0; i<batch_size; ++i) all_data.push_back(r_ptr[i]);

        Tensor<T> x_all({3*batch_size, 1}, all_data);

        // Forward
        Tensor<T> psi_all = model.forward(x_all);

        // Split output to compute loss/grads
        // Since we don't have split, we access data pointer.
        // Output is (3B, 1).
        const T* p_ptr = psi_all.data();
        std::vector<T> g_all(3*batch_size);

        T total_loss = 0;

        for(size_t i=0; i<batch_size; ++i) {
            T pl = p_ptr[i];
            T pc = p_ptr[batch_size + i];
            T pr = p_ptr[2*batch_size + i];

            T d2 = (pr - 2*pc + pl) / (epsilon * epsilon);
            T xv = x_data[i];
            T pot = 0.5 * xv * xv - E;
            T r = -0.5 * d2 + pot * pc;

            total_loss += r * r;

            // Gradients
            // dL/dr = 2r / B
            T gr = 2.0 * r / batch_size;

            T g_pl = gr * (-0.5 / (epsilon*epsilon));
            T g_pr = gr * (-0.5 / (epsilon*epsilon));
            T g_pc = gr * (0.5 * 2.0 / (epsilon*epsilon) + pot);

            // Norm penalty (only on center?)
            // We used center for norm.
            // dL_norm/dpc
            // We calculated grad_norm_scale * pc earlier.
            // Recalc local contribution:
            // S = sum(pc^2)/B * L. Loss=(S-1)^2.
            // dLoss/dpc_i = 2(S-1) * (2*pc_i/B * L)
            // We need S first.
        }

        // Compute Norm S
        T sum_sq_c = 0;
        for(size_t i=0; i<batch_size; ++i) {
            T pc = p_ptr[batch_size + i];
            sum_sq_c += pc*pc;
        }
        T S = (sum_sq_c / batch_size) * domain_L;
        total_loss = total_loss / batch_size; // Mean PDE loss
        total_loss += (S - 1.0)*(S - 1.0); // + Norm loss

        T norm_grad_factor = 2.0 * (S - 1.0) * 2.0 * domain_L / batch_size;

        // Fill gradients
        for(size_t i=0; i<batch_size; ++i) {
            T pl = p_ptr[i];
            T pc = p_ptr[batch_size + i];
            T pr = p_ptr[2*batch_size + i];

            T d2 = (pr - 2*pc + pl) / (epsilon * epsilon);
            T xv = x_data[i];
            T pot = 0.5 * xv * xv - E;
            T r = -0.5 * d2 + pot * pc;

            T gr = 2.0 * r / batch_size;

            g_all[i] = gr * (-0.5 / (epsilon*epsilon)); // Left
            g_all[batch_size + i] = gr * (1.0/(epsilon*epsilon) + pot) + norm_grad_factor * pc; // Center
            g_all[2*batch_size + i] = gr * (-0.5 / (epsilon*epsilon)); // Right
        }

        Tensor<T> grad_all({3*batch_size, 1}, g_all);

        // Backward
        model.backward(grad_all);

        // Optimizer Step
        optimizer.step();

        if ((epoch+1) % 100 == 0) {
            std::cout << "Epoch " << epoch+1 << " | Loss: " << total_loss << std::endl;
        }
    }

    // Evaluate
    std::cout << "Evaluating..." << std::endl;
    std::ofstream out("quantum_benchmark_results.csv", std::ios::app);
    if (use_custom) out << "Type,x,psi_pred,psi_true\n";

    // Evaluation grid
    std::vector<T> x_eval;
    for(T val = -10.0; val <= 10.0; val += 0.1) x_eval.push_back(val);

    Tensor<T> x_t({x_eval.size(), 1}, x_eval);
    Tensor<T> psi_pred = model.forward(x_t);

    const T* pred_ptr = psi_pred.data();
    for(size_t i=0; i<x_eval.size(); ++i) {
        T xv = x_eval[i];
        T pv = pred_ptr[i];
        T tv = get_analytical_psi(xv, 30);

        // Note: Sign might be flipped.
        // We handle sign flip in post-processing or assume user plots abs/squared.
        // Or we check alignment.

        out << (use_custom ? "DeepSpectralLinear" : "StandardMLP") << ","
            << xv << "," << pv << "," << tv << "\n";
    }
    out.close();
}

int main() {
    std::cout << "=== Quantum PINN Benchmark (C++) ===" << std::endl;

    // Clear CSV
    std::ofstream out("quantum_benchmark_results.csv");
    out.close();

    // Train Standard
    train_benchmark<float>(false, 2000);

    // Train Custom
    train_benchmark<float>(true, 2000);

    return 0;
}
