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
        grad_weights_ = input_.transpose().matmul(grad_output);
        grad_bias_ = grad_output.sum(0);

        // Curvature Approximation: diag(H) approx sum(X^2)
        Tensor<T> x_sq = input_ * input_;
        Tensor<T> x_sum_sq = x_sq.sum(0);

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

// Slice Layer
template <typename T>
class Slice : public layers::Layer<T> {
public:
    Slice(size_t target_dim) : target_dim_(target_dim) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        input_shape_ = input.shape();
        return input.slice_last_dim(target_dim_);
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Pad back to input shape with zeros
        return grad_output.pad_last_dim(input_shape_.back());
    }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }
    std::vector<Tensor<T>*> curvatures() override { return {}; }
    std::string name() const override { return "Slice"; }

private:
    size_t target_dim_;
    std::vector<size_t> input_shape_;
};

// Pad Layer
template <typename T>
class Pad : public layers::Layer<T> {
public:
    Pad(size_t target_dim) : target_dim_(target_dim) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        input_shape_ = input.shape();
        return input.pad_last_dim(target_dim_);
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Slice back to input shape
        return grad_output.slice_last_dim(input_shape_.back());
    }

    std::vector<Tensor<T>*> parameters() override { return {}; }
    std::vector<Tensor<T>*> gradients() override { return {}; }
    std::vector<Tensor<T>*> curvatures() override { return {}; }
    std::string name() const override { return "Pad"; }

private:
    size_t target_dim_;
    std::vector<size_t> input_shape_;
};

// Autoencoder Model
template <typename T>
class Autoencoder {
public:
    Autoencoder(bool use_custom, size_t input_dim = 128, size_t latent_dim = 16) {
        if (use_custom) {
            std::cout << "Model: Spectral Autoencoder (DSL)" << std::endl;
            // Encoder
            // 128 -> DSL(128) -> Slice(16)
            layers_.push_back(new layers::DeepSpectralLinear<T>(input_dim, 4));
            layers_.push_back(new Tanh<T>());
            layers_.push_back(new Slice<T>(latent_dim));

            // Decoder
            // 16 -> Pad(128) -> DSL(128)
            layers_.push_back(new Pad<T>(input_dim));
            layers_.push_back(new layers::DeepSpectralLinear<T>(input_dim, 4));

            // Final activation? Signals are usually roughly bounded, but let's keep linear out
            // To allow amplitude reconstruction.
        } else {
            std::cout << "Model: Standard Autoencoder (Dense)" << std::endl;
            size_t hidden = 64;
            // Encoder
            layers_.push_back(new DenseWithCurvature<T>(input_dim, hidden));
            layers_.push_back(new Tanh<T>());
            layers_.push_back(new DenseWithCurvature<T>(hidden, latent_dim));
            layers_.push_back(new Tanh<T>()); // Latent activation

            // Decoder
            layers_.push_back(new DenseWithCurvature<T>(latent_dim, hidden));
            layers_.push_back(new Tanh<T>());
            layers_.push_back(new DenseWithCurvature<T>(hidden, input_dim));
        }
    }

    ~Autoencoder() {
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

// Data Generator
template <typename T>
void generate_wavelets(Tensor<T>& data, size_t batch_size, size_t dim) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist_A(0.5, 2.0);
    std::uniform_real_distribution<T> dist_mu(20.0, 100.0); // Center in [20, 100]
    std::uniform_real_distribution<T> dist_sigma(5.0, 20.0);
    std::uniform_real_distribution<T> dist_omega(0.1, 0.5);
    std::uniform_real_distribution<T> dist_phi(0.0, 6.28);

    T* ptr = data.data();

    for (size_t i = 0; i < batch_size; ++i) {
        T A = dist_A(gen);
        T mu = dist_mu(gen);
        T sigma = dist_sigma(gen);
        T omega = dist_omega(gen);
        T phi = dist_phi(gen);

        for (size_t t = 0; t < dim; ++t) {
            T val = static_cast<T>(t);
            T env = std::exp(-std::pow(val - mu, 2) / (2 * sigma * sigma));
            T wave = std::cos(omega * val + phi);
            ptr[i * dim + t] = A * env * wave;
        }
    }
}

// Training Step
template <typename T>
T train_step(Autoencoder<T>& model, optim::DiagonalNewton<T>& optimizer, const Tensor<T>& x) {
    optimizer.zero_grad();

    Tensor<T> y = model.forward(x);

    // Loss = MSE(y, x)
    // Diff = y - x
    // To do y - x, we need operator-.
    // Tensor supports operator- assuming shapes match.
    // Autoencoder output should match input dim.

    // Note: y - x might allocate.
    // Error = y - x
    // Since Tensor.hpp has strict operator- check, and y, x are (B, 128), it works.

    // Manually computing diff to avoid operator- if paranoid, but it should work.
    // Let's use operator- and if it crashes we fix.

    // dL/dy = 2 * (y - x) / B
    size_t batch_size = x.shape()[0];
    size_t dim = x.shape()[1];

    // We can implement loss and gradient manually to be efficient
    T loss = 0;
    const T* x_ptr = x.data();
    const T* y_ptr = y.data();

    // Create grad tensor
    Tensor<T> grad({batch_size, dim});
    grad.fill(0);
    T* g_ptr = grad.data();

    T scale = 2.0 / (batch_size * dim); // Mean over batch and dim? Usually MSE is mean.

    for(size_t i=0; i<x.size(); ++i) {
        T diff = y_ptr[i] - x_ptr[i];
        loss += diff * diff;
        g_ptr[i] = diff * scale;
    }
    loss /= (batch_size * dim);

    model.backward(grad);
    optimizer.step();

    return loss;
}

int main() {
    std::cout << "=== Autoencoder Benchmark (Wavelets) ===" << std::endl;

    size_t input_dim = 128;
    size_t latent_dim = 16;
    size_t batch_size = 64;
    size_t epochs = 1000;

    // Initialize Models
    Autoencoder<float> model_std(false, input_dim, latent_dim);
    Autoencoder<float> model_dsl(true, input_dim, latent_dim);

    optim::DiagonalNewton<float> opt_std(1e-2f);
    optim::DiagonalNewton<float> opt_dsl(1e-4f); // DSL usually needs lower LR

    opt_std.add_parameters(model_std.parameters(), model_std.gradients(), model_std.curvatures());
    opt_dsl.add_parameters(model_dsl.parameters(), model_dsl.gradients(), model_dsl.curvatures());

    // DSL Init fix
    auto params = model_dsl.parameters();
    for(auto* p : params) {
        if(p->size() > 1) {
            float* ptr = p->data();
            for(size_t i=0; i<p->size(); ++i) ptr[i] *= 0.1f;
        }
    }

    double time_std = 0;
    double time_dsl = 0;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        Tensor<float> x({batch_size, input_dim});
        generate_wavelets(x, batch_size, input_dim);

        auto t1 = std::chrono::high_resolution_clock::now();
        float loss_std = train_step(model_std, opt_std, x);
        auto t2 = std::chrono::high_resolution_clock::now();
        time_std += std::chrono::duration<double>(t2 - t1).count();

        auto t3 = std::chrono::high_resolution_clock::now();
        float loss_dsl = train_step(model_dsl, opt_dsl, x);
        auto t4 = std::chrono::high_resolution_clock::now();
        time_dsl += std::chrono::duration<double>(t4 - t3).count();

        if ((epoch+1) % 100 == 0) {
            std::cout << "Epoch " << epoch+1
                      << " | Std: " << loss_std
                      << " | DSL: " << loss_dsl << std::endl;
        }
    }

    std::cout << "Time Std: " << time_std << "s" << std::endl;
    std::cout << "Time DSL: " << time_dsl << "s" << std::endl;

    // Save Example
    std::cout << "Saving results..." << std::endl;
    std::ofstream out("autoencoder_results.csv");
    out << "t,input,std_recon,dsl_recon\n";

    Tensor<float> val_x({1, input_dim});
    generate_wavelets(val_x, 1, input_dim);
    Tensor<float> rec_std = model_std.forward(val_x);
    Tensor<float> rec_dsl = model_dsl.forward(val_x);

    const float* in_ptr = val_x.data();
    const float* s_ptr = rec_std.data();
    const float* d_ptr = rec_dsl.data();

    for(size_t t=0; t<input_dim; ++t) {
        out << t << "," << in_ptr[t] << "," << s_ptr[t] << "," << d_ptr[t] << "\n";
    }
    out.close();

    return 0;
}
