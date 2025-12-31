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
#include "../include/dreidel/optim/SGD.hpp"
#include "../include/dreidel/optim/RMSProp.hpp"
#include "../include/dreidel/optim/Adam.hpp"

using namespace dreidel;

// Standard Dense Layer (No curvature for this test)
template <typename T>
class Dense : public layers::Layer<T> {
public:
    Dense(size_t input_dim, size_t output_dim, T bias_init = 0.0)
        : input_dim_(input_dim), output_dim_(output_dim),
          weights_({input_dim, output_dim}), bias_({1, output_dim}),
          grad_weights_({input_dim, output_dim}), grad_bias_({1, output_dim})
    {
        // Xavier Init
        T stddev = std::sqrt(2.0 / (input_dim + output_dim));
        weights_.random(0, stddev);
        bias_.fill(bias_init);
        grad_weights_.fill(0);
        grad_bias_.fill(0);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        input_ = input; // Cache input
        Tensor<T> out = input.matmul(weights_);
        return out + bias_;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        grad_weights_ = input_.transpose().matmul(grad_output);
        grad_bias_ = grad_output.sum(0);
        return grad_output.matmul(weights_.transpose());
    }

    std::vector<Tensor<T>*> parameters() override { return {&weights_, &bias_}; }
    std::vector<Tensor<T>*> gradients() override { return {&grad_weights_, &grad_bias_}; }
    std::string name() const override { return "Dense"; }

private:
    size_t input_dim_, output_dim_;
    Tensor<T> weights_, bias_;
    Tensor<T> grad_weights_, grad_bias_;
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
    std::string name() const override { return "Tanh"; }

private:
    Tensor<T> output_;
};

// Autoencoder Model
template <typename T>
class Autoencoder {
public:
    Autoencoder(size_t input_dim = 64, size_t latent_dim = 16) {
        // Encoder
        layers_.push_back(new Dense<T>(input_dim, 32));
        layers_.push_back(new Tanh<T>());
        layers_.push_back(new Dense<T>(32, latent_dim));
        layers_.push_back(new Tanh<T>());

        // Decoder
        layers_.push_back(new Dense<T>(latent_dim, 32));
        layers_.push_back(new Tanh<T>());
        layers_.push_back(new Dense<T>(32, input_dim));
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

    // Reset weights
    void reset() {
        // Simple hack: Re-initialize. But easier to just recreate model in loop.
    }

private:
    std::vector<layers::Layer<T>*> layers_;
};

// Data Generator (Simplified Mixed Wavelets)
template <typename T>
void generate_data(Tensor<T>& data, size_t batch_size, size_t dim) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist_x(-5.0, 5.0);

    T* ptr = data.data();
    for (size_t i = 0; i < batch_size; ++i) {
        // Just generate a simple Gaussian or Gabor
        T mu = dist_x(gen);
        T s = 1.0;
        for (size_t t = 0; t < dim; ++t) {
             T x = (T(t) - dim/2.0) / (dim/10.0) - mu;
             ptr[i * dim + t] = std::exp(-x*x/2.0);
        }
    }
}

// Training Step
template <typename T>
T train_step(Autoencoder<T>& model, optim::Optimizer<T>& optimizer, const Tensor<T>& x) {
    optimizer.zero_grad();

    Tensor<T> y = model.forward(x);

    // MSE Loss
    size_t batch_size = x.shape()[0];
    size_t dim = x.shape()[1];

    T loss = 0;
    const T* x_ptr = x.data();
    const T* y_ptr = y.data();

    // Create grad tensor
    Tensor<T> grad({batch_size, dim});
    T* g_ptr = grad.data();

    T scale = 2.0 / (batch_size * dim);

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

struct BenchmarkResult {
    std::string name;
    std::vector<float> loss_history;
    double time_seconds;
};

int main() {
    std::cout << "=== Optimizer Benchmark on Autoencoder ===" << std::endl;

    size_t input_dim = 64;
    size_t latent_dim = 16;
    size_t batch_size = 32;
    size_t epochs = 1000;

    std::vector<BenchmarkResult> results;

    // 1. SGD
    {
        std::cout << "Benchmarking SGD..." << std::endl;
        Autoencoder<float> model(input_dim, latent_dim);
        optim::SGD<float> opt(0.01f); // Standard learning rate
        opt.add_parameters(model.parameters(), model.gradients());

        BenchmarkResult res;
        res.name = "SGD";

        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i=0; i<epochs; ++i) {
            Tensor<float> x({batch_size, input_dim});
            generate_data(x, batch_size, input_dim);
            float loss = train_step(model, opt, x);
            if(i % 100 == 0) res.loss_history.push_back(loss);
        }
        auto end = std::chrono::high_resolution_clock::now();
        res.time_seconds = std::chrono::duration<double>(end - start).count();
        results.push_back(res);
        std::cout << "SGD Final Loss: " << res.loss_history.back() << " Time: " << res.time_seconds << "s" << std::endl;
    }

    // 2. RMSProp
    {
        std::cout << "Benchmarking RMSProp..." << std::endl;
        Autoencoder<float> model(input_dim, latent_dim);
        optim::RMSProp<float> opt(0.001f);
        opt.add_parameters(model.parameters(), model.gradients());

        BenchmarkResult res;
        res.name = "RMSProp";

        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i=0; i<epochs; ++i) {
            Tensor<float> x({batch_size, input_dim});
            generate_data(x, batch_size, input_dim);
            float loss = train_step(model, opt, x);
            if(i % 100 == 0) res.loss_history.push_back(loss);
        }
        auto end = std::chrono::high_resolution_clock::now();
        res.time_seconds = std::chrono::duration<double>(end - start).count();
        results.push_back(res);
        std::cout << "RMSProp Final Loss: " << res.loss_history.back() << " Time: " << res.time_seconds << "s" << std::endl;
    }

    // 3. Adam
    {
        std::cout << "Benchmarking Adam..." << std::endl;
        Autoencoder<float> model(input_dim, latent_dim);
        optim::Adam<float> opt(0.001f);
        opt.add_parameters(model.parameters(), model.gradients());

        BenchmarkResult res;
        res.name = "Adam";

        auto start = std::chrono::high_resolution_clock::now();
        for(size_t i=0; i<epochs; ++i) {
            Tensor<float> x({batch_size, input_dim});
            generate_data(x, batch_size, input_dim);
            float loss = train_step(model, opt, x);
            if(i % 100 == 0) res.loss_history.push_back(loss);
        }
        auto end = std::chrono::high_resolution_clock::now();
        res.time_seconds = std::chrono::duration<double>(end - start).count();
        results.push_back(res);
        std::cout << "Adam Final Loss: " << res.loss_history.back() << " Time: " << res.time_seconds << "s" << std::endl;
    }

    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << std::setw(10) << "Optimizer" << std::setw(15) << "Final Loss" << std::setw(15) << "Time (s)" << std::endl;
    std::string winner;
    float min_loss = 1e9;

    for(const auto& res : results) {
        std::cout << std::setw(10) << res.name << std::setw(15) << res.loss_history.back() << std::setw(15) << res.time_seconds << std::endl;
        if(res.loss_history.back() < min_loss) {
            min_loss = res.loss_history.back();
            winner = res.name;
        }
    }
    std::cout << "\nBest Optimizer (Loss): " << winner << std::endl;

    return 0;
}
