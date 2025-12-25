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
            // Scale hidden dimension with input dimension (N/4) to demonstrate O(N^2) complexity
            size_t hidden = std::max((size_t)64, input_dim / 4);
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

// Mixed Wavelet Generator (20 Types)
template <typename T>
void generate_mixed_wavelets(Tensor<T>& data, size_t batch_size, size_t dim) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Parameters
    std::uniform_real_distribution<T> dist_A(0.5, 1.5);
    std::uniform_real_distribution<T> dist_mu(dim * 0.2, dim * 0.8);
    std::uniform_real_distribution<T> dist_s(2.0, 15.0);
    std::uniform_real_distribution<T> dist_w(0.1, 0.8);
    std::uniform_real_distribution<T> dist_phi(0.0, 6.28);
    std::uniform_int_distribution<int> dist_type(0, 19);

    T* ptr = data.data();

    for (size_t i = 0; i < batch_size; ++i) {
        int type = dist_type(gen);
        T A = dist_A(gen);
        T mu = dist_mu(gen);
        T s = dist_s(gen);
        T w = dist_w(gen);
        T phi = dist_phi(gen);

        // Secondary scale for DoG etc
        T s2 = s * 2.0;

        for (size_t t = 0; t < dim; ++t) {
            T x = static_cast<T>(t) - mu;
            T val = 0;

            switch(type) {
                case 0: // Gabor
                    val = std::cos(w*x + phi) * std::exp(-x*x/(2*s*s));
                    break;
                case 1: // Mexican Hat (Ricker)
                    {
                        T x2 = (x*x)/(s*s);
                        val = (1.0 - x2) * std::exp(-x2/2.0);
                    }
                    break;
                case 2: // Gaussian
                    val = std::exp(-x*x/(2*s*s));
                    break;
                case 3: // Gaussian Derivative 1
                    val = -x * std::exp(-x*x/(2*s*s));
                    break;
                case 4: // Haar-like (Smooth approximation or hard)
                    // Hard Haar
                    if (x >= -s && x < 0) val = 1.0;
                    else if (x >= 0 && x < s) val = -1.0;
                    else val = 0.0;
                    break;
                case 5: // Shannon (Sinc * Cos)
                    if (std::abs(x) < 1e-5) val = std::cos(w*x);
                    else val = (std::sin(x/s)/(x/s)) * std::cos(w*x);
                    break;
                case 6: // Chirp (Linear FM)
                    val = std::cos(w*x + 0.01*x*x) * std::exp(-x*x/(2*s*s));
                    break;
                case 7: // Lorentzian
                    val = 1.0 / (1.0 + x*x/(s*s));
                    break;
                case 8: // Sech
                    val = 1.0 / std::cosh(x/s);
                    break;
                case 9: // Boxcar (Rect)
                    val = (std::abs(x) < s) ? 1.0 : 0.0;
                    break;
                case 10: // Triangular
                    val = std::max((T)0.0, (T)1.0 - std::abs(x)/s);
                    break;
                case 11: // DoG (Difference of Gaussians)
                    val = std::exp(-x*x/(2*s*s)) - 0.5 * std::exp(-x*x/(2*s2*s2));
                    break;
                case 12: // Sinc Squared
                    if (std::abs(x) < 1e-5) val = 1.0;
                    else {
                        T sn = std::sin(x/s)/(x/s);
                        val = sn*sn;
                    }
                    break;
                case 13: // Gammatone (approx, x>0 part centered)
                    {
                        T xt = x + s*2; // Shift to make visible
                        if (xt > 0) val = xt * std::exp(-xt/s) * std::cos(w*xt);
                        else val = 0;
                    }
                    break;
                case 14: // Morlet (Real)
                    val = std::exp(-x*x/(2*s*s)) * std::cos(5.0*x/s);
                    break;
                case 15: // Poisson Wavelet
                    {
                        T xt = x + s; // shift
                        if (xt > 0) val = xt * std::exp(-xt/s);
                        else val = 0;
                    }
                    break;
                case 16: // Beta Wavelet
                    {
                        T xn = x/s;
                        if (std::abs(xn) < 1.0) {
                            // (1-x^2) * cos...
                            val = std::pow(1.0 - xn*xn, 2) * std::cos(w*x);
                        } else val = 0;
                    }
                    break;
                case 17: // Hermite H3
                    {
                        T z = x/s;
                        val = (8*z*z*z - 12*z) * std::exp(-z*z/2);
                    }
                    break;
                case 18: // Sawtooth Pulse
                    if (std::abs(x) < s) val = x/s;
                    else val = 0;
                    break;
                case 19: // Random Walk (Brownian Bridge approx)
                    // Hard to generate pointwise statelessly.
                    // Use a functional approximation: Sum of 3 Cosines
                    val = std::cos(w*x) + 0.5*std::cos(2*w*x) + 0.25*std::cos(3*w*x);
                    val *= std::exp(-x*x/(2*s*s));
                    break;
            }

            ptr[i * dim + t] = A * val;
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

struct ScalingResult {
    size_t dim;
    double time_std;
    double time_dsl;
};

// Helper to write SVG plot
void generate_scaling_svg(const std::vector<ScalingResult>& results) {
    std::ofstream svg("scaling_graph.svg");
    if (!svg.is_open()) return;

    double max_time = 0;
    size_t max_dim = 0;
    for (const auto& r : results) {
        if (r.time_std > max_time) max_time = r.time_std;
        if (r.time_dsl > max_time) max_time = r.time_dsl;
        if (r.dim > max_dim) max_dim = r.dim;
    }

    // Canvas setup
    double width = 800;
    double height = 600;
    double padding = 60;

    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\" />\n";

    // Axes
    svg << "<line x1=\"" << padding << "\" y1=\"" << height - padding << "\" x2=\"" << width - padding << "\" y2=\"" << height - padding << "\" stroke=\"black\" />\n"; // X axis
    svg << "<line x1=\"" << padding << "\" y1=\"" << height - padding << "\" x2=\"" << padding << "\" y2=\"" << padding << "\" stroke=\"black\" />\n"; // Y axis

    // Labels
    svg << "<text x=\"" << width/2 << "\" y=\"" << height - 10 << "\" text-anchor=\"middle\">Dimension (N)</text>\n";
    svg << "<text x=\"20\" y=\"" << height/2 << "\" text-anchor=\"middle\" transform=\"rotate(-90 20 " << height/2 << ")\">Time (s)</text>\n";

    auto map_x = [&](size_t d) { return padding + (double)d / max_dim * (width - 2*padding); };
    auto map_y = [&](double t) { return height - padding - (t / max_time * (height - 2*padding)); };

    // Plot Std (Red)
    svg << "<path d=\"M";
    for (size_t i = 0; i < results.size(); ++i) {
        svg << map_x(results[i].dim) << " " << map_y(results[i].time_std);
        if (i < results.size() - 1) svg << " L ";
    }
    svg << "\" fill=\"none\" stroke=\"red\" stroke-width=\"2\" />\n";

    // Plot DSL (Blue)
    svg << "<path d=\"M";
    for (size_t i = 0; i < results.size(); ++i) {
        svg << map_x(results[i].dim) << " " << map_y(results[i].time_dsl);
        if (i < results.size() - 1) svg << " L ";
    }
    svg << "\" fill=\"none\" stroke=\"blue\" stroke-width=\"2\" />\n";

    // Points and Tooltips/Text
    for (const auto& r : results) {
        double x = map_x(r.dim);
        double y_std = map_y(r.time_std);
        double y_dsl = map_y(r.time_dsl);

        svg << "<circle cx=\"" << x << "\" cy=\"" << y_std << "\" r=\"4\" fill=\"red\" />\n";
        svg << "<circle cx=\"" << x << "\" cy=\"" << y_dsl << "\" r=\"4\" fill=\"blue\" />\n";
    }

    // Legend
    svg << "<rect x=\"" << width - 150 << "\" y=\"50\" width=\"120\" height=\"60\" fill=\"white\" stroke=\"black\" />\n";
    svg << "<line x1=\"" << width - 140 << "\" y1=\"70\" x2=\"" << width - 110 << "\" y2=\"70\" stroke=\"red\" stroke-width=\"2\" />\n";
    svg << "<text x=\"" << width - 100 << "\" y=\"75\">Dense (O(N^2))</text>\n";
    svg << "<line x1=\"" << width - 140 << "\" y1=\"90\" x2=\"" << width - 110 << "\" y2=\"90\" stroke=\"blue\" stroke-width=\"2\" />\n";
    svg << "<text x=\"" << width - 100 << "\" y=\"95\">DSL (O(N log N))</text>\n";

    svg << "</svg>\n";
    svg.close();
    std::cout << "Graph saved to scaling_graph.svg" << std::endl;
}

// Helper to run benchmark for a specific dimension
ScalingResult run_scaling_benchmark(size_t dim, std::ofstream& scaling_csv) {
    size_t input_dim = dim;
    size_t latent_dim = 64;
    size_t batch_size = 64;
    size_t epochs = 100; // Balanced run for scaling test

    std::cout << "\nRunning Benchmark for Input Dim: " << input_dim << std::endl;

    // Initialize Models
    Autoencoder<float> model_std(false, input_dim, latent_dim);
    Autoencoder<float> model_dsl(true, input_dim, latent_dim);

    optim::DiagonalNewton<float> opt_std(1e-2f);
    optim::DiagonalNewton<float> opt_dsl(1e-4f);

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

    // Warmup? No, just run

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        Tensor<float> x({batch_size, input_dim});
        generate_mixed_wavelets(x, batch_size, input_dim);

        auto t1 = std::chrono::high_resolution_clock::now();
        train_step(model_std, opt_std, x);
        auto t2 = std::chrono::high_resolution_clock::now();
        time_std += std::chrono::duration<double>(t2 - t1).count();

        auto t3 = std::chrono::high_resolution_clock::now();
        train_step(model_dsl, opt_dsl, x);
        auto t4 = std::chrono::high_resolution_clock::now();
        time_dsl += std::chrono::duration<double>(t4 - t3).count();
    }

    std::cout << "Time Std: " << time_std << "s" << std::endl;
    std::cout << "Time DSL: " << time_dsl << "s" << std::endl;

    scaling_csv << input_dim << "," << time_std << "," << time_dsl << "\n";

    // If largest dimension, output reconstruction CSV
    if (input_dim == 4096) {
        std::cout << "Saving reconstruction results..." << std::endl;
        std::ofstream out("autoencoder_results.csv");
        out << "t,input,std_recon,dsl_recon\n";

        Tensor<float> val_x({1, input_dim});
        generate_mixed_wavelets(val_x, 1, input_dim);
        Tensor<float> rec_std = model_std.forward(val_x);
        Tensor<float> rec_dsl = model_dsl.forward(val_x);

        const float* in_ptr = val_x.data();
        const float* s_ptr = rec_std.data();
        const float* d_ptr = rec_dsl.data();

        for(size_t t=0; t<input_dim; ++t) {
            out << t << "," << in_ptr[t] << "," << s_ptr[t] << "," << d_ptr[t] << "\n";
        }
        out.close();
    }
    return {dim, time_std, time_dsl};
}

int main() {
    std::cout << "=== Autoencoder Scaling Benchmark (Wavelets) ===" << std::endl;

    std::ofstream scaling_csv("scaling_results.csv");
    scaling_csv << "Dim,Time_Std,Time_DSL\n";

    std::vector<size_t> dims = {512, 1024, 2048, 4096};
    std::vector<ScalingResult> results;

    for (size_t dim : dims) {
        results.push_back(run_scaling_benchmark(dim, scaling_csv));
    }

    scaling_csv.close();

    generate_scaling_svg(results);

    return 0;
}
