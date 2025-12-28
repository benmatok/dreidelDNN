#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <sys/stat.h>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/Layer.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/layers/Conv2D.hpp"
#include "../include/dreidel/layers/Dense.hpp"
#include "../include/dreidel/layers/GELU.hpp"
#include "../include/dreidel/layers/ReLU.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"

using namespace dreidel;

// ---- Utilities ----

void save_png(const std::string& filename, const Tensor<float>& img_tensor) {
    auto shape = img_tensor.shape();
    size_t H = shape[0];
    size_t W = shape[1];
    if (shape.size() >= 3) { H = shape[0]; W = shape[1]; } // Handle (H,W,C) if needed but usually we pass (H,W)
    if (shape.size() == 4) { H = shape[1]; W = shape[2]; } // (1, H, W, 1)

    std::vector<unsigned char> pixels(H*W);
    const float* data = img_tensor.data();

    float min_v = 1e9, max_v = -1e9;
    size_t sz = img_tensor.size();
    for(size_t i=0; i<sz; ++i) {
        if (data[i] < min_v) min_v = data[i];
        if (data[i] > max_v) max_v = data[i];
    }

    for(size_t i=0; i<H*W; ++i) {
        float val = data[i]; // Assuming single channel or flattened
        float norm = (val - min_v) / (max_v - min_v + 1e-6);
        pixels[i] = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, norm * 255.0f)));
    }

    stbi_write_png(filename.c_str(), W, H, 1, pixels.data(), W);
}

// 2D Wavelet Generator
template <typename T>
class WaveletGenerator2D {
public:
    static void generate(Tensor<T>& batch, size_t size, size_t channels) {
        size_t batch_size = batch.shape()[0];
        T* ptr = batch.data();

        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd() + omp_get_thread_num());
            std::uniform_int_distribution<int> dist_type(0, 29);
            T scale_factor = static_cast<T>(size) / 64.0f;
            std::uniform_real_distribution<T> dist_pos(20.0f * scale_factor, 44.0f * scale_factor);
            std::uniform_real_distribution<T> dist_scale(3.0f * scale_factor, 12.0f * scale_factor);
            std::uniform_real_distribution<T> dist_angle(0.0, 3.14159);
            std::uniform_real_distribution<T> dist_freq(0.1, 0.5);
            std::uniform_real_distribution<T> dist_phase(0.0, 6.28);

            #pragma omp for
            for (size_t b = 0; b < batch_size; ++b) {
                T* img_ptr = ptr + b * size * size * channels;
                int type = dist_type(gen);
                auto fill_plane = [&](T* p) {
                    if (type < 5) { // Gabor
                        T cx = dist_pos(gen); T cy = dist_pos(gen); T sx = dist_scale(gen);
                        T sy = sx * std::uniform_real_distribution<T>(0.5, 1.5)(gen);
                        T theta = dist_angle(gen);
                        T cos_t = std::cos(theta), sin_t = std::sin(theta);
                        T freq = dist_freq(gen), psi = dist_phase(gen);
                        for(size_t y=0; y<size; ++y) {
                            for(size_t x=0; x<size; ++x) {
                                T dx = (T)x - cx; T dy = (T)y - cy;
                                T xp = dx * cos_t + dy * sin_t;
                                T yp = -dx * sin_t + dy * cos_t;
                                T env = std::exp(-(xp*xp)/(2*sx*sx) - (yp*yp)/(2*sy*sy));
                                T carrier = std::cos(2*3.14159 * freq * xp + psi);
                                p[y*size + x] = env * carrier;
                            }
                        }
                    } else if (type < 10) { // Mexican Hat
                        T cx = dist_pos(gen); T cy = dist_pos(gen); T s = dist_scale(gen);
                        for(size_t y=0; y<size; ++y) {
                            for(size_t x=0; x<size; ++x) {
                                T dx = (T)x - cx; T dy = (T)y - cy;
                                T r2 = dx*dx + dy*dy; T s2 = s*s;
                                p[y*size + x] = (1.0 - r2/s2) * std::exp(-r2/(2*s2));
                            }
                        }
                    } else { // Noise
                        for(int k=0; k<5; ++k) {
                            T kx = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                            T ky = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                            T phase = dist_phase(gen);
                            for(size_t y=0; y<size; ++y)
                                for(size_t x=0; x<size; ++x)
                                    p[y*size+x] += std::cos(kx*x + ky*y + phase);
                        }
                        // Normalize
                         for(size_t i=0; i<size*size; ++i) p[i] *= 0.2;
                    }
                };
                fill_plane(img_ptr);
                for(size_t c=1; c<channels; ++c) std::copy(img_ptr, img_ptr + size*size, img_ptr + c*size*size);
            }
        }
    }
};

// ---- Architectures ----

// Pointwise Linear for Stem/Head
template <typename T>
class PointwiseLinear : public layers::Layer<T> {
public:
    PointwiseLinear(size_t in_channels, size_t out_channels) : dense_(in_channels, out_channels) {}
    Tensor<T> forward(const Tensor<T>& input) override {
        auto shape = input.shape();
        Tensor<T> flat({shape[0]*shape[1]*shape[2], shape[3]});
        std::copy(input.data(), input.data() + input.size(), flat.data());
        Tensor<T> out_flat = dense_.forward(flat);
        Tensor<T> output({shape[0], shape[1], shape[2], out_flat.shape()[1]});
        std::copy(out_flat.data(), out_flat.data() + out_flat.size(), output.data());
        return output;
    }
    Tensor<T> backward(const Tensor<T>& grad_output) override {
        auto shape = grad_output.shape();
        Tensor<T> flat({shape[0]*shape[1]*shape[2], shape[3]});
        std::copy(grad_output.data(), grad_output.data() + grad_output.size(), flat.data());
        Tensor<T> grad_in_flat = dense_.backward(flat);
        Tensor<T> grad_input({shape[0], shape[1], shape[2], grad_in_flat.shape()[1]});
        std::copy(grad_in_flat.data(), grad_in_flat.data() + grad_in_flat.size(), grad_input.data());
        return grad_input;
    }
    std::vector<Tensor<T>*> parameters() override { return dense_.parameters(); }
    std::vector<Tensor<T>*> gradients() override { return dense_.gradients(); }
    std::string name() const override { return "PointwiseLinear"; }
private:
    layers::Dense<T> dense_;
};

// Standard Conv Autoencoder
template <typename T>
class ConvAutoencoder {
public:
    ConvAutoencoder(size_t channels = 16) {
        // Encoder
        // Input (1) -> (C)
        layers_.push_back(new layers::Conv2D<T>(1, channels, 3, 1, 1)); // Padding 1 to keep size
        layers_.push_back(new layers::ReLU<T>());
        layers_.push_back(new layers::Conv2D<T>(channels, channels, 3, 1, 1));
        layers_.push_back(new layers::ReLU<T>());
        // Decoder
        layers_.push_back(new layers::Conv2D<T>(channels, channels, 3, 1, 1));
        layers_.push_back(new layers::ReLU<T>());
        layers_.push_back(new layers::Conv2D<T>(channels, 1, 3, 1, 1)); // Out (1)
    }

    ~ConvAutoencoder() { for(auto l : layers_) delete l; }

    Tensor<T> forward(const Tensor<T>& x) {
        Tensor<T> h = x;
        for(auto l : layers_) h = l->forward(h);
        return h;
    }

    void backward(const Tensor<T>& grad) {
        Tensor<T> g = grad;
        for(int i=layers_.size()-1; i>=0; --i) g = layers_[i]->backward(g);
    }

    std::vector<Tensor<T>*> parameters() {
        std::vector<Tensor<T>*> p;
        for(auto l : layers_) { auto lp = l->parameters(); p.insert(p.end(), lp.begin(), lp.end()); }
        return p;
    }
    std::vector<Tensor<T>*> gradients() {
        std::vector<Tensor<T>*> g;
        for(auto l : layers_) { auto lg = l->gradients(); g.insert(g.end(), lg.begin(), lg.end()); }
        return g;
    }
    std::vector<Tensor<T>*> curvatures() { return {}; } // Conv doesn't support yet

private:
    std::vector<layers::Layer<T>*> layers_;
};

// Zenith Autoencoder
template <typename T>
class ZenithAutoencoder {
public:
    ZenithAutoencoder(size_t channels = 16) {
        // Encoder
        layers_.push_back(new PointwiseLinear<T>(1, channels)); // Stem

        layers_.push_back(new layers::ZenithBlock<T>(channels, 3, channels));
        layers_.push_back(new layers::GELU<T>());

        layers_.push_back(new layers::ZenithBlock<T>(channels, 3, channels));
        layers_.push_back(new layers::GELU<T>());

        // Decoder
        layers_.push_back(new layers::ZenithBlock<T>(channels, 3, channels));
        layers_.push_back(new layers::GELU<T>());

        layers_.push_back(new PointwiseLinear<T>(channels, 1)); // Head
    }

    ~ZenithAutoencoder() { for(auto l : layers_) delete l; }

    Tensor<T> forward(const Tensor<T>& x) {
        Tensor<T> h = x;
        for(auto l : layers_) h = l->forward(h);
        return h;
    }

    void backward(const Tensor<T>& grad) {
        Tensor<T> g = grad;
        for(int i=layers_.size()-1; i>=0; --i) g = layers_[i]->backward(g);
    }

    std::vector<Tensor<T>*> parameters() {
        std::vector<Tensor<T>*> p;
        for(auto l : layers_) { auto lp = l->parameters(); p.insert(p.end(), lp.begin(), lp.end()); }
        return p;
    }
    std::vector<Tensor<T>*> gradients() {
        std::vector<Tensor<T>*> g;
        for(auto l : layers_) { auto lg = l->gradients(); g.insert(g.end(), lg.begin(), lg.end()); }
        return g;
    }
    std::vector<Tensor<T>*> curvatures() { return {}; }

private:
    std::vector<layers::Layer<T>*> layers_;
};

// ---- Benchmarking Logic ----

template<typename Model>
void run_benchmark(std::string name, size_t epochs, size_t batch_size, size_t channels) {
    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Benchmarking: " << name << std::endl;
    std::cout << "Note: For this benchmark, Conv2D is run serially to compare algorithmic efficiency" << std::endl;
    std::cout << "      against the serial Zenith Block implementation." << std::endl;

    Model model(channels);
    optim::DiagonalNewton<float> optimizer(0.01);
    optimizer.add_parameters(model.parameters(), model.gradients(), model.curvatures());

    // Count parameters
    size_t param_count = 0;
    for(auto p : model.parameters()) param_count += p->size();
    std::cout << "Parameter Count: " << param_count << std::endl;

    size_t size = 64;
    Tensor<float> x({batch_size, size, size, 1});
    WaveletGenerator2D<float>::generate(x, size, 1);

    auto start_time = std::chrono::high_resolution_clock::now();
    float final_mse = 0;

    for(size_t epoch=0; epoch<epochs; ++epoch) {
        optimizer.zero_grad();
        Tensor<float> y = model.forward(x);

        // MSE
        Tensor<float> diff = y - x;
        float mse = 0;
        const float* d = diff.data();
        for(size_t i=0; i<diff.size(); ++i) mse += d[i]*d[i];
        mse /= diff.size();
        final_mse = mse;

        Tensor<float> grad = diff * (2.0f / diff.size());
        model.backward(grad);
        optimizer.step();

        if(epoch % (epochs/5) == 0) {
            std::cout << "  Epoch " << epoch << " MSE: " << mse << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Results for " << name << ":" << std::endl;
    std::cout << "  Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "  TPS (Train steps/sec): " << epochs / elapsed.count() << std::endl;
    std::cout << "  Final MSE: " << final_mse << std::endl;

    // Save output image
    Tensor<float> y = model.forward(x);
    // Extract first image
    Tensor<float> out_img({size, size});
    Tensor<float> in_img({size, size});
    for(size_t i=0; i<size*size; ++i) {
        out_img.data()[i] = y.data()[i * 1]; // Batch 0, Channel 0
        in_img.data()[i] = x.data()[i * 1];
    }
    save_png(name + "_output.png", out_img);
    if (name.find("Conv") != std::string::npos) save_png("input_target.png", in_img); // Save input once
}

int main() {
    size_t epochs = 50; // Short run for validation
    size_t batch_size = 4;
    size_t channels = 16;

    std::cout << "Starting Zenith vs Conv2D Comparison..." << std::endl;
    std::cout << "Image Size: 64x64, Batch: " << batch_size << ", Channels: " << channels << std::endl;

    // We disable OpenMP at runtime via environment or just ensure single thread via simple way?
    // Conv2D implementation currently uses '#pragma omp parallel for'.
    // To ensure fair serial comparison, we can set OMP_NUM_THREADS=1 from shell when running.

    run_benchmark<ConvAutoencoder<float>>("Conv2D_Baseline", epochs, batch_size, channels);
    run_benchmark<ZenithAutoencoder<float>>("Zenith_Model", epochs, batch_size, channels);

    return 0;
}
