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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/Layer.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/layers/ZenithVariants.hpp"
#include "../include/dreidel/layers/Dense.hpp"
#include "../include/dreidel/layers/GELU.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"

using namespace dreidel;

// Pointwise Linear Layer (1x1 Conv via Reshape + Dense)
template <typename T>
class PointwiseLinear : public layers::Layer<T> {
public:
    PointwiseLinear(size_t in_channels, size_t out_channels)
        : dense_(in_channels, out_channels) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        // Input: (N, H, W, C)
        auto shape = input.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3];

        // Flatten: (N*H*W, C)
        Tensor<T> flat({N*H*W, C});
        std::copy(input.data(), input.data() + input.size(), flat.data());

        Tensor<T> out_flat = dense_.forward(flat);
        // Reshape back
        size_t C_out = out_flat.shape()[1];
        Tensor<T> output({N, H, W, C_out});
        std::copy(out_flat.data(), out_flat.data() + out_flat.size(), output.data());

        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        auto shape = grad_output.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C_out = shape[3];

        Tensor<T> flat({N*H*W, C_out});
        std::copy(grad_output.data(), grad_output.data() + grad_output.size(), flat.data());

        Tensor<T> grad_in_flat = dense_.backward(flat);

        size_t C_in = grad_in_flat.shape()[1];
        Tensor<T> grad_input({N, H, W, C_in});
        std::copy(grad_in_flat.data(), grad_in_flat.data() + grad_in_flat.size(), grad_input.data());

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override { return dense_.parameters(); }
    std::vector<Tensor<T>*> gradients() override { return dense_.gradients(); }
    std::string name() const override { return "PointwiseLinear"; }

private:
    layers::Dense<T> dense_;
};

// 2D Wavelet Generator (Reused)
template <typename T>
class WaveletGenerator2D {
public:
    static void generate(Tensor<T>& batch, size_t size, size_t channels) {
        size_t batch_size = batch.shape()[0];
        size_t dim = size * size * channels;
        T* ptr = batch.data();

        // Assuming channels=1 for generation logic, repeated or separate?
        // Let's generate 1 channel images.
        // If channels > 1, we can replicate or generate distinct.
        // For Zenith test, let's stick to C=1 input.

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
                // Generate for channel 0
                T* img_ptr = ptr + b * size * size * channels;

                int type = dist_type(gen);

                // Helper to fill one plane
                auto fill_plane = [&](T* p) {
                    if (type < 5) { // Gabor
                        T cx = dist_pos(gen); T cy = dist_pos(gen); T sx = dist_scale(gen);
                        T sy = sx * std::uniform_real_distribution<T>(0.5, 1.5)(gen);
                        T theta = dist_angle(gen);
                        generate_gabor(p, size, cx, cy, sx, sy, theta, dist_freq(gen), dist_phase(gen));
                    } else if (type < 10) { // Mexican Hat
                        T cx = dist_pos(gen); T cy = dist_pos(gen); T s = dist_scale(gen);
                        generate_mexican_hat(p, size, cx, cy, s);
                    } else { // Noise/Texture
                        for(int k=0; k<5; ++k) {
                            T kx = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                            T ky = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                            T phase = dist_phase(gen);
                            for(size_t y=0; y<size; ++y)
                                for(size_t x=0; x<size; ++x)
                                    p[y*size+x] += std::cos(kx*x + ky*y + phase);
                        }
                    }
                };

                fill_plane(img_ptr); // Fill Ch 0
                // Replicate to other channels if any
                for(size_t c=1; c<channels; ++c) {
                    std::copy(img_ptr, img_ptr + size*size, img_ptr + c*size*size);
                }
            }
        }
    }

private:
    static void generate_gabor(T* buffer, size_t size, T cx, T cy, T sx, T sy, T theta, T freq, T psi) {
        T cos_t = std::cos(theta);
        T sin_t = std::sin(theta);
        for(size_t y=0; y<size; ++y) {
            for(size_t x=0; x<size; ++x) {
                T dx = (T)x - cx;
                T dy = (T)y - cy;
                T xp = dx * cos_t + dy * sin_t;
                T yp = -dx * sin_t + dy * cos_t;
                T env = std::exp(-(xp*xp)/(2*sx*sx) - (yp*yp)/(2*sy*sy));
                T carrier = std::cos(2*3.14159 * freq * xp + psi);
                buffer[y*size + x] = env * carrier;
            }
        }
    }

    static void generate_mexican_hat(T* buffer, size_t size, T cx, T cy, T sigma) {
        for(size_t y=0; y<size; ++y) {
            for(size_t x=0; x<size; ++x) {
                T dx = (T)x - cx;
                T dy = (T)y - cy;
                T r2 = dx*dx + dy*dy;
                T s2 = sigma*sigma;
                buffer[y*size + x] = (1.0 - r2/s2) * std::exp(-r2/(2*s2));
            }
        }
    }
};

// Zenith Autoencoder
template <typename T>
class ZenithAutoencoder {
public:
    ZenithAutoencoder(size_t channels = 16) {
        // Encoder
        // Input: 1 -> 16
        stem_ = new PointwiseLinear<T>(1, channels);
        layers_.push_back(stem_);

        // Zenith Block 1 (Use ZenithFloatEyes for training in float)
        z1_ = new layers::ZenithFloatEyes<T>(channels, 3, channels);
        layers_.push_back(z1_);

        act1_ = new layers::GELU<T>();
        layers_.push_back(act1_);

        // Zenith Block 2
        z2_ = new layers::ZenithFloatEyes<T>(channels, 3, channels);
        layers_.push_back(z2_);

        act2_ = new layers::GELU<T>();
        layers_.push_back(act2_);

        // Decoder
        // Zenith Block 3
        z3_ = new layers::ZenithFloatEyes<T>(channels, 3, channels);
        layers_.push_back(z3_);

        act3_ = new layers::GELU<T>();
        layers_.push_back(act3_);

        // Output Head: 16 -> 1
        head_ = new PointwiseLinear<T>(channels, 1);
        layers_.push_back(head_);
    }

    ~ZenithAutoencoder() {
        for(auto l : layers_) delete l;
    }

    Tensor<T> forward(const Tensor<T>& x) {
        Tensor<T> h = stem_->forward(x);
        h = z1_->forward(h);
        h = act1_->forward(h);
        h = z2_->forward(h);
        h = act2_->forward(h); // Latent

        h = z3_->forward(h);
        h = act3_->forward(h);
        return head_->forward(h);
    }

    // Manual Backward chain
    void backward(const Tensor<T>& grad_recon) {
        Tensor<T> g = head_->backward(grad_recon);
        g = act3_->backward(g);
        g = z3_->backward(g);

        g = act2_->backward(g);
        g = z2_->backward(g);
        g = act1_->backward(g);
        g = z1_->backward(g);

        stem_->backward(g);
    }

    std::vector<layers::Layer<T>*> get_layers() { return layers_; }

    std::vector<Tensor<T>*> parameters() {
        std::vector<Tensor<T>*> p;
        for(auto l : layers_) {
            auto lp = l->parameters();
            p.insert(p.end(), lp.begin(), lp.end());
        }
        return p;
    }

    std::vector<Tensor<T>*> gradients() {
        std::vector<Tensor<T>*> g;
        for(auto l : layers_) {
            auto lg = l->gradients();
            g.insert(g.end(), lg.begin(), lg.end());
        }
        return g;
    }

    std::vector<Tensor<T>*> curvatures() {
        // Return dummy or empty if not used by DiagonalNewton correctly?
        // DiagonalNewton checks parameters size match.
        // We can return gradients as dummy curvatures for now if layers don't support it,
        // but typically Layer base returns empty.
        // DiagonalNewton handles empty curvatures by using Identity/SGD approx if needed or fails.
        // We will stick to SGD or simple optimizer if curvature is missing.
        std::vector<Tensor<T>*> c;
        for(auto l : layers_) {
            auto lc = l->curvatures();
            c.insert(c.end(), lc.begin(), lc.end());
        }
        return c;
    }

private:
    std::vector<layers::Layer<T>*> layers_;
    PointwiseLinear<T>* stem_;
    layers::ZenithFloatEyes<T>* z1_;
    layers::GELU<T>* act1_;
    layers::ZenithFloatEyes<T>* z2_;
    layers::GELU<T>* act2_;
    layers::ZenithFloatEyes<T>* z3_;
    layers::GELU<T>* act3_;
    PointwiseLinear<T>* head_;
};

void save_png(const std::string& filename, const Tensor<float>& img_tensor) {
    // img_tensor: (H, W, 1) or (H, W)
    auto shape = img_tensor.shape();
    size_t H = shape[0];
    size_t W = shape[1];
    if (shape.size() == 3) { H = shape[0]; W = shape[1]; }

    std::vector<unsigned char> pixels(H*W);
    const float* data = img_tensor.data();

    float min_v = 1e9, max_v = -1e9;
    for(size_t i=0; i<H*W; ++i) {
        if (data[i] < min_v) min_v = data[i];
        if (data[i] > max_v) max_v = data[i];
    }

    for(size_t i=0; i<H*W; ++i) {
        float norm = (data[i] - min_v) / (max_v - min_v + 1e-6);
        pixels[i] = static_cast<unsigned char>(norm * 255.0f);
    }

    stbi_write_png(filename.c_str(), W, H, 1, pixels.data(), W);
}

int main(int argc, char** argv) {
    size_t epochs = 100;
    size_t batch_size = 4;
    size_t size = 64;
    size_t channels = 16; // Internal channels

    ZenithAutoencoder<float> model(channels);
    optim::DiagonalNewton<float> optimizer(0.01); // Simple LR

    // Register params
    optimizer.add_parameters(model.parameters(), model.gradients(), model.curvatures());

    std::cout << "Training Zenith Autoencoder (64x64 Wavelets)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for(size_t epoch=0; epoch<epochs; ++epoch) {
        Tensor<float> x({batch_size, size, size, 1});
        WaveletGenerator2D<float>::generate(x, size, 1);

        optimizer.zero_grad();

        Tensor<float> y = model.forward(x);

        Tensor<float> diff = y - x;
        float mse = 0;
        const float* d = diff.data();
        for(size_t i=0; i<diff.size(); ++i) mse += d[i]*d[i];
        mse /= diff.size();

        Tensor<float> grad_recon = diff * (2.0f / diff.size());

        model.backward(grad_recon);
        optimizer.step(); // Uses SGD if curvature is empty

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " MSE: " << mse << std::endl;
        }

        if (epoch == epochs - 1) {
            // Save sample
            Tensor<float> x0({size, size});
            Tensor<float> y0({size, size});
            // Extract batch 0
            for(size_t h=0; h<size; ++h)
                for(size_t w=0; w<size; ++w) {
                    x0.data()[h*size+w] = x.data()[((0*size+h)*size+w)*1];
                    y0.data()[h*size+w] = y.data()[((0*size+h)*size+w)*1];
                }
            save_png("zenith_input.png", x0);
            save_png("zenith_output.png", y0);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Training finished in " << elapsed.count() << "s" << std::endl;
    std::cout << "Performance: " << (epochs * batch_size) / elapsed.count() << " images/sec" << std::endl;

    return 0;
}
