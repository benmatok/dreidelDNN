#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/Layer.hpp"
#include "../include/dreidel/layers/DeepSpectralLinear.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"

using namespace dreidel;

// Tanh Activation (copied from benchmark_autoencoder.cpp)
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

// 2D Wavelet Generator
// Generates 64x64 patches flattened to 4096
template <typename T>
class WaveletGenerator2D {
public:
    static void generate(Tensor<T>& batch, size_t batch_size) {
        // Output shape: (batch_size, 4096)
        // We assume batch is already allocated with correct size
        T* ptr = batch.data();
        size_t dim = 4096;
        size_t size = 64; // 64x64

        #pragma omp parallel
        {
            // Thread-local RNG
            std::random_device rd;
            std::mt19937 gen(rd() + omp_get_thread_num());

            std::uniform_int_distribution<int> dist_type(0, 19);
            std::uniform_real_distribution<T> dist_pos(20.0, 44.0);
            std::uniform_real_distribution<T> dist_scale(3.0, 12.0);
            std::uniform_real_distribution<T> dist_angle(0.0, 3.14159);
            std::uniform_real_distribution<T> dist_freq(0.1, 0.5);
            std::uniform_real_distribution<T> dist_phase(0.0, 6.28);

            #pragma omp for
            for (size_t b = 0; b < batch_size; ++b) {
                int type = dist_type(gen);
                size_t offset = b * dim;

                if (type < 5) {
                    // Gabor
                    T cx = dist_pos(gen);
                    T cy = dist_pos(gen);
                    T sx = dist_scale(gen);
                    T sy = sx * std::uniform_real_distribution<T>(0.5, 1.5)(gen);
                    T theta = dist_angle(gen);
                    T freq = dist_freq(gen);
                    T psi = dist_phase(gen);
                    generate_gabor(ptr + offset, size, cx, cy, sx, sy, theta, freq, psi);
                } else if (type < 10) {
                    // Curvelet-like (High Aspect Ratio)
                    T cx = dist_pos(gen);
                    T cy = dist_pos(gen);
                    T sx = dist_scale(gen) * 1.5; // Longer
                    T sy = sx * 0.1; // Very thin
                    T theta = dist_angle(gen);
                    T freq = dist_freq(gen);
                    T psi = dist_phase(gen);
                    generate_gabor(ptr + offset, size, cx, cy, sx, sy, theta, freq, psi);
                } else if (type < 15) {
                    // Mexican Hat 2D (Isotropic or Anisotropic)
                    T cx = dist_pos(gen);
                    T cy = dist_pos(gen);
                    T s = dist_scale(gen);
                    generate_mexican_hat(ptr + offset, size, cx, cy, s);
                } else {
                    // High Complexity: Sum of 3 random wavelets
                    // Clear buffer first
                    std::fill(ptr + offset, ptr + offset + dim, 0);

                    // Add 3 components
                    for(int k=0; k<3; ++k) {
                        // Random subtype
                        int subtype = std::uniform_int_distribution<int>(0, 2)(gen);
                        std::vector<T> temp(dim);
                        T cx = dist_pos(gen);
                        T cy = dist_pos(gen);
                        T theta = dist_angle(gen);

                        if (subtype == 0) {
                             T sx = dist_scale(gen);
                             generate_gabor(temp.data(), size, cx, cy, sx, sx, theta, dist_freq(gen), dist_phase(gen));
                        } else if (subtype == 1) {
                             T sx = dist_scale(gen) * 1.5;
                             generate_gabor(temp.data(), size, cx, cy, sx, sx*0.1, theta, dist_freq(gen), dist_phase(gen));
                        } else {
                             generate_mexican_hat(temp.data(), size, cx, cy, dist_scale(gen));
                        }

                        // Accumulate
                        for(size_t i=0; i<dim; ++i) ptr[offset+i] += temp[i];
                    }

                    // Normalize roughly
                    for(size_t i=0; i<dim; ++i) ptr[offset+i] /= 1.5;
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

// Wavelet Autoencoder
template <typename T>
class WaveletAutoencoder {
public:
    WaveletAutoencoder(size_t dim) {
        // Encoder
        // DSL(dim) -> Tanh
        // Latent is dim-sized but we constrain it in loss
        encoder_dsl_ = new layers::DeepSpectralLinear<T>(dim, 4);
        encoder_act_ = new Tanh<T>();

        // Decoder
        // DSL(dim) -> Identity (Linear reconstruction)
        // Usually decoder mirrors encoder.
        decoder_dsl_ = new layers::DeepSpectralLinear<T>(dim, 4);

        layers_.push_back(encoder_dsl_);
        layers_.push_back(encoder_act_);
        layers_.push_back(decoder_dsl_);
    }

    ~WaveletAutoencoder() {
        for(auto l : layers_) delete l;
    }

    // Returns latent code z
    Tensor<T> encode(const Tensor<T>& x) {
        Tensor<T> z = encoder_dsl_->forward(x);
        z = encoder_act_->forward(z);
        return z;
    }

    // Returns reconstruction
    Tensor<T> decode(const Tensor<T>& z) {
        return decoder_dsl_->forward(z);
    }

    Tensor<T> forward(const Tensor<T>& x) {
        return decode(encode(x));
    }

    // We need a custom forward pass for training to access z for regularization
    Tensor<T> forward_train(const Tensor<T>& x, Tensor<T>& z_out) {
        Tensor<T> z = encoder_dsl_->forward(x);
        z = encoder_act_->forward(z);
        z_out = z; // Copy z for loss calculation
        return decoder_dsl_->forward(z);
    }

    void backward(const Tensor<T>& grad_output) {
         // This only backprops from decoder output
         // If we have regularization on z, we need to inject gradient at z.
         // See train_step.
         // Standard backward chain
         Tensor<T> grad = grad_output;
         for (int i = layers_.size() - 1; i >= 0; --i) {
             grad = layers_[i]->backward(grad);
         }
    }

    // Custom backward for regularization
    // grad_recon: gradient of MSE w.r.t output
    // grad_reg_z: gradient of Reg w.r.t z
    void backward_with_reg(const Tensor<T>& grad_recon, const Tensor<T>& grad_reg_z) {
        // 1. Backprop through Decoder
        Tensor<T> grad_z_from_recon = decoder_dsl_->backward(grad_recon);

        // 2. Add regularization gradient
        // Total dL/dz = dL_recon/dz + dL_reg/dz
        // grad_reg_z is dL_reg/dz

        // Ensure shapes match
        if (grad_z_from_recon.shape() != grad_reg_z.shape()) {
             std::cerr << "Shape mismatch in backward_with_reg" << std::endl;
             exit(1);
        }

        Tensor<T> total_grad_z = grad_z_from_recon + grad_reg_z;

        // 3. Backprop through Encoder Activation
        Tensor<T> grad_enc_out = encoder_act_->backward(total_grad_z);

        // 4. Backprop through Encoder DSL
        encoder_dsl_->backward(grad_enc_out);
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
    layers::DeepSpectralLinear<T>* encoder_dsl_;
    layers::Layer<T>* encoder_act_;
    layers::DeepSpectralLinear<T>* decoder_dsl_;
};

// Training Loop
template <typename T>
void train(size_t iterations, size_t batch_size) {
    size_t dim = 4096; // 64x64
    WaveletAutoencoder<T> model(dim);
    optim::DiagonalNewton<T> optimizer(1e-4); // Low LR for spectral

    // Register params
    optimizer.add_parameters(model.parameters(), model.gradients(), model.curvatures());

    // Init scales to small values to avoid explosion
    auto params = model.parameters();
    for(auto* p : params) {
        if(p->size() > 1) { // Likely scale vector
             // DeepSpectralLinear initializes to 1/sqrt(dim).
             // 1/64 = 0.015. This is fine.
             // Benchmark used 0.1 explicitly. Let's trust default or set if needed.
        }
    }

    std::cout << "Starting training: " << iterations << " iterations, batch " << batch_size << std::endl;

    for (size_t i = 0; i < iterations; ++i) {
        Tensor<T> x({batch_size, dim});
        WaveletGenerator2D<T>::generate(x, batch_size);

        optimizer.zero_grad();

        Tensor<T> z; // Latent
        Tensor<T> y = model.forward_train(x, z);

        // 1. Reconstruction Loss (MSE)
        // L_recon = mean((y - x)^2)
        // dL/dy = 2(y-x)/N

        Tensor<T> diff = y + (x * -1.0); // y - x

        // Calc MSE for reporting
        T mse = 0;
        {
             const T* d_ptr = diff.data();
             for(size_t k=0; k<diff.size(); ++k) mse += d_ptr[k]*d_ptr[k];
             mse /= diff.size();
        }

        // Grad Recon
        Tensor<T> grad_recon = diff * (2.0 / diff.size());

        // 2. Binary Regularization
        // L_reg = mean((z^2 - 1)^2)
        // dL/dz = 2 * (z^2 - 1) * 2z = 4z(z^2 - 1)

        // Need to calculate dL/dz
        Tensor<T> grad_reg_z({batch_size, dim});
        T reg_loss = 0;

        const T* z_ptr = z.data();
        T* gz_ptr = grad_reg_z.data();
        T reg_lambda = 0.1; // Weight for regularization

        for(size_t k=0; k<z.size(); ++k) {
            T val = z_ptr[k];
            T term = val*val - 1.0;
            reg_loss += term*term;

            // Derivative: 4 * z * (z^2 - 1)
            // Normalize by size? Yes, usually mean.
            gz_ptr[k] = reg_lambda * (4.0 * val * term) / z.size();
        }
        reg_loss /= z.size();
        reg_loss *= reg_lambda;

        // Backward
        model.backward_with_reg(grad_recon, grad_reg_z);

        optimizer.step();

        if (i % 10 == 0) {
            std::cout << "Iter " << i << ": MSE=" << mse << " Reg=" << reg_loss << " Total=" << mse+reg_loss << std::endl;
        }
    }

    // Save a sample
    std::cout << "Saving reconstruction sample..." << std::endl;
    Tensor<T> val_x({1, dim});
    WaveletGenerator2D<T>::generate(val_x, 1);
    Tensor<T> val_z;
    Tensor<T> val_y = model.forward_train(val_x, val_z);

    std::ofstream out("wavelet_reconstruction.csv");
    out << "idx,original,latent,reconstructed\n";
    const T* orig = val_x.data();
    const T* lat = val_z.data();
    const T* rec = val_y.data();
    for(size_t k=0; k<dim; ++k) {
        out << k << "," << orig[k] << "," << lat[k] << "," << rec[k] << "\n";
    }
    out.close();
    std::cout << "Saved to wavelet_reconstruction.csv" << std::endl;
}

int main() {
    train<float>(100, 16); // 100 iters, batch 16
    return 0;
}
