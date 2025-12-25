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

// Helper to save SVG grid
void save_svg_grid(const std::string& filename, const std::vector<std::vector<float>>& images, int rows, int cols, int size) {
    std::ofstream out(filename);

    int scale = 2; // Scale up for visibility
    int padding = 10;
    int img_w = size * scale;
    int img_h = size * scale;
    int total_width = cols * (img_w + padding) + padding;
    int total_height = rows * (img_h + padding) + padding;

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << total_width << "\" height=\"" << total_height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\" />\n";

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int img_idx = r * cols + c;
            if (img_idx >= images.size()) continue;

            const auto& img = images[img_idx];

            // Find min/max for normalization
            float min_val = 1e9, max_val = -1e9;
            for (float v : img) {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
            if (max_val == min_val) max_val = min_val + 1e-6;

            int start_x = padding + c * (img_w + padding);
            int start_y = padding + r * (img_h + padding);

            // Draw pixels
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    float val = img[y * size + x];
                    float norm = (val - min_val) / (max_val - min_val);
                    int gray = static_cast<int>(std::min(255.0f, std::max(0.0f, norm * 255.0f)));

                    out << "<rect x=\"" << start_x + x*scale << "\" y=\"" << start_y + y*scale
                        << "\" width=\"" << scale << "\" height=\"" << scale
                        << "\" fill=\"rgb(" << gray << "," << gray << "," << gray << ")\" />\n";
                }
            }
        }
    }
    out << "</svg>\n";
    out.close();
}

// Training Loop
template <typename T>
void train(size_t epochs, size_t batches_per_epoch, size_t batch_size) {
    size_t dim = 4096; // 64x64
    WaveletAutoencoder<T> model(dim);
    optim::DiagonalNewton<T> optimizer(1e-4); // Low LR for spectral

    // Register params
    optimizer.add_parameters(model.parameters(), model.gradients(), model.curvatures());

    std::cout << "Starting training: " << epochs << " epochs, " << batches_per_epoch << " batches/epoch, batch " << batch_size << std::endl;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        T epoch_mse = 0;
        T epoch_reg = 0;

        for (size_t b = 0; b < batches_per_epoch; ++b) {
            Tensor<T> x({batch_size, dim});
            WaveletGenerator2D<T>::generate(x, batch_size);

            optimizer.zero_grad();

            Tensor<T> z; // Latent
            Tensor<T> y = model.forward_train(x, z);

            // 1. Reconstruction Loss (MSE)
            Tensor<T> diff = y + (x * -1.0);

            T mse = 0;
            {
                 const T* d_ptr = diff.data();
                 for(size_t k=0; k<diff.size(); ++k) mse += d_ptr[k]*d_ptr[k];
                 mse /= diff.size();
            }

            Tensor<T> grad_recon = diff * (2.0 / diff.size());

            // 2. Binary Regularization
            Tensor<T> grad_reg_z({batch_size, dim});
            T reg_loss = 0;

            const T* z_ptr = z.data();
            T* gz_ptr = grad_reg_z.data();
            T reg_lambda = 0.1;

            for(size_t k=0; k<z.size(); ++k) {
                T val = z_ptr[k];
                T term = val*val - 1.0;
                reg_loss += term*term;

                gz_ptr[k] = reg_lambda * (4.0 * val * term) / z.size();
            }
            reg_loss /= z.size();
            reg_loss *= reg_lambda;

            model.backward_with_reg(grad_recon, grad_reg_z);

            optimizer.step();

            epoch_mse += mse;
            epoch_reg += reg_loss;
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ": MSE=" << epoch_mse/batches_per_epoch << " Reg=" << epoch_reg/batches_per_epoch << std::endl;
        }
    }

    // --- Testing and Visualization ---
    std::cout << "Testing Compression..." << std::endl;
    size_t test_size = 8;
    Tensor<T> test_x({test_size, dim});
    WaveletGenerator2D<T>::generate(test_x, test_size);

    Tensor<T> test_z = model.encode(test_x);
    Tensor<T> test_rec = model.decode(test_z); // Float reconstruction

    // Quantize z using sign function
    Tensor<T> test_z_q = test_z;
    T* zq_ptr = test_z_q.data();
    for(size_t i=0; i<test_z_q.size(); ++i) {
        // Sign function: -1 for x < 0, 1 for x >= 0
        zq_ptr[i] = (zq_ptr[i] >= 0) ? 1.0f : -1.0f;
    }

    Tensor<T> test_rec_q = model.decode(test_z_q); // Quantized reconstruction

    // Compute Quantized MSE
    Tensor<T> diff_q = test_rec_q + (test_x * -1.0);
    T mse_q = 0;
    const T* dq_ptr = diff_q.data();
    for(size_t i=0; i<diff_q.size(); ++i) mse_q += dq_ptr[i]*dq_ptr[i];
    mse_q /= diff_q.size();

    std::cout << "Quantized MSE: " << mse_q << std::endl;

    // Visualize
    std::vector<std::vector<float>> vis_images;
    const T* x_ptr = test_x.data();
    const T* z_ptr = test_z.data();
    const T* r_ptr = test_rec.data();
    const T* rq_ptr = test_rec_q.data();

    // We visualize 8 samples, 4 rows: Original, Latent, Rec Float, Rec Quant
    // 1. Originals
    for(int i=0; i<8; ++i) {
        std::vector<float> img(dim);
        for(int j=0; j<dim; ++j) img[j] = x_ptr[i*dim + j];
        vis_images.push_back(img);
    }
    // 2. Latents (reshaped to 64x64)
    for(int i=0; i<8; ++i) {
        std::vector<float> img(dim);
        for(int j=0; j<dim; ++j) img[j] = z_ptr[i*dim + j];
        vis_images.push_back(img);
    }
    // 3. Rec Float
    for(int i=0; i<8; ++i) {
        std::vector<float> img(dim);
        for(int j=0; j<dim; ++j) img[j] = r_ptr[i*dim + j];
        vis_images.push_back(img);
    }
    // 4. Rec Quant
    for(int i=0; i<8; ++i) {
        std::vector<float> img(dim);
        for(int j=0; j<dim; ++j) img[j] = rq_ptr[i*dim + j];
        vis_images.push_back(img);
    }

    save_svg_grid("reconstruction_grid.svg", vis_images, 4, 8, 64);
    std::cout << "Saved visualization to reconstruction_grid.svg" << std::endl;
}

int main() {
    train<float>(1000, 1, 16); // 1000 epochs, 1 batch/epoch (1000 total batches)
    return 0;
}
