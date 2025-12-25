#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/Layer.hpp"
#include "../include/dreidel/layers/DeepSpectralLinear.hpp"
#include "../include/dreidel/layers/GELU.hpp"
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

            // Expanded distribution: 0-29
            std::uniform_int_distribution<int> dist_type(0, 29);
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
                } else if (type < 20) {
                    // High Complexity: Sum of 3 random wavelets
                    std::fill(ptr + offset, ptr + offset + dim, 0);
                    for(int k=0; k<3; ++k) {
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
                        for(size_t i=0; i<dim; ++i) ptr[offset+i] += temp[i];
                    }
                    for(size_t i=0; i<dim; ++i) ptr[offset+i] /= 1.5;
                } else if (type < 25) {
                    // Texture: Interference pattern (Sum of 5 cosines)
                    std::fill(ptr + offset, ptr + offset + dim, 0);
                    for(int k=0; k<5; ++k) {
                        T kx = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                        T ky = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                        T phase = dist_phase(gen);
                        for(size_t y=0; y<size; ++y) {
                            for(size_t x=0; x<size; ++x) {
                                ptr[offset + y*size + x] += std::cos(kx*x + ky*y + phase);
                            }
                        }
                    }
                    for(size_t i=0; i<dim; ++i) ptr[offset+i] /= 2.5;
                } else {
                    // Geometric: Rectangles or Lines
                    std::fill(ptr + offset, ptr + offset + dim, -1.0f);
                    T cx = dist_pos(gen);
                    T cy = dist_pos(gen);
                    T w = dist_scale(gen);
                    T h = dist_scale(gen);
                    T angle = dist_angle(gen);
                    T ca = std::cos(angle);
                    T sa = std::sin(angle);

                    for(size_t y=0; y<size; ++y) {
                        for(size_t x=0; x<size; ++x) {
                            T dx = (T)x - cx;
                            T dy = (T)y - cy;
                            T rx = dx * ca - dy * sa;
                            T ry = dx * sa + dy * ca;
                            if (std::abs(rx) < w && std::abs(ry) < h) {
                                ptr[offset + y*size + x] = 1.0f;
                            }
                        }
                    }
                }

                // Normalize batch item to [-1, 1]
                T max_val = 0;
                for(size_t i=0; i<dim; ++i) {
                    max_val = std::max(max_val, std::abs(ptr[offset+i]));
                }
                if (max_val > 1e-6) {
                    for(size_t i=0; i<dim; ++i) ptr[offset+i] /= max_val;
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
        // Encoder: DSL -> GELU -> DSL -> Tanh
        enc_1_ = new layers::DeepSpectralLinear<T>(dim, 8);
        enc_act_1_ = new layers::GELU<T>();
        enc_2_ = new layers::DeepSpectralLinear<T>(dim, 8);
        enc_act_2_ = new Tanh<T>();

        // Decoder: DSL -> GELU -> DSL
        dec_1_ = new layers::DeepSpectralLinear<T>(dim, 8);
        dec_act_1_ = new layers::GELU<T>();
        dec_2_ = new layers::DeepSpectralLinear<T>(dim, 8);

        layers_.push_back(enc_1_);
        layers_.push_back(enc_act_1_);
        layers_.push_back(enc_2_);
        layers_.push_back(enc_act_2_);

        layers_.push_back(dec_1_);
        layers_.push_back(dec_act_1_);
        layers_.push_back(dec_2_);
    }

    ~WaveletAutoencoder() {
        for(auto l : layers_) delete l;
    }

    // Returns latent code z
    Tensor<T> encode(const Tensor<T>& x) {
        Tensor<T> h = enc_1_->forward(x);
        h = enc_act_1_->forward(h);
        h = enc_2_->forward(h);
        return enc_act_2_->forward(h);
    }

    // Returns reconstruction
    Tensor<T> decode(const Tensor<T>& z) {
        Tensor<T> h = dec_1_->forward(z);
        h = dec_act_1_->forward(h);
        return dec_2_->forward(h);
    }

    Tensor<T> forward(const Tensor<T>& x) {
        return decode(encode(x));
    }

    Tensor<T> forward_train(const Tensor<T>& x, Tensor<T>& z_out) {
        z_out = encode(x);
        return decode(z_out);
    }

    // Custom backward for regularization
    void backward_with_reg(const Tensor<T>& grad_recon, const Tensor<T>& grad_reg_z) {
        // Decoder backward
        Tensor<T> g = dec_2_->backward(grad_recon);
        g = dec_act_1_->backward(g);
        g = dec_1_->backward(g);

        // Inject regularization at z (output of enc_act_2_)
        // Total dL/dz = dL_recon/dz + dL_reg/dz
        g = g + grad_reg_z;

        // Encoder backward
        g = enc_act_2_->backward(g);
        g = enc_2_->backward(g);
        g = enc_act_1_->backward(g);
        enc_1_->backward(g);
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
    layers::Layer<T>* enc_1_;
    layers::Layer<T>* enc_act_1_;
    layers::Layer<T>* enc_2_;
    layers::Layer<T>* enc_act_2_;

    layers::Layer<T>* dec_1_;
    layers::Layer<T>* dec_act_1_;
    layers::Layer<T>* dec_2_;
};

// Helper to save PNG grid
void save_png_grid(const std::string& filename, const std::vector<std::vector<float>>& images, int rows, int cols, int size) {
    int scale = 2;
    int padding = 10;
    int img_w = size * scale;
    int img_h = size * scale;
    int total_width = cols * (img_w + padding) + padding;
    int total_height = rows * (img_h + padding) + padding;

    std::vector<unsigned char> pixels(total_width * total_height, 255); // White background

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
                    unsigned char gray = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, norm * 255.0f)));

                    for(int sy = 0; sy < scale; ++sy) {
                        for(int sx = 0; sx < scale; ++sx) {
                            int px = start_x + x*scale + sx;
                            int py = start_y + y*scale + sy;
                            if (px < total_width && py < total_height) {
                                pixels[py * total_width + px] = gray;
                            }
                        }
                    }
                }
            }
        }
    }

    stbi_write_png(filename.c_str(), total_width, total_height, 1, pixels.data(), total_width);
}

// Training Loop
template <typename T>
void train(size_t epochs, size_t batches_per_epoch, size_t batch_size) {
    size_t dim = 4096; // 64x64
    WaveletAutoencoder<T> model(dim);
    optim::DiagonalNewton<T> optimizer(0.01); // Lower LR for stability with complex textures

    // Register params
    optimizer.add_parameters(model.parameters(), model.gradients(), model.curvatures());

    std::cout << "Starting training: " << epochs << " epochs, " << batches_per_epoch << " batches/epoch, batch " << batch_size << std::endl;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        T epoch_mse = 0;
        T epoch_reg = 0;
        T last_avg_abs_z = 0;

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

            // 2. Binary Regularization (DISABLED)
            // Just train on MSE as requested
            Tensor<T> grad_reg_z({batch_size, dim});
            grad_reg_z.fill(0); // No regularization gradient

            T reg_loss = 0;
            T avg_abs_z = 0;
            const T* z_ptr = z.data();
            for(size_t k=0; k<z.size(); ++k) {
                avg_abs_z += std::abs(z_ptr[k]);
            }
            avg_abs_z /= z.size();
            last_avg_abs_z = avg_abs_z;

            model.backward_with_reg(grad_recon, grad_reg_z);

            optimizer.step();

            epoch_mse += mse;
            epoch_reg += reg_loss;
        }

        if (epoch % 10 == 0) {
            // Log statistics
            std::cout << "Epoch " << epoch << ": MSE=" << epoch_mse/batches_per_epoch
                      << " Reg=" << epoch_reg/batches_per_epoch
                      << " Avg|z|=" << last_avg_abs_z
                      << std::endl;
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

    // Visualize (Reduce to 2 samples to keep SVG size manageable)
    std::vector<std::vector<float>> vis_images;
    const T* x_ptr = test_x.data();
    const T* z_ptr = test_z.data();
    const T* r_ptr = test_rec.data();
    const T* rq_ptr = test_rec_q.data();

    int vis_cols = 2;

    // We visualize vis_cols samples, 4 rows: Original, Latent, Rec Float, Rec Quant
    // 1. Originals
    for(int i=0; i<vis_cols; ++i) {
        std::vector<float> img(dim);
        for(int j=0; j<dim; ++j) img[j] = x_ptr[i*dim + j];
        vis_images.push_back(img);
    }
    // 2. Latents (reshaped to 64x64)
    for(int i=0; i<vis_cols; ++i) {
        std::vector<float> img(dim);
        for(int j=0; j<dim; ++j) img[j] = z_ptr[i*dim + j];
        vis_images.push_back(img);
    }
    // 3. Rec Float
    for(int i=0; i<vis_cols; ++i) {
        std::vector<float> img(dim);
        for(int j=0; j<dim; ++j) img[j] = r_ptr[i*dim + j];
        vis_images.push_back(img);
    }
    // 4. Rec Quant
    for(int i=0; i<vis_cols; ++i) {
        std::vector<float> img(dim);
        for(int j=0; j<dim; ++j) img[j] = rq_ptr[i*dim + j];
        vis_images.push_back(img);
    }

    save_png_grid("reconstruction_grid.png", vis_images, 4, vis_cols, 64);
    std::cout << "Saved visualization to reconstruction_grid.png" << std::endl;
}

int main() {
    train<float>(500, 1, 16); // 500 epochs (Increase for full convergence)
    return 0;
}
