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
#include "../include/dreidel/layers/DeepSpectralLinear.hpp"
#include "../include/dreidel/layers/GELU.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"

using namespace dreidel;

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

// 2D Wavelet Generator
template <typename T>
class WaveletGenerator2D {
public:
    static void generate(Tensor<T>& batch, size_t batch_size) {
        T* ptr = batch.data();
        size_t dim = 4096;
        size_t size = 64;

        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd() + omp_get_thread_num());

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

                if (type < 5) { // Gabor
                    T cx = dist_pos(gen);
                    T cy = dist_pos(gen);
                    T sx = dist_scale(gen);
                    T sy = sx * std::uniform_real_distribution<T>(0.5, 1.5)(gen);
                    T theta = dist_angle(gen);
                    generate_gabor(ptr + offset, size, cx, cy, sx, sy, theta, dist_freq(gen), dist_phase(gen));
                } else if (type < 10) { // Curvelet-like
                    T cx = dist_pos(gen);
                    T cy = dist_pos(gen);
                    T sx = dist_scale(gen) * 1.5;
                    T sy = sx * 0.1;
                    T theta = dist_angle(gen);
                    generate_gabor(ptr + offset, size, cx, cy, sx, sy, theta, dist_freq(gen), dist_phase(gen));
                } else if (type < 15) { // Mexican Hat
                    T cx = dist_pos(gen);
                    T cy = dist_pos(gen);
                    T s = dist_scale(gen);
                    generate_mexican_hat(ptr + offset, size, cx, cy, s);
                } else if (type < 20) { // High Complexity
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
                } else if (type < 25) { // Texture
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
                } else { // Geometric
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

                T max_val = 0;
                for(size_t i=0; i<dim; ++i) max_val = std::max(max_val, std::abs(ptr[offset+i]));
                if (max_val > 1e-6) for(size_t i=0; i<dim; ++i) ptr[offset+i] /= max_val;
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

// Wavelet Autoencoder with Residuals
template <typename T>
class WaveletAutoencoder {
public:
    WaveletAutoencoder(size_t dim) {
        // Reduced depth to 4 for stability
        dsl_enc_1_ = new layers::DeepSpectralLinear<T>(dim, 4);
        enc_act_1_ = new layers::GELU<T>();
        dsl_enc_2_ = new layers::DeepSpectralLinear<T>(dim, 4);
        enc_act_2_ = new Tanh<T>();

        dsl_dec_1_ = new layers::DeepSpectralLinear<T>(dim, 4);
        dec_act_1_ = new layers::GELU<T>();
        dsl_dec_2_ = new layers::DeepSpectralLinear<T>(dim, 4);

        layers_.push_back(dsl_enc_1_);
        layers_.push_back(enc_act_1_);
        layers_.push_back(dsl_enc_2_);
        layers_.push_back(enc_act_2_);
        layers_.push_back(dsl_dec_1_);
        layers_.push_back(dec_act_1_);
        layers_.push_back(dsl_dec_2_);
    }

    ~WaveletAutoencoder() {
        for(auto l : layers_) delete l;
    }

    // Force initialization of scales to 1/sqrt(dim) to preserve variance
    void init_scales(size_t dim) {
        T scale_val = 1.0 / std::sqrt(static_cast<T>(dim));
        auto set_params = [&](layers::DeepSpectralLinear<T>* dsl) {
            auto params = dsl->parameters();
            for(auto* p : params) {
                if(p->size() == dim) { // Identify scale vector
                    p->fill(scale_val);
                    // Add small noise to break symmetry?
                    // Actually, uniform scale is fine for identity + residuals.
                    // Let's add very small noise.
                    T* ptr = p->data();
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<T> dist(-0.001*scale_val, 0.001*scale_val);
                    for(size_t i=0; i<dim; ++i) ptr[i] += dist(gen);
                }
            }
        };
        set_params(dsl_enc_1_);
        set_params(dsl_enc_2_);
        set_params(dsl_dec_1_);
        set_params(dsl_dec_2_);
    }

    Tensor<T> encode(const Tensor<T>& x) {
        // ResNet Block 1
        Tensor<T> h = dsl_enc_1_->forward(x);
        Tensor<T> res1 = x + h;
        h = enc_act_1_->forward(res1);

        // ResNet Block 2
        Tensor<T> h2 = dsl_enc_2_->forward(h);
        Tensor<T> res2 = h + h2;
        return enc_act_2_->forward(res2);
    }

    Tensor<T> decode(const Tensor<T>& z) {
        // ResNet Block 1
        Tensor<T> h = dsl_dec_1_->forward(z);
        Tensor<T> res1 = z + h;
        h = dec_act_1_->forward(res1);

        // ResNet Block 2
        Tensor<T> h2 = dsl_dec_2_->forward(h);
        return h + h2;
    }

    Tensor<T> forward_train(const Tensor<T>& x, Tensor<T>& z_out) {
        z_out = encode(x);
        return decode(z_out);
    }

    void backward_with_reg(const Tensor<T>& grad_recon, const Tensor<T>& grad_reg_z) {
        // Decoder
        // h_out = h + DSL2(h)
        // dL/dh_out = grad_recon
        // dL/dh = dL/dh_out + DSL2_back(dL/dh_out)

        // Block 2
        Tensor<T> grad_h2_out = grad_recon;
        Tensor<T> grad_dsl2_in = dsl_dec_2_->backward(grad_h2_out);
        Tensor<T> grad_h = grad_h2_out + grad_dsl2_in; // Residual backward

        // Block 1
        // h = Act1(res1)
        Tensor<T> grad_res1 = dec_act_1_->backward(grad_h);
        // res1 = z + DSL1(z)
        Tensor<T> grad_dsl1_in = dsl_dec_1_->backward(grad_res1);
        Tensor<T> grad_z = grad_res1 + grad_dsl1_in; // Residual backward

        // Add Reg
        grad_z = grad_z + grad_reg_z;

        // Encoder
        // Block 2
        // z = Act2(res2)
        Tensor<T> grad_res2 = enc_act_2_->backward(grad_z);
        // res2 = h_enc + DSL2(h_enc)
        Tensor<T> grad_dsl2_enc_in = dsl_enc_2_->backward(grad_res2);
        Tensor<T> grad_h_enc = grad_res2 + grad_dsl2_enc_in;

        // Block 1
        // h_enc = Act1(res1_enc)
        Tensor<T> grad_res1_enc = enc_act_1_->backward(grad_h_enc);
        // res1_enc = x + DSL1(x)
        Tensor<T> grad_dsl1_enc_in = dsl_enc_1_->backward(grad_res1_enc);
        // grad_x = grad_res1_enc + grad_dsl1_enc_in
        // We don't need grad_x unless chaining further.
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

    void save(const std::string& filename, size_t epoch) {
        std::ofstream out(filename, std::ios::binary);
        if(!out) return;

        out.write((char*)&epoch, sizeof(size_t));

        auto save_dsl = [&](layers::DeepSpectralLinear<T>* dsl) {
            auto params = dsl->parameters();
            size_t depth = params.size();
            out.write((char*)&depth, sizeof(size_t));

            for(size_t k=0; k<depth; ++k) {
                // Save Scale
                size_t sz = params[k]->size();
                out.write((char*)&sz, sizeof(size_t));
                out.write((char*)params[k]->data(), sz * sizeof(T));

                // Save Permutation
                const auto& p = dsl->get_permutation(k);
                size_t p_sz = p.size();
                out.write((char*)&p_sz, sizeof(size_t));
                out.write((char*)p.data(), p_sz * sizeof(size_t));
            }
        };

        save_dsl(dsl_enc_1_);
        save_dsl(dsl_enc_2_);
        save_dsl(dsl_dec_1_);
        save_dsl(dsl_dec_2_);
        out.close();
        std::cout << "Saved Checkpoint to " << filename << std::endl;
    }

    bool load(const std::string& filename, size_t& epoch) {
        std::ifstream in(filename, std::ios::binary);
        if(!in) return false;

        in.read((char*)&epoch, sizeof(size_t));

        auto load_dsl = [&](layers::DeepSpectralLinear<T>* dsl) {
            size_t depth;
            in.read((char*)&depth, sizeof(size_t));
            auto params = dsl->parameters();

            for(size_t k=0; k<depth; ++k) {
                size_t sz;
                in.read((char*)&sz, sizeof(size_t));
                in.read((char*)params[k]->data(), sz * sizeof(T));

                size_t p_sz;
                in.read((char*)&p_sz, sizeof(size_t));
                std::vector<size_t> p(p_sz);
                in.read((char*)p.data(), p_sz * sizeof(size_t));

                dsl->set_permutation(k, p);
            }
        };

        load_dsl(dsl_enc_1_);
        load_dsl(dsl_enc_2_);
        load_dsl(dsl_dec_1_);
        load_dsl(dsl_dec_2_);
        in.close();
        std::cout << "Loaded Checkpoint from " << filename << " (Epoch " << epoch << ")" << std::endl;
        return true;
    }

private:
    std::vector<layers::Layer<T>*> layers_;
    layers::DeepSpectralLinear<T>* dsl_enc_1_;
    layers::Layer<T>* enc_act_1_;
    layers::DeepSpectralLinear<T>* dsl_enc_2_;
    layers::Layer<T>* enc_act_2_;

    layers::DeepSpectralLinear<T>* dsl_dec_1_;
    layers::Layer<T>* dec_act_1_;
    layers::DeepSpectralLinear<T>* dsl_dec_2_;
};

void save_png_grid(const std::string& filename, const std::vector<std::vector<float>>& images, int rows, int cols, int size) {
    int scale = 2;
    int padding = 10;
    int img_w = size * scale;
    int img_h = size * scale;
    int total_width = cols * (img_w + padding) + padding;
    int total_height = rows * (img_h + padding) + padding;

    std::vector<unsigned char> pixels(total_width * total_height, 255);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int img_idx = r * cols + c;
            if (img_idx >= images.size()) continue;

            const auto& img = images[img_idx];
            float min_val = 1e9, max_val = -1e9;
            for (float v : img) {
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }
            if (max_val == min_val) max_val = min_val + 1e-6;

            int start_x = padding + c * (img_w + padding);
            int start_y = padding + r * (img_h + padding);

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

template <typename T>
void compute_tv_grad(const Tensor<T>& img_batch, Tensor<T>& grad_tv, size_t width, size_t height) {
    size_t batch_size = img_batch.shape()[0];
    const T* img_ptr = img_batch.data();
    T* g_ptr = grad_tv.data();
    grad_tv.fill(0);

    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        size_t offset = b * width * height;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t idx = offset + y * width + x;

                if (x + 1 < width) {
                    size_t idx_right = idx + 1;
                    T diff = img_ptr[idx_right] - img_ptr[idx];
                    T sign = (diff > 0) ? 1.0f : ((diff < 0) ? -1.0f : 0.0f);
                    g_ptr[idx_right] += sign;
                    g_ptr[idx] -= sign;
                }

                if (y + 1 < height) {
                    size_t idx_down = idx + width;
                    T diff = img_ptr[idx_down] - img_ptr[idx];
                    T sign = (diff > 0) ? 1.0f : ((diff < 0) ? -1.0f : 0.0f);
                    g_ptr[idx_down] += sign;
                    g_ptr[idx] -= sign;
                }
            }
        }
    }

    T scale = 1.0f / batch_size;
    for(size_t i=0; i<grad_tv.size(); ++i) g_ptr[i] *= scale;
}

template <typename T>
void train(size_t epochs_to_run, size_t batches_per_epoch, size_t batch_size, const std::string& checkpoint_file) {
    size_t dim = 4096;
    size_t side = 64;
    WaveletAutoencoder<T> model(dim);

    // Check if checkpoint exists
    size_t start_epoch = 0;
    bool loaded = model.load(checkpoint_file, start_epoch);
    if (loaded) {
        start_epoch += 1;
    } else {
        std::cout << "No checkpoint found. Starting from scratch with Identity Initialization." << std::endl;
        model.init_scales(dim);
    }

    // LR Schedule: 0.1 -> 0.01
    optim::DiagonalNewton<T> optimizer(0.05, 1e-8, 1.0);
    optimizer.add_parameters(model.parameters(), model.gradients(), model.curvatures());

    size_t end_epoch = start_epoch + epochs_to_run;
    std::cout << "Training from epoch " << start_epoch << " to " << end_epoch << std::endl;

    for (size_t epoch = start_epoch; epoch < end_epoch; ++epoch) {
        T epoch_mse = 0;
        T last_avg_abs_z = 0;

        for (size_t b = 0; b < batches_per_epoch; ++b) {
            Tensor<T> x({batch_size, dim});
            WaveletGenerator2D<T>::generate(x, batch_size);

            optimizer.zero_grad();

            Tensor<T> z;
            Tensor<T> y = model.forward_train(x, z);

            // MSE
            Tensor<T> diff = y + (x * -1.0);
            T mse = 0;
            const T* d_ptr = diff.data();
            for(size_t k=0; k<diff.size(); ++k) mse += d_ptr[k]*d_ptr[k];
            mse /= diff.size();

            Tensor<T> grad_recon = diff * (2.0 / diff.size());

            // TV Loss
            Tensor<T> grad_tv({batch_size, dim});
            compute_tv_grad(y, grad_tv, side, side);

            T tv_weight = 0.0001;
            const T* gtv_ptr = grad_tv.data();
            T* gr_ptr = grad_recon.data();
            for(size_t k=0; k<grad_recon.size(); ++k) {
                gr_ptr[k] += tv_weight * gtv_ptr[k];
            }

            // Binary Reg DISABLED
            Tensor<T> grad_reg_z({batch_size, dim});
            grad_reg_z.fill(0);

            T avg_abs_z = 0;
            const T* z_ptr = z.data();
            for(size_t k=0; k<z.size(); ++k) avg_abs_z += std::abs(z_ptr[k]);
            avg_abs_z /= z.size();
            last_avg_abs_z = avg_abs_z;

            model.backward_with_reg(grad_recon, grad_reg_z);

            optimizer.step();

            epoch_mse += mse;
        }

        if (epoch % 10 == 0 || epoch == end_epoch - 1) {
            std::cout << "Epoch " << epoch << ": MSE=" << epoch_mse/batches_per_epoch
                      << " Avg|z|=" << last_avg_abs_z
                      << std::endl;
        }

        if (epoch > 0 && epoch % 100 == 0) {
            model.save(checkpoint_file, epoch);
        }
    }

    model.save(checkpoint_file, end_epoch - 1);

    // Visualization
    std::cout << "Generating Visualization..." << std::endl;
    size_t test_size = 4;
    Tensor<T> test_x({test_size, dim});
    WaveletGenerator2D<T>::generate(test_x, test_size);
    Tensor<T> test_z = model.encode(test_x);
    Tensor<T> test_rec = model.decode(test_z);

    std::vector<std::vector<float>> vis_images;
    const T* x_ptr = test_x.data();
    const T* r_ptr = test_rec.data();

    for(size_t i=0; i<test_size; ++i) {
        std::vector<float> img(dim);
        for(size_t j=0; j<dim; ++j) img[j] = x_ptr[i*dim+j];
        vis_images.push_back(img);
    }
    for(size_t i=0; i<test_size; ++i) {
        std::vector<float> img(dim);
        for(size_t j=0; j<dim; ++j) img[j] = r_ptr[i*dim+j];
        vis_images.push_back(img);
    }

    save_png_grid("reconstruction_grid.png", vis_images, 2, test_size, 64);
}

int main(int argc, char** argv) {
    size_t epochs = 1000;
    if(argc > 1) epochs = std::atoi(argv[1]);
    train<float>(epochs, 1, 64, "checkpoint.bin");
    return 0;
}
