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
#include "../include/dreidel/models/ZenithGhostAE.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"

using namespace dreidel;

// 2D Wavelet Generator
template <typename T>
class WaveletGenerator2D {
public:
    static void generate(Tensor<T>& batch, size_t batch_size) {
        T* ptr = batch.data();
        size_t dim = batch.shape()[1];
        size_t size = static_cast<size_t>(std::sqrt(dim));
        if (size * size > dim) size--;

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
                int type = dist_type(gen);
                size_t offset = b * dim;
                if (type < 5) { // Gabor
                    T cx = dist_pos(gen); T cy = dist_pos(gen); T sx = dist_scale(gen); T sy = sx * std::uniform_real_distribution<T>(0.5, 1.5)(gen);
                    generate_gabor(ptr + offset, size, cx, cy, sx, sy, dist_angle(gen), dist_freq(gen), dist_phase(gen));
                } else if (type < 10) { // Curvelet
                    T cx = dist_pos(gen); T cy = dist_pos(gen); T sx = dist_scale(gen) * 1.5; T sy = sx * 0.1;
                    generate_gabor(ptr + offset, size, cx, cy, sx, sy, dist_angle(gen), dist_freq(gen), dist_phase(gen));
                } else if (type < 15) { // Mexican Hat
                    T cx = dist_pos(gen); T cy = dist_pos(gen); T s = dist_scale(gen);
                    generate_mexican_hat(ptr + offset, size, cx, cy, s);
                } else if (type < 20) { // Complex
                    std::fill(ptr + offset, ptr + offset + dim, 0);
                    for(int k=0; k<3; ++k) {
                        int subtype = std::uniform_int_distribution<int>(0, 2)(gen);
                        std::vector<T> temp(dim);
                        T cx = dist_pos(gen); T cy = dist_pos(gen); T theta = dist_angle(gen);
                        if (subtype == 0) generate_gabor(temp.data(), size, cx, cy, dist_scale(gen), dist_scale(gen), theta, dist_freq(gen), dist_phase(gen));
                        else if (subtype == 1) generate_gabor(temp.data(), size, cx, cy, dist_scale(gen)*1.5, dist_scale(gen)*0.15, theta, dist_freq(gen), dist_phase(gen));
                        else generate_mexican_hat(temp.data(), size, cx, cy, dist_scale(gen));
                        for(size_t i=0; i<dim; ++i) ptr[offset+i] += temp[i];
                    }
                    for(size_t i=0; i<dim; ++i) ptr[offset+i] /= 1.5;
                } else if (type < 25) { // Texture
                    std::fill(ptr + offset, ptr + offset + dim, 0);
                    for(int k=0; k<5; ++k) {
                        T kx = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                        T ky = std::uniform_real_distribution<T>(0.1, 0.8)(gen);
                        T phase = dist_phase(gen);
                        for(size_t y=0; y<size; ++y) for(size_t x=0; x<size; ++x) ptr[offset + y*size + x] += std::cos(kx*x + ky*y + phase);
                    }
                    for(size_t i=0; i<dim; ++i) ptr[offset+i] /= 2.5;
                } else { // Geometric
                    std::fill(ptr + offset, ptr + offset + dim, -1.0f);
                    T cx = dist_pos(gen); T cy = dist_pos(gen); T w = dist_scale(gen); T h = dist_scale(gen);
                    T angle = dist_angle(gen);
                    T ca = std::cos(angle); T sa = std::sin(angle);
                    for(size_t y=0; y<size; ++y) for(size_t x=0; x<size; ++x) {
                        T dx = (T)x - cx; T dy = (T)y - cy;
                        if (std::abs(dx * ca - dy * sa) < w && std::abs(dx * sa + dy * ca) < h) ptr[offset + y*size + x] = 1.0f;
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
        T cos_t = std::cos(theta); T sin_t = std::sin(theta);
        for(size_t y=0; y<size; ++y) for(size_t x=0; x<size; ++x) {
            T dx = (T)x - cx; T dy = (T)y - cy;
            T xp = dx * cos_t + dy * sin_t; T yp = -dx * sin_t + dy * cos_t;
            buffer[y*size + x] = std::exp(-(xp*xp)/(2*sx*sx) - (yp*yp)/(2*sy*sy)) * std::cos(2*3.14159 * freq * xp + psi);
        }
    }
    static void generate_mexican_hat(T* buffer, size_t size, T cx, T cy, T sigma) {
        for(size_t y=0; y<size; ++y) for(size_t x=0; x<size; ++x) {
            T dx = (T)x - cx; T dy = (T)y - cy;
            T r2 = dx*dx + dy*dy; T s2 = sigma*sigma;
            buffer[y*size + x] = (1.0 - r2/s2) * std::exp(-r2/(2*s2));
        }
    }
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
            for (float v : img) { if (v < min_val) min_val = v; if (v > max_val) max_val = v; }
            if (max_val == min_val) max_val = min_val + 1e-6;
            int start_x = padding + c * (img_w + padding);
            int start_y = padding + r * (img_h + padding);
            for (int y = 0; y < size; ++y) for (int x = 0; x < size; ++x) {
                float val = img[y * size + x];
                float norm = (val - min_val) / (max_val - min_val);
                unsigned char gray = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, norm * 255.0f)));
                for(int sy = 0; sy < scale; ++sy) for(int sx = 0; sx < scale; ++sx) {
                    int px = start_x + x*scale + sx;
                    int py = start_y + y*scale + sy;
                    if (px < total_width && py < total_height) pixels[py * total_width + px] = gray;
                }
            }
        }
    }
    stbi_write_png(filename.c_str(), total_width, total_height, 1, pixels.data(), total_width);
}

template <typename T>
void train(size_t epochs, size_t batches_per_epoch, size_t batch_size, size_t dim, const std::string& init_scheme) {
    size_t size = static_cast<size_t>(std::sqrt(dim)); // 128

    std::cout << "Initializing ZenithGhostAE. Dim=" << dim << " (" << size << "x" << size << ")" << " Init=" << init_scheme << std::endl;
    models::ZenithGhostAE<T> model(init_scheme);

    // Optimizer
    // ZenithBlock works best with low LR?
    // Memory says "3.25x training speedup... requires lower learning rates".
    // "SimpleAdam optimizer supports Coordinate-Wise Clipping".
    optim::SimpleAdam<T> optimizer(0.001f);
    optimizer.add_parameters(model.parameters(), model.gradients());

    std::cout << "Starting Training (" << epochs << " epochs, " << batches_per_epoch << " batches/epoch)..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        T epoch_recon_mae = 0;
        T epoch_ghost_mae = 0;

        for (size_t b = 0; b < batches_per_epoch; ++b) {
            // Generate Wavelets (B, H*W)
            Tensor<T> flat_wavelets({batch_size, dim});
            WaveletGenerator2D<T>::generate(flat_wavelets, batch_size);

            // Convert to (B, H, W, 3)
            Tensor<T> input({batch_size, size, size, 3});
            T* in_ptr = input.data();
            const T* flat_ptr = flat_wavelets.data();

            #pragma omp parallel for collapse(2)
            for(size_t i=0; i<batch_size; ++i) {
                for(size_t j=0; j<dim; ++j) {
                    T val = flat_ptr[i*dim + j];
                    // Replicate to 3 channels
                    in_ptr[((i*size + j/size)*size + j%size)*3 + 0] = val;
                    in_ptr[((i*size + j/size)*size + j%size)*3 + 1] = val;
                    in_ptr[((i*size + j/size)*size + j%size)*3 + 2] = val;
                }
            }

            optimizer.zero_grad();
            model.set_training(true);

            auto out = model.forward_train(input);

            // 1. Reconstruction Loss (MAE)
            Tensor<T> diff = out.reconstruction - input;

            T mae_recon = 0;
            T* diff_ptr = diff.data();
            for(size_t k=0; k<diff.size(); ++k) mae_recon += std::abs(diff_ptr[k]);
            mae_recon /= diff.size();

            // Grad Recon (Sign of diff)
            Tensor<T> grad_recon = diff;
            T* gr_ptr = grad_recon.data();
            for(size_t k=0; k<grad_recon.size(); ++k) {
                T v = gr_ptr[k];
                gr_ptr[k] = (v > 0) ? 1.0f : ((v < 0) ? -1.0f : 0.0f);
                gr_ptr[k] /= diff.size();
                gr_ptr[k] *= 1000.0f;
            }

            // 2. Ghost Loss
            T mae_ghost = 0;
            std::vector<Tensor<T>> grad_ghosts;

            for(size_t i=0; i<out.ghost_preds.size(); ++i) {
                Tensor<T> g_diff = out.ghost_preds[i] - out.encoder_targets[i];
                T g_mae = 0;
                const T* gd_ptr = g_diff.data();
                for(size_t k=0; k<g_diff.size(); ++k) g_mae += std::abs(gd_ptr[k]);
                g_mae /= g_diff.size();
                mae_ghost += g_mae;

                Tensor<T> g_grad = g_diff;
                T* gg_ptr = g_grad.data();
                for(size_t k=0; k<g_grad.size(); ++k) {
                    T v = gg_ptr[k];
                    gg_ptr[k] = (v > 0) ? 1.0f : ((v < 0) ? -1.0f : 0.0f);
                    gg_ptr[k] /= g_diff.size();
                    gg_ptr[k] *= 100.0f; // Weight for ghost loss
                }
                grad_ghosts.push_back(g_grad);
            }

            model.backward_train(grad_recon, grad_ghosts);
            optimizer.step();

            epoch_recon_mae += mae_recon;
            epoch_ghost_mae += mae_ghost;
        }

        if (epoch % 1 == 0) { // Log every epoch
            std::cout << "Epoch " << epoch
                      << ": ReconMAE=" << epoch_recon_mae / batches_per_epoch
                      << " GhostMAE=" << epoch_ghost_mae / batches_per_epoch
                      << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Training Time: " << elapsed.count() << "s for " << epochs * batches_per_epoch << " batches." << std::endl;
    std::cout << "Speed: " << (epochs * batches_per_epoch * batch_size) / elapsed.count() << " samples/sec" << std::endl;

    // Benchmarking Inference Speed
    std::cout << "\nBenchmarking Inference Speed..." << std::endl;
    model.set_training(false);
    size_t bench_batches = 10;

    Tensor<T> bench_input({batch_size, size, size, 3});
    bench_input.fill(0.5f); // Dummy data

    auto t1 = std::chrono::high_resolution_clock::now();
    for(size_t b=0; b<bench_batches; ++b) {
        // Reuse input
        Tensor<T> out = model.forward(bench_input); // Inference Forward
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t2 - t1;
    double fps_infer = (bench_batches * batch_size) / dt.count();
    std::cout << "Inference Speed: " << fps_infer << " samples/sec" << std::endl;

    // Benchmarking Training Forward Speed (Ghosts included)
    std::cout << "Benchmarking Training Forward Speed (with Ghosts)..." << std::endl;
    model.set_training(true);
    t1 = std::chrono::high_resolution_clock::now();
    for(size_t b=0; b<bench_batches; ++b) {
        auto out = model.forward_train(bench_input);
    }
    t2 = std::chrono::high_resolution_clock::now();
    dt = t2 - t1;
    double fps_train_fwd = (bench_batches * batch_size) / dt.count();
    std::cout << "Training Forward Speed: " << fps_train_fwd << " samples/sec" << std::endl;

    // Visualization
    std::cout << "Generating Visualization..." << std::endl;
    model.set_training(false);

    size_t test_size = 4;
    Tensor<T> flat_test({test_size, dim});
    WaveletGenerator2D<T>::generate(flat_test, test_size);
    Tensor<T> input_test({test_size, size, size, 3});

    T* in_ptr = input_test.data();
    const T* flat_ptr = flat_test.data();
    for(size_t b=0; b<test_size; ++b) {
        for(size_t i=0; i<dim; ++i) {
            T val = flat_ptr[b*dim + i];
            in_ptr[((b*size + i/size)*size + i%size)*3 + 0] = val;
            in_ptr[((b*size + i/size)*size + i%size)*3 + 1] = val;
            in_ptr[((b*size + i/size)*size + i%size)*3 + 2] = val;
        }
    }

    Tensor<T> rec = model.forward(input_test);

    std::vector<std::vector<float>> vis_images;
    // Extract first channel for vis
    for(size_t i=0; i<test_size; ++i) {
        std::vector<float> img(dim);
        for(size_t j=0; j<dim; ++j) img[j] = in_ptr[i*dim*3 + j*3];
        vis_images.push_back(img);
    }
    const T* r_ptr = rec.data();
    for(size_t i=0; i<test_size; ++i) {
        std::vector<float> img(dim);
        for(size_t j=0; j<dim; ++j) img[j] = r_ptr[i*dim*3 + j*3];
        vis_images.push_back(img);
    }

    save_png_grid("ghost_reconstruction.png", vis_images, 2, test_size, size);
}

int main(int argc, char** argv) {
    size_t epochs = 100;
    size_t batches_per_epoch = 10;
    size_t batch_size = 4;
    // Needs 128x128 -> dim = 16384
    size_t dim = 16384;
    std::string init_scheme = "he";

    if(argc > 1) epochs = std::atoi(argv[1]);
    if(argc > 2) batches_per_epoch = std::atoi(argv[2]);
    if(argc > 3) init_scheme = argv[3];

    train<float>(epochs, batches_per_epoch, batch_size, dim, init_scheme);
    return 0;
}
