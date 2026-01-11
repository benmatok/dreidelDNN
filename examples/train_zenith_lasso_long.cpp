#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <string>

// Include stb_image_write
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/models/ZenithLassoAE.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"

using namespace dreidel;

// Helper to save Tensor as PNG
// Assumes Tensor shape [Batch, H, W, Channels] and saves the first item in batch.
// Normalizes output to 0-255.
template <typename T>
void save_tensor_as_png(const Tensor<T>& tensor, const std::string& filename, size_t batch_idx = 0) {
    auto shape = tensor.shape();
    size_t H = shape[1];
    size_t W = shape[2];
    size_t C = shape[3];

    std::vector<unsigned char> image_data(H * W * (C == 1 ? 1 : 3));

    const T* data = tensor.data() + batch_idx * (H * W * C);

    // Find min/max for normalization
    T min_val = 1e9;
    T max_val = -1e9;
    for (size_t i = 0; i < H * W * C; ++i) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    std::cout << "Saving " << filename << " | Range: [" << min_val << ", " << max_val << "]" << std::endl;

    // Fallback if range is zero or very small
    T range = max_val - min_val;
    if (range < 1e-12f) { // Lowered threshold significantly
        max_val = min_val + 1.0f;
        range = 1.0f;
    }

    for (size_t i = 0; i < H * W; ++i) {
        if (C == 1) {
            float val = (float)(data[i] - min_val) / range;
            // Clamp
            if (val < 0) val = 0;
            if (val > 1) val = 1;
            image_data[i] = (unsigned char)(val * 255.0f);
        } else {
             for(size_t c=0; c<3 && c<C; ++c) {
                 float val = (float)(data[i*C + c] - min_val) / range;
                 if (val < 0) val = 0;
                 if (val > 1) val = 1;
                 image_data[i*3 + c] = (unsigned char)(val * 255.0f);
             }
        }
    }

    int comp = (C == 1) ? 1 : 3;
    stbi_write_png(filename.c_str(), (int)W, (int)H, comp, image_data.data(), (int)W * comp);
}

// Helper to compute Gradient Loss (MSE of Gradients)
template <typename T>
Tensor<T> compute_gradient_tensor(const Tensor<T>& img) {
    auto shape = img.shape();
    size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    Tensor<T> grads({N, H, W, 2 * C}); // Store dx, dy per channel
    grads.fill(0);

    const T* in = img.data();
    T* out = grads.data();

    // Compute central differences (ignoring boundaries for simplicity or using forward/backward)
    // Sobel-like: dx = x[h,w+1] - x[h,w-1], dy = x[h+1,w] - x[h-1,w]
    // Simple diff: dx = x[h,w+1] - x[h,w]
    // User requested: |x[w+1] - x[w]| in python code.
    // Let's implement standard finite difference x[i+1] - x[i].

    #pragma omp parallel for collapse(3)
    for(size_t n=0; n<N; ++n) {
        for(size_t h=0; h<H; ++h) {
            for(size_t w=0; w<W; ++w) {
                for(size_t c=0; c<C; ++c) {
                    T val = in[((n*H + h)*W + w)*C + c];

                    // dx
                    T next_x = (w + 1 < W) ? in[((n*H + h)*W + w + 1)*C + c] : val;
                    T dx = next_x - val;

                    // dy
                    T next_y = (h + 1 < H) ? in[((n*H + h + 1)*W + w)*C + c] : val;
                    T dy = next_y - val;

                    // Store in 2*C layout: [dx_c0, dy_c0, dx_c1, dy_c1...]
                    size_t out_idx = ((n*H + h)*W + w)*(2*C) + 2*c;
                    out[out_idx] = dx;
                    out[out_idx+1] = dy;
                }
            }
        }
    }
    return grads;
}

// Compute Gradient Loss and accumulate gradients into grad_output
// L_grad = Mean((Grad(Pred) - Grad(Target))^2)
// dL_grad/dPred needs to be propagated back.
// d/dx ( (x_{i+1} - x_i)^2 ) = 2(x_{i+1}-x_i) * (-1) [at i] + 2(x_i - x_{i-1}) * (1) [at i]
// Essentially Laplacian-like operator on the difference.
template <typename T>
float accumulate_gradient_loss(Tensor<T>& grad_accum, const Tensor<T>& pred, const Tensor<T>& target, float weight) {
    Tensor<T> g_pred = compute_gradient_tensor(pred);
    Tensor<T> g_tgt = compute_gradient_tensor(target);

    auto shape = pred.shape();
    size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    size_t total = g_pred.size();

    const T* gp = g_pred.data();
    const T* gt = g_tgt.data();
    T* ga = grad_accum.data(); // Accumulate into existing MSE grads

    float loss = 0;

    // We need temporary buffer for dL/dGrad
    Tensor<T> d_grad_diff(g_pred.shape());
    T* dgd = d_grad_diff.data();

    #pragma omp parallel for reduction(+:loss)
    for(size_t i=0; i<total; ++i) {
        T diff = gp[i] - gt[i];
        loss += diff * diff;
        dgd[i] = 2.0f * diff / total; // Scale by mean
    }

    // Propagate dL/dGrad back to dL/dPred
    // For each pixel (h,w), it contributed to dx at (h,w) and (h,w-1), and dy at (h,w) and (h-1,w)
    // dx[w] = p[w+1] - p[w].  d(dx[w])/dp[w] = -1. d(dx[w-1])/dp[w] = 1.
    // dy[h] = p[h+1] - p[h].  d(dy[h])/dp[h] = -1. d(dy[h-1])/dp[h] = 1.

    #pragma omp parallel for collapse(3)
    for(size_t n=0; n<N; ++n) {
        for(size_t h=0; h<H; ++h) {
            for(size_t w=0; w<W; ++w) {
                for(size_t c=0; c<C; ++c) {
                    // Index in dgd
                    // 2*C stride

                    T d_val = 0;

                    // Contribution from dx[w] (where we are the subtractor)
                    size_t idx_dx_curr = ((n*H + h)*W + w)*(2*C) + 2*c;
                    d_val += dgd[idx_dx_curr] * (-1.0f);

                    // Contribution from dx[w-1] (where we are the adder)
                    if (w > 0) {
                         size_t idx_dx_prev = ((n*H + h)*W + w - 1)*(2*C) + 2*c;
                         d_val += dgd[idx_dx_prev] * (1.0f);
                    }

                    // Contribution from dy[h]
                    size_t idx_dy_curr = ((n*H + h)*W + w)*(2*C) + 2*c + 1;
                    d_val += dgd[idx_dy_curr] * (-1.0f);

                    // Contribution from dy[h-1]
                    if (h > 0) {
                        size_t idx_dy_prev = ((n*H + h - 1)*W + w)*(2*C) + 2*c + 1;
                        d_val += dgd[idx_dy_prev] * (1.0f);
                    }

                    // Accumulate with weight
                    #pragma omp atomic
                    ga[((n*H + h)*W + w)*C + c] += weight * d_val;
                }
            }
        }
    }

    return loss * weight;
}

// Data Gen
template <typename T>
void generate_wavelet_batch(Tensor<T>& data, size_t seed_offset) {
    auto shape = data.shape();
    size_t batch = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
    // Use consistent seed + offset
    std::mt19937 gen(12345 + seed_offset);

    T* ptr = data.data();

    #pragma omp parallel for
    for(size_t n=0; n<batch; ++n) {
        size_t eff_n = n + seed_offset;

        // Ensure visible features
        T mu_x = 0.5 * W + 0.25 * W * std::sin(eff_n * 0.3);
        T mu_y = 0.5 * H + 0.25 * H * std::cos(eff_n * 0.2);
        T s_x = (W/8.0) * (1.0 + 0.3 * std::sin(eff_n * 0.5));

        for(size_t c=0; c<C; ++c) {
            for(size_t h=0; h<H; ++h) {
                for(size_t w_idx=0; w_idx<W; ++w_idx) {
                     T x = (T)w_idx - mu_x;
                     T y = (T)h - mu_y;
                     T val = std::exp(-(x*x + y*y)/(2*s_x*s_x));
                     ptr[((n*H + h)*W + w_idx)*C + c] = val;
                }
            }
        }
    }
}

int main() {
    std::cout << "=== Zenith-Lasso Long Training (128x128) ===" << std::endl;

    // Config
    size_t batch_size = 8;
    size_t H = 128, W = 128;
    size_t epochs = 5000;
    float lr = 0.001f; // Lowered LR for stability and smoothness
    float max_lambda = 1e-4f;

    // Model
    models::ZenithLassoAE<float> model;

    // Optimizer
    optim::SimpleAdam<float> optimizer(lr);
    optimizer.set_coordinate_wise_clipping(true, 5.0);

    auto params = model.parameters();
    auto grads = model.gradients();
    optimizer.add_parameters(params, grads);

    Tensor<float> input({batch_size, H, W, 1});
    Tensor<float> target({batch_size, H, W, 1});

    // Fixed visualization batch
    Tensor<float> vis_input({batch_size, H, W, 1});
    generate_wavelet_batch(vis_input, 99999); // Fixed seed for visualization

    auto start_time = std::chrono::high_resolution_clock::now();
    size_t dataset_size = 100; // Simulate 100 fixed batches

    float current_eps = 1.0f; // Start high for stability
    float best_mse = 1e9;

    for(size_t epoch=0; epoch<epochs; ++epoch) {
        // Lambda Schedule
        float current_lambda = 0.0f;
        if (epoch > 100) {
            float progress = (float)(epoch - 100) / 400.0f;
            if (progress > 1.0f) progress = 1.0f;
            current_lambda = max_lambda * progress;
        }

        // Epsilon Annealing
        // If loss is stagnant or improving, we try to lower eps to allow more signal
        if (epoch > 50 && epoch % 10 == 0) {
             current_eps = std::max(1e-5f, current_eps * 0.95f);
             model.set_epsilon(current_eps);
        }

        // Cycle through a fixed "dataset" of seeds
        generate_wavelet_batch(input, (epoch % dataset_size) * batch_size);
        target = input;

        // Forward
        Tensor<float> output = model.forward(input);

        // Loss (MSE)
        float loss = 0;
        float mae = 0; // MAE accumulator
        size_t total_elements = output.size();
        const float* out_ptr = output.data();
        const float* tgt_ptr = target.data();

        Tensor<float> grad_output(output.shape());
        float* go_ptr = grad_output.data();

        #pragma omp parallel for reduction(+:loss, mae)
        for(size_t i=0; i<total_elements; ++i) {
            float diff = out_ptr[i] - tgt_ptr[i];
            loss += diff * diff;
            mae += std::abs(diff); // Compute MAE
            go_ptr[i] = 2.0f * diff / total_elements;
        }
        loss /= total_elements;
        mae /= total_elements;

        // Add Gradient Loss (0.1 weight as requested)
        float grad_loss_val = accumulate_gradient_loss(grad_output, output, target, 0.1f);
        loss += grad_loss_val;

        // Backward
        optimizer.zero_grad();
        model.backward(grad_output);

        // Update
        optimizer.step();

        // Regularizer
        model.apply_lasso(current_lambda, lr);

        // Monitoring
        if (epoch % 50 == 0 || epoch == epochs - 1) {
            size_t total_blocks, zero_blocks;
            model.get_sparsity_stats(total_blocks, zero_blocks);
            float sparsity = (float)zero_blocks / (total_blocks + 1e-9f);

            std::cout << "Epoch " << std::setw(4) << epoch
                      << " | Loss (MSE): " << loss
                      << " | MAE: " << mae
                      << " | Lambda: " << current_lambda
                      << " | Eps: " << current_eps
                      << " | Sparsity: " << (sparsity * 100.0f) << "%" << std::endl;

            // Save Ablation Images (Use Fixed Visualization Batch)
            Tensor<float> vis_output = model.forward(vis_input);
            std::string suffix = std::to_string(epoch);
            save_tensor_as_png(vis_input, "ablation_target_" + suffix + ".png", 0);
            save_tensor_as_png(vis_output, "ablation_recon_" + suffix + ".png", 0);

            // Analyze Latent Space
            Tensor<float> latent = model.forward_encoder(vis_input);
            float lat_min = 1e9, lat_max = -1e9, lat_sum = 0, lat_abs_sum = 0;
            size_t lat_zeros = 0;
            const float* l_ptr = latent.data();
            for(size_t i=0; i<latent.size(); ++i) {
                float v = l_ptr[i];
                if (v < lat_min) lat_min = v;
                if (v > lat_max) lat_max = v;
                lat_sum += v;
                lat_abs_sum += std::abs(v);
                if (std::abs(v) < 1e-9f) lat_zeros++;
            }
            float lat_mean = lat_sum / latent.size();
            float lat_sparsity = (float)lat_zeros / latent.size();

            std::cout << "Latent Stats | Range: [" << lat_min << ", " << lat_max
                      << "] | Mean: " << lat_mean << " | Sparsity: " << (lat_sparsity*100.0f) << "%" << std::endl;

            // Visualize Latent (Reshape 2x2x64 -> 16x16 or similar for visual)
            // Actually it's [N, 2, 2, 64].
            // We can just dump it as a 1D strip or small block.
            // Let's just use the standard saver which will handle it as 2x2 image with 64 channels (saving first 3).
            // That's not very useful.
            // Let's save it as a flattened heatmap if possible, but for now just log stats is enough for analysis.
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_time = end_time - start_time;
    std::cout << "Training Time: " << train_time.count() << "s" << std::endl;

    return 0;
}
