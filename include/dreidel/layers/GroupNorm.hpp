#pragma once

#include "Layer.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace dreidel {
namespace layers {

template <typename T>
class GroupNorm : public Layer<T> {
public:
    GroupNorm(size_t num_groups, size_t num_channels, T eps = 1e-5)
        : num_groups_(num_groups), num_channels_(num_channels), eps_(eps),
          gamma_({1, num_channels}), beta_({1, num_channels}),
          grad_gamma_({1, num_channels}), grad_beta_({1, num_channels})
    {
        if (num_channels % num_groups != 0) {
            throw std::invalid_argument("GroupNorm: num_channels must be divisible by num_groups");
        }
        gamma_.fill(1.0);
        beta_.fill(0.0);
        grad_gamma_.fill(0.0);
        grad_beta_.fill(0.0);
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        // Input: N, H, W, C
        auto shape = input.shape();
        size_t N = shape[0];
        size_t H = shape[1];
        size_t W = shape[2];
        size_t C = shape[3];

        if (C != num_channels_) throw std::runtime_error("GroupNorm channel mismatch");

        input_cached_ = input; // Cache for backward

        Tensor<T> output(shape);
        T* out_ptr = output.data();
        const T* in_ptr = input.data();
        const T* g_ptr = gamma_.data();
        const T* b_ptr = beta_.data();

        size_t C_per_G = C / num_groups_;
        size_t spatial_size = H * W;

        // We need to cache mean and inv_std for backward
        // Shape: [N, G]
        if (mean_cached_.shape().size() == 0 || mean_cached_.shape()[0] != N) {
            mean_cached_ = Tensor<T>({N, num_groups_});
            inv_std_cached_ = Tensor<T>({N, num_groups_});
        }
        T* mean_ptr = mean_cached_.data();
        T* inv_std_ptr = inv_std_cached_.data();

        #pragma omp parallel for
        for (size_t n = 0; n < N; ++n) {
            for (size_t g = 0; g < num_groups_; ++g) {
                // Compute Mean
                T sum = 0;
                size_t start_c = g * C_per_G;
                size_t end_c = start_c + C_per_G;

                // Loop over spatial and channels in group
                // Optim: This iteration pattern is strided (channels last)
                for (size_t hw = 0; hw < spatial_size; ++hw) {
                    const T* pixel = in_ptr + ((n * spatial_size) + hw) * C;
                    for (size_t c = start_c; c < end_c; ++c) {
                        sum += pixel[c];
                    }
                }
                T mean = sum / (spatial_size * C_per_G);
                mean_ptr[n * num_groups_ + g] = mean;

                // Compute Variance
                T sum_sq = 0;
                for (size_t hw = 0; hw < spatial_size; ++hw) {
                    const T* pixel = in_ptr + ((n * spatial_size) + hw) * C;
                    for (size_t c = start_c; c < end_c; ++c) {
                        T diff = pixel[c] - mean;
                        sum_sq += diff * diff;
                    }
                }
                T var = sum_sq / (spatial_size * C_per_G);
                T inv_std = 1.0f / std::sqrt(var + eps_);
                inv_std_ptr[n * num_groups_ + g] = inv_std;

                // Normalize and Apply Affine
                for (size_t hw = 0; hw < spatial_size; ++hw) {
                    const T* pixel_in = in_ptr + ((n * spatial_size) + hw) * C;
                    T* pixel_out = out_ptr + ((n * spatial_size) + hw) * C;
                    for (size_t c = start_c; c < end_c; ++c) {
                        T norm = (pixel_in[c] - mean) * inv_std;
                        pixel_out[c] = norm * g_ptr[c] + b_ptr[c];
                    }
                }
            }
        }

        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        auto shape = input_cached_.shape();
        size_t N = shape[0]; size_t H = shape[1]; size_t W = shape[2]; size_t C = shape[3];
        size_t C_per_G = C / num_groups_;
        size_t spatial_size = H * W;

        grad_gamma_.fill(0);
        grad_beta_.fill(0);

        Tensor<T> grad_input(shape);
        T* gi_ptr = grad_input.data();
        const T* go_ptr = grad_output.data();
        const T* in_ptr = input_cached_.data();
        const T* g_ptr = gamma_.data();

        T* gg_ptr = grad_gamma_.data();
        T* gb_ptr = grad_beta_.data();

        const T* mean_ptr = mean_cached_.data();
        const T* inv_std_ptr = inv_std_cached_.data();

        // Accumulate gradients for Gamma and Beta
        // And compute gradient wrt input
        // Standard GroupNorm Backward logic

        #pragma omp parallel for
        for (size_t n = 0; n < N; ++n) {
            for (size_t g = 0; g < num_groups_; ++g) {
                size_t start_c = g * C_per_G;
                size_t end_c = start_c + C_per_G;
                T mean = mean_ptr[n * num_groups_ + g];
                T inv_std = inv_std_ptr[n * num_groups_ + g];

                T d_std = 0;
                T d_mean = 0;

                // First pass: dL/dGamma, dL/dBeta, and accumulate dL/dx_hat
                // Also compute dL/dVar and dL/dMean helpers

                for (size_t hw = 0; hw < spatial_size; ++hw) {
                    size_t offset = ((n * spatial_size) + hw) * C;
                    const T* pi = in_ptr + offset;
                    const T* pgo = go_ptr + offset;

                    for (size_t c = start_c; c < end_c; ++c) {
                        T x_hat = (pi[c] - mean) * inv_std;
                        T dy = pgo[c];

                        // Grads for affine
                        // Need atomic or reduction?
                        // We are in parallel per N, G.
                        // Gamma/Beta are global [C].
                        // Multiple threads write to same Gamma[c] if N > 1.
                        // We need atomic add for gamma/beta.
                        #pragma omp atomic
                        gg_ptr[c] += dy * x_hat;
                        #pragma omp atomic
                        gb_ptr[c] += dy;

                        // Backprop through affine to get dx_hat
                        T dx_hat = dy * g_ptr[c];

                        // dL/dVar calculation involves x_hat
                        // dL/dStd = sum(dL/dx_hat * (x-mu)) * -1/std^2
                        // x-mu = x_hat * std
                        // dL/dStd = sum(dL/dx_hat * x_hat * std) * -1/std^2 = -sum(dL/dx_hat * x_hat) / std
                        // dL/dVar = dL/dStd * 0.5 / std

                        d_std += dx_hat * (pi[c] - mean); // This is sum(dL/dx_hat * (x-mu))
                        d_mean += dx_hat;
                    }
                }

                T d_var = d_std * (-0.5f * inv_std * inv_std * inv_std);

                // Second pass: dL/dx
                // dL/dx = dL/dx_hat * dx_hat/dx + dL/dVar * dVar/dx + dL/dMean * dMean/dx

                T M = static_cast<T>(spatial_size * C_per_G);

                // Correction for mean gradient part from variance
                // dL/dMean_total = d_mean * (-inv_std) + dL/dVar * (-2/M * sum(x-mu))
                // sum(x-mu) is 0 by definition. So second term vanishes.
                T d_mean_final = 0;
                // Wait. dL/dx = (1/std) * ( dL/dx_hat - mean(dL/dx_hat) - x_hat * mean(dL/dx_hat * x_hat) ) ?
                // Let's use the explicit expansion.

                T term1 = d_var * 2.0f / M;
                T term2 = 0; // accumulated for mean part

                // Re-iterate to calculate d_mean_total properly
                // dMean/dx = 1/M
                // dVar/dx = 2(x-mu)/M
                // dx_hat/dx = 1/std

                // Actually, let's use the standard formula for BN/GN backward:
                // dx = (1/std) * ( dy*gamma - mean(dy*gamma) - x_hat * mean(dy*gamma*x_hat) )
                // where means are taken over the group.

                // Calculate means of gradients
                T sum_dy_gamma = 0;
                T sum_dy_gamma_xhat = 0;

                 for (size_t hw = 0; hw < spatial_size; ++hw) {
                    size_t offset = ((n * spatial_size) + hw) * C;
                    const T* pi = in_ptr + offset;
                    const T* pgo = go_ptr + offset;
                    for (size_t c = start_c; c < end_c; ++c) {
                        T dy_gamma = pgo[c] * g_ptr[c];
                        T x_hat = (pi[c] - mean) * inv_std;
                        sum_dy_gamma += dy_gamma;
                        sum_dy_gamma_xhat += dy_gamma * x_hat;
                    }
                 }

                 T mean_dy_gamma = sum_dy_gamma / M;
                 T mean_dy_gamma_xhat = sum_dy_gamma_xhat / M;

                 for (size_t hw = 0; hw < spatial_size; ++hw) {
                    size_t offset = ((n * spatial_size) + hw) * C;
                    const T* pi = in_ptr + offset;
                    const T* pgo = go_ptr + offset;
                    T* pgi = gi_ptr + offset;
                    for (size_t c = start_c; c < end_c; ++c) {
                        T dy_gamma = pgo[c] * g_ptr[c];
                        T x_hat = (pi[c] - mean) * inv_std;
                        pgi[c] = inv_std * (dy_gamma - mean_dy_gamma - x_hat * mean_dy_gamma_xhat);
                    }
                 }
            }
        }

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&gamma_, &beta_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_gamma_, &grad_beta_};
    }

    std::string name() const override { return "GroupNorm"; }

private:
    size_t num_groups_;
    size_t num_channels_;
    T eps_;
    Tensor<T> gamma_;
    Tensor<T> beta_;
    Tensor<T> grad_gamma_;
    Tensor<T> grad_beta_;

    // Cache
    Tensor<T> input_cached_;
    Tensor<T> mean_cached_;
    Tensor<T> inv_std_cached_;
};

} // namespace layers
} // namespace dreidel
