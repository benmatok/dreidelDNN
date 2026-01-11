#pragma once

#include "Layer.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "../hal/ops.hpp"
#include "../hal/x86.hpp"

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

                // Using standard sqrt as approximate rsqrt showed no consistent gain and potential regression
                T inv_std = 1.0f / std::sqrt(var + eps_);
                inv_std_ptr[n * num_groups_ + g] = inv_std;

                // Normalize and Apply Affine
                // Vectorization here is beneficial if C_per_G is large, but loop is over HW.
                // Inner loop is C. If C_per_G is e.g. 64/32=2, it's tiny.
                // If C=512, G=32, C_per_G=16. Vectorizable.
                for (size_t hw = 0; hw < spatial_size; ++hw) {
                    const T* pixel_in = in_ptr + ((n * spatial_size) + hw) * C;
                    T* pixel_out = out_ptr + ((n * spatial_size) + hw) * C;

#ifdef DREIDEL_ARCH_AVX2
                    // Only apply AVX2 if chunk is large enough to matter
                    size_t c = start_c;
                    if (C_per_G >= 8) {
                        __m256 v_mean = _mm256_set1_ps(mean);
                        __m256 v_inv_std = _mm256_set1_ps(inv_std);
                        for (; c + 7 < end_c; c += 8) {
                             __m256 x = _mm256_loadu_ps(pixel_in + c);
                             __m256 gamma = _mm256_loadu_ps(g_ptr + c);
                             __m256 beta = _mm256_loadu_ps(b_ptr + c);

                             __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x, v_mean), v_inv_std);
                             __m256 res = _mm256_add_ps(_mm256_mul_ps(norm, gamma), beta);

                             _mm256_storeu_ps(pixel_out + c, res);
                        }
                    }
                    for (; c < end_c; ++c) {
                        T norm = (pixel_in[c] - mean) * inv_std;
                        pixel_out[c] = norm * g_ptr[c] + b_ptr[c];
                    }
#else
                    for (size_t c = start_c; c < end_c; ++c) {
                        T norm = (pixel_in[c] - mean) * inv_std;
                        pixel_out[c] = norm * g_ptr[c] + b_ptr[c];
                    }
#endif
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

        #pragma omp parallel
        {
            std::vector<T> local_grad_gamma(C, 0.0);
            std::vector<T> local_grad_beta(C, 0.0);

            #pragma omp for collapse(2)
            for (size_t n = 0; n < N; ++n) {
                for (size_t g = 0; g < num_groups_; ++g) {
                    size_t start_c = g * C_per_G;
                    size_t end_c = start_c + C_per_G;
                    T mean = mean_ptr[n * num_groups_ + g];
                    T inv_std = inv_std_ptr[n * num_groups_ + g];
                    T M = static_cast<T>(spatial_size * C_per_G);

                    T sum_dy_gamma = 0;
                    T sum_dy_gamma_xhat = 0;

                    for (size_t hw = 0; hw < spatial_size; ++hw) {
                        size_t offset = ((n * spatial_size) + hw) * C;
                        const T* pi = in_ptr + offset;
                        const T* pgo = go_ptr + offset;

                        for (size_t c = start_c; c < end_c; ++c) {
                            T dy = pgo[c];
                            T x_hat = (pi[c] - mean) * inv_std;

                            local_grad_gamma[c] += dy * x_hat;
                            local_grad_beta[c] += dy;

                            T dy_gamma = dy * g_ptr[c];
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

            #pragma omp critical
            {
                for (size_t c = 0; c < C; ++c) {
                    gg_ptr[c] += local_grad_gamma[c];
                    gb_ptr[c] += local_grad_beta[c];
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

    void set_epsilon(T eps) {
        eps_ = eps;
    }

private:
    size_t num_groups_;
    size_t num_channels_;
    T eps_;
    Tensor<T> gamma_;
    Tensor<T> beta_;
    Tensor<T> grad_gamma_;
    Tensor<T> grad_beta_;

    Tensor<T> input_cached_;
    Tensor<T> mean_cached_;
    Tensor<T> inv_std_cached_;
};

} // namespace layers
} // namespace dreidel
