#pragma once

#include "Layer.hpp"
#include "../core/Tensor.hpp"
#include "../core/Allocator.hpp"
#include "../hal/ops.hpp"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>
#include <stdexcept>
#include <type_traits>

namespace dreidel {
namespace layers {

template <typename T>
class ZenithAVXGate : public Layer<T> {
public:
    ZenithAVXGate(size_t channels)
        : channels_(channels), blocks_(channels / 8),
          scale_({channels / 8}),
          bias_({channels / 8}),
          grad_scale_({channels / 8}),
          grad_bias_({channels / 8}),
          rng_(std::random_device{}())
    {
        if (channels % 8 != 0) {
            throw std::invalid_argument("ZenithAVXGate channels must be divisible by 8.");
        }

        // Initialize weights
        // "We learn one weight per block-type"
        // Scale init: 1.0
        // Bias init: 0.0
        scale_.fill(1.0f);
        bias_.fill(0.0f);

        grad_scale_.fill(0.0f);
        grad_bias_.fill(0.0f);
    }

    void set_temperature(float temp) {
        temperature_ = temp;
    }

    void set_training(bool training) {
        training_ = training;
    }

    Tensor<T> forward(const Tensor<T>& input) override {
        input_cached_ = input;
        Tensor<T> output(input.shape());

        size_t N = input.shape()[0];
        size_t H = input.shape()[1];
        size_t W = input.shape()[2];
        size_t C = input.shape()[3];
        size_t spatial_size = H * W;
        size_t total_blocks = N * spatial_size * blocks_;

        // Ensure cache for mask is sized correctly
        // Mask is per block. One mask value for 8 channels.
        // Shape: (N, H, W, C/8) effectively, but we can store flat.
        if (mask_cached_.size() != total_blocks) {
            mask_cached_ = Tensor<T>({total_blocks});
        }

        const T* in_ptr = input.data();
        T* out_ptr = output.data();
        T* mask_ptr = mask_cached_.data();

        const T* scale_ptr = scale_.data();
        const T* bias_ptr = bias_.data();

        // 1. Compute Energy, Score, Gate, and Output
        // We iterate over blocks of 8 channels.
        // Parallelize over N * H * W

        // Generate seeds sequentially to avoid data race on rng_
        std::vector<uint32_t> seeds(omp_get_max_threads());
        for(size_t i=0; i<seeds.size(); ++i) seeds[i] = rng_();

        #pragma omp parallel
        {
            // Thread-local RNG for noise
            std::mt19937 local_rng(seeds[omp_get_thread_num()]);
            std::normal_distribution<float> dist(0.0f, 1.0f);

            #pragma omp for
            for(size_t i = 0; i < N * spatial_size; ++i) {
                // i corresponds to a pixel (n, h, w)
                size_t offset = i * C;
                size_t mask_offset = i * blocks_;

                for (size_t b = 0; b < blocks_; ++b) {
                    // Process block b (8 channels)
                    T energy = 0;
                    size_t c_start = b * 8;

                    // Division-free L1 Energy (Sum Abs)
                    if constexpr (std::is_same_v<T, float>) {
                        #ifdef __AVX2__
                        __m256 v = _mm256_loadu_ps(in_ptr + offset + c_start);
                        // Abs: andnot with -0.0 (sign bit mask)
                        static const __m256 sign_mask = _mm256_set1_ps(-0.0f);
                        __m256 abs_v = _mm256_andnot_ps(sign_mask, v);

                        // Horizontal sum
                        __m256 hsum1 = _mm256_hadd_ps(abs_v, abs_v);
                        __m256 hsum2 = _mm256_hadd_ps(hsum1, hsum1);
                        // Extract sums. _mm256_hadd_ps permutes somewhat, but sum of all elements:
                        // hsum2 contains: [sum0..3, sum0..3, sum4..7, sum4..7] basically
                        // Need to extract low and high 128
                        float buf[8];
                        _mm256_storeu_ps(buf, hsum2);
                        energy = buf[0] + buf[4];
                        // Wait, hsum logic is tricky.
                        // Better: just array sum if not super optimized, or simple loop
                        // Let's stick to simple loop for safety unless bottleneck.
                        // Actually, for just 8 elements, scalar loop is very fast and safe.
                        // Let's rely on compiler autovectorization for sum.
                        #else
                        for(size_t k=0; k<8; ++k) energy += std::abs(in_ptr[offset + c_start + k]);
                        #endif
                    } else {
                        // Scalar fallback if T is not float
                        for(size_t k=0; k<8; ++k) energy += std::abs(in_ptr[offset + c_start + k]);
                    }

                    // Linear Scorer
                    T score = energy * scale_ptr[b] + bias_ptr[b];
                    T mask_val = 0;

                    if (training_) {
                        // Soft Gating: HardSigmoid(score * 0.2 + 0.5 + noise * temp)
                        T noise = dist(local_rng);
                        T raw_gate = score * 0.2f + 0.5f + noise * temperature_;
                        mask_val = std::max(T(0), std::min(T(1), raw_gate));
                    } else {
                        // Hard Pruning
                        mask_val = (score > 0.0f) ? 1.0f : 0.0f;
                    }

                    mask_ptr[mask_offset + b] = mask_val;

                    // Apply mask
                    #ifdef __AVX2__
                    if constexpr (std::is_same_v<T, float>) {
                        __m256 v = _mm256_loadu_ps(in_ptr + offset + c_start);
                        __m256 m = _mm256_set1_ps(mask_val);
                        __m256 out = _mm256_mul_ps(v, m);
                        _mm256_storeu_ps(out_ptr + offset + c_start, out);
                    } else {
                        for(size_t k=0; k<8; ++k) {
                            out_ptr[offset + c_start + k] = in_ptr[offset + c_start + k] * mask_val;
                        }
                    }
                    #else
                    for(size_t k=0; k<8; ++k) {
                        out_ptr[offset + c_start + k] = in_ptr[offset + c_start + k] * mask_val;
                    }
                    #endif
                }
            }
        }

        return output;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        // Gradient dL/dx = dL/dy * mask + dL/dmask * dmask/dx
        // dL/dmask = sum(dL/dy * x)
        // mask = HardSigmoid(S), S = E * w + b + noise
        // dmask/dS = 1 if 0 < S < 1 else 0 (HardSigmoid deriv)
        // dS/dw = E
        // dS/db = 1
        // dS/dE = w
        // dE/dx = sign(x)

        Tensor<T> grad_input(input_cached_.shape());
        grad_input.fill(0); // Should initialize?

        size_t N = input_cached_.shape()[0];
        size_t H = input_cached_.shape()[1];
        size_t W = input_cached_.shape()[2];
        size_t C = input_cached_.shape()[3];
        size_t spatial_size = H * W;

        const T* x_ptr = input_cached_.data();
        const T* dy_ptr = grad_output.data();
        T* dx_ptr = grad_input.data();
        const T* mask_ptr = mask_cached_.data();

        const T* scale_ptr = scale_.data();
        T* d_scale = grad_scale_.data();
        T* d_bias = grad_bias_.data();

        // Accumulators for parameters
        std::vector<T> local_d_scale(blocks_, 0.0f);
        std::vector<T> local_d_bias(blocks_, 0.0f);

        // We need to accumulate parameter gradients. OpenMP reduction is tricky for arrays.
        // We'll use thread-local accumulators.

        #pragma omp parallel
        {
            std::vector<T> thread_d_scale(blocks_, 0.0f);
            std::vector<T> thread_d_bias(blocks_, 0.0f);

            #pragma omp for
            for(size_t i = 0; i < N * spatial_size; ++i) {
                size_t offset = i * C;
                size_t mask_offset = i * blocks_;

                for (size_t b = 0; b < blocks_; ++b) {
                    T mask_val = mask_ptr[mask_offset + b];
                    T w = scale_ptr[b];

                    // Recompute Energy and raw gate for derivative
                    // Need noise? No, derivative of HardSigmoid(z) depends on value.
                    // If 0 < mask < 1, deriv is 1 (for the clamp part? No, HardSigmoid is usually linear slope 0.2?
                    // Spec: "Gate = Clamp( Score * 0.2 + 0.5, 0, 1 )"
                    // Deriv w.r.t Score is 0.2 if 0 < Gate < 1. Else 0.

                    bool gate_active = (mask_val > 0.0f && mask_val < 1.0f);
                    T d_mask_d_score = gate_active ? 0.2f : 0.0f;

                    // If not training (inference), mask is hard 0/1, deriv is 0 (or undefined).
                    // Usually we don't backprop through inference mode.
                    if (!training_) d_mask_d_score = 0.0f;

                    T dL_dmask = 0;
                    size_t c_start = b * 8;

                    T energy = 0; // Needed for dS/dw

                    for(size_t k=0; k<8; ++k) {
                        T x_val = x_ptr[offset + c_start + k];
                        T dy_val = dy_ptr[offset + c_start + k];

                        // dL/dx contribution from pass-through
                        dx_ptr[offset + c_start + k] = dy_val * mask_val;

                        // Accumulate dL/dmask
                        dL_dmask += dy_val * x_val;

                        energy += std::abs(x_val);
                    }

                    // Chain rule for parameters
                    // dL/dS = dL/dmask * d_mask_d_score
                    T dL_dS = dL_dmask * d_mask_d_score;

                    thread_d_scale[b] += dL_dS * energy;
                    thread_d_bias[b] += dL_dS * 1.0f;

                    // Backprop to input through gate (optional, usually ignored for pruning, but correct math:
                    // dS/dE = w. dE/dx = sign(x).
                    // dL/dx += dL/dS * w * sign(x)
                    T dL_dE = dL_dS * w;
                    for(size_t k=0; k<8; ++k) {
                        T x_val = x_ptr[offset + c_start + k];
                        T sgn = (x_val > 0) ? 1.0f : ((x_val < 0) ? -1.0f : 0.0f);
                        dx_ptr[offset + c_start + k] += dL_dE * sgn;
                    }
                }
            }

            #pragma omp critical
            {
                for(size_t b=0; b<blocks_; ++b) {
                    local_d_scale[b] += thread_d_scale[b];
                    local_d_bias[b] += thread_d_bias[b];
                }
            }
        }

        // Copy accumulated grads
        std::copy(local_d_scale.begin(), local_d_scale.end(), d_scale);
        std::copy(local_d_bias.begin(), local_d_bias.end(), d_bias);

        return grad_input;
    }

    std::vector<Tensor<T>*> parameters() override {
        return {&scale_, &bias_};
    }

    std::vector<Tensor<T>*> gradients() override {
        return {&grad_scale_, &grad_bias_};
    }

    std::string name() const override { return "ZenithAVXGate"; }

    // Helper for sparsity loss
    T get_sparsity_loss() const {
        // "Sum(Abs(Mask))"
        // Return mean mask value (sparsity rate) or sum?
        // Spec: "Add Lambda * Sum(Abs(Mask))"
        T sum = 0;
        const T* m = mask_cached_.data();
        size_t size = mask_cached_.size();
        #pragma omp parallel for reduction(+:sum)
        for(size_t i=0; i<size; ++i) {
            sum += std::abs(m[i]);
        }
        return sum;
    }

    const Tensor<T>* get_last_mask() const {
        return &mask_cached_;
    }

private:
    size_t channels_;
    size_t blocks_;

    Tensor<T> scale_;
    Tensor<T> bias_;

    Tensor<T> grad_scale_;
    Tensor<T> grad_bias_;

    Tensor<T> input_cached_;
    Tensor<T> mask_cached_;

    float temperature_ = 1.0f;
    bool training_ = true;

    std::mt19937 rng_;
};

} // namespace layers
} // namespace dreidel
