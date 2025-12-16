#ifndef DREIDEL_LAYERS_LINEAR_WHT_HPP
#define DREIDEL_LAYERS_LINEAR_WHT_HPP

#include "Layer.hpp"
#include "../algo/WHT.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class LinearWHT : public Layer<T, B> {
public:
    // dim must be a power of 2
    // k is for TopK sparsity. If k=0 or k>=dim, no sparsity is applied.
    LinearWHT(size_t dim, size_t k = 0)
        : dim_(dim), k_(k),
          scale_({1, dim}), grad_scale_({1, dim})
    {
        // Check power of 2
        if (dim == 0 || (dim & (dim - 1)) != 0) {
            throw std::invalid_argument("LinearWHT dimension must be a power of 2.");
        }

        // Initialize scale (D)
        // D is a diagonal matrix.
        // We want to preserve variance.
        // FWHT increases variance by N.
        // So we might want D to be small?
        // Let's try initializing around 1.0/N or similar, or just standard normal.
        // For now, let's use Xavier-like but for diagonal.
        // stddev = sqrt(2 / (dim + dim)) = 1 / sqrt(dim).
        // But FWHT multiplies by N (in magnitude sum) or sqrt(N) in Euclidean?
        // Unnormalized FWHT: vector length scales by sqrt(N).
        // So to keep variance, we need to divide by sqrt(N).
        // Combined with Xavier: 1/sqrt(N) * 1/sqrt(N) = 1/N?
        // Let's stick to small random values.

        // Using 1/N might be too small if we don't normalize FWHT.
        // Let's try Normal(0, 1.0 / dim).
        scale_.random(0, 1.0 / static_cast<T>(dim));
        grad_scale_.fill(0);
    }

    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        // Input: (Batch, N)
        // Store input for backward
        input_ = input;

        // 1. Element-wise multiply by D (Broadcasting)
        // z = x * scale
        Tensor<T, B> z = input * scale_;

        // 2. FWHT in-place
        algo::WHT::FWHT(z);

        // 3. TopK (Optional)
        if (k_ > 0 && k_ < dim_) {
            // Apply TopK mask
            // This is tricky to do efficiently in parallel/SIMD without specific kernel.
            // For now, simple implementation: iterate rows, find k-th largest abs value, zero out others.
            // Also need to store mask for backward.
            mask_ = Tensor<T, B>(z.shape());
            mask_.fill(0);

            size_t batch_size = z.shape()[0];
            T* data = z.data();
            T* mask_data = mask_.data();

            // Parallel over batch
            #pragma omp parallel for
            for (long i = 0; i < (long)batch_size; ++i) {
                size_t offset = i * dim_;
                std::vector<std::pair<T, size_t>> vals(dim_);
                for (size_t j = 0; j < dim_; ++j) {
                    vals[j] = {std::abs(data[offset + j]), j};
                }

                // Partial sort to find top k
                std::nth_element(vals.begin(), vals.begin() + k_, vals.end(),
                                 [](const auto& a, const auto& b) {
                                     return a.first > b.first;
                                 });

                // Keep top k, mask others
                // The nth_element puts the k-th largest at position k (0-indexed? No, it puts top k in 0..k-1).
                // Wait, nth_element sorts such that element at n is in correct position.
                // Elements before n are <= (or >=) element at n.
                // We want largest. So logic `a.first > b.first`.
                // Elements 0 to k-1 are the top k.

                for (size_t j = 0; j < k_; ++j) {
                    size_t idx = vals[j].second;
                    mask_data[offset + idx] = 1;
                }

                // Apply mask to data
                for (size_t j = 0; j < dim_; ++j) {
                    if (mask_data[offset + j] == 0) {
                        data[offset + j] = 0;
                    }
                }
            }
        }

        return z;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // grad_output: dL/dy
        Tensor<T, B> grad_z = grad_output;

        // 1. Backprop through TopK
        if (k_ > 0 && k_ < dim_) {
             grad_z = grad_z * mask_;
        }

        // 2. Backprop through FWHT
        // FWHT is symmetric.
        algo::WHT::FWHT(grad_z);

        // 3. Backprop through scaling (z = x * scale)
        // dL/dScale = sum(dL/dz * x, axis=0)
        // dL/dx = dL/dz * scale

        // Compute dL/dScale
        // (Batch, N) * (Batch, N) -> sum -> (1, N)
        // Or (Batch, D, H, W, C) * (Batch, D, H, W, C) -> sum -> (1, C)
        // We can do element-wise mul then reduce over all axes except last.
        Tensor<T, B> elem_grads = grad_z * input_;

        // Manual reduction to handle arbitrary dimensions
        size_t last_dim = elem_grads.shape().back();
        size_t total_elements = elem_grads.size();
        size_t outer_dims = total_elements / last_dim;

        grad_scale_.fill(0);
        T* grad_s_ptr = grad_scale_.data();
        const T* elem_ptr = elem_grads.data();

        // Accumulate gradients
        // This can be parallelized but needs thread-safe accumulation for grad_s_ptr.
        // For simplicity and safety (since C is small, typically 64-1024), we can iterate outer first.

        // Or transpose-like sum:
        // For each channel c, sum over all outer_dims.

        // Use temporary buffer for thread local sums if we want parallel.
        // Or just use parallel for over channels? (C is usually small).
        // Or parallel for over outer dims with reduction.

        // Simple OMP reduction if C is small enough to fit in cache lines efficiently?
        // Actually, if we loop over outer_dims, we access memory linearly.

        std::vector<T> temp_sum(last_dim, 0);

        // Since we can't easily reduce array in OMP without custom reducer in C++17 (requires user defined reduction)
        // We will use a critical section or just serial sum for now if outer_dims is small.
        // If outer_dims is large, this is slow.

        // Better: Use atomics if T=float?
        // Or accumulate per thread and reduce.

        #pragma omp parallel
        {
            std::vector<T> local_sum(last_dim, 0);

            #pragma omp for
            for (long i = 0; i < (long)outer_dims; ++i) {
                size_t offset = i * last_dim;
                for (size_t j = 0; j < last_dim; ++j) {
                    local_sum[j] += elem_ptr[offset + j];
                }
            }

            #pragma omp critical
            {
                for (size_t j = 0; j < last_dim; ++j) {
                    grad_s_ptr[j] += local_sum[j];
                }
            }
        }

        // Compute dL/dx
        // grad_z is arbitrary shape, scale_ is (1, C)
        // operator* now supports broadcasting scale_ to grad_z
        Tensor<T, B> grad_input = grad_z * scale_;

        return grad_input;
    }

    std::vector<Tensor<T, B>*> parameters() override {
        return {&scale_};
    }

    std::vector<Tensor<T, B>*> gradients() override {
        return {&grad_scale_};
    }

    std::string name() const override { return "LinearWHT"; }

private:
    size_t dim_;
    size_t k_;
    Tensor<T, B> scale_;
    Tensor<T, B> grad_scale_;

    // Cache
    Tensor<T, B> input_;
    Tensor<T, B> mask_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_LINEAR_WHT_HPP
