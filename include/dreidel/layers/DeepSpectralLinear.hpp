#ifndef DREIDEL_LAYERS_DEEP_SPECTRAL_LINEAR_HPP
#define DREIDEL_LAYERS_DEEP_SPECTRAL_LINEAR_HPP

#include "Layer.hpp"
#include "../algo/WHT.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class DeepSpectralLinear : public Layer<T, B> {
public:
    // dim: spectral dimension (power of 2)
    // depth: number of stacked layers (K)
    DeepSpectralLinear(size_t dim, size_t depth = 4)
        : dim_(dim), depth_(depth)
    {
        // Check power of 2
        if (dim == 0 || (dim & (dim - 1)) != 0) {
            throw std::invalid_argument("DeepSpectralLinear dimension must be a power of 2.");
        }

        // Initialize layers
        for (size_t k = 0; k < depth_; ++k) {
            // Scales
            scales_.emplace_back(std::vector<size_t>{1, dim});
            grad_scales_.emplace_back(std::vector<size_t>{1, dim});
            curv_scales_.emplace_back(std::vector<size_t>{1, dim});

            // Initialize scales to be close to 1 (identity flow) or random.
            // FWHT scales energy by N. To preserve variance, we usually scale by 1/sqrt(N) or 1/N.
            // Actually, if we use standard FWHT, it is orthogonal up to scale factor sqrt(N).
            // If we want unitary, we need scale 1/sqrt(N).
            // We initialize with mean 0 and stddev 1/sqrt(dim) to preserve variance across layers.
            T stddev = 1.0 / std::sqrt(static_cast<T>(dim));
            scales_.back().random(0, stddev);

            grad_scales_.back().fill(0);
            curv_scales_.back().fill(0);

            // Permutations
            // Generate random permutation
            std::vector<size_t> p(dim);
            std::iota(p.begin(), p.end(), 0);
            if (k > 0) { // Keep first layer unpermuted? Or all permuted? Let's permute all.
                static std::random_device rd;
                static std::mt19937 g(rd());
                std::shuffle(p.begin(), p.end(), g);
            }
            perms_.push_back(p);

            // Inverse Permutation
            std::vector<size_t> inv_p(dim);
            for (size_t i = 0; i < dim; ++i) {
                inv_p[p[i]] = i;
            }
            inv_perms_.push_back(inv_p);
        }
    }

    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        // Pad input if needed
        if (input.shape().back() < dim_) {
             cached_input_raw_ = input; // Save original for backward padding logic if needed?
             // Actually we just need to know it was padded.
             // But simpler to just work in padded space.
             cached_input_padded_ = input.pad_last_dim(dim_);
        } else if (input.shape().back() > dim_) {
             throw std::runtime_error("Input dimension larger than DeepSpectralLinear dimension.");
        } else {
             cached_input_padded_ = input;
        }

        Tensor<T, B> x = cached_input_padded_;

        // Clear cache
        intermediate_activations_.clear();
        // We store input to each layer k.
        // Input to layer 0 is x.
        intermediate_activations_.push_back(x);

        for (size_t k = 0; k < depth_; ++k) {
            // 1. Permutation
            // x_perm = P(x)
            // We need a helper to permute last dim.
            // Since Tensor doesn't have it, we implement loop.

            Tensor<T, B> x_perm(x.shape());
            permute_forward(x, x_perm, perms_[k]);

            // 2. Scale
            // x_scaled = x_perm * D
            Tensor<T, B> x_scaled = x_perm * scales_[k];

            // 3. FWHT
            algo::WHT::FWHT(x_scaled);

            x = x_scaled;

            // Save output of layer k (which is input to k+1)
            if (k < depth_ - 1) {
                intermediate_activations_.push_back(x);
            }
        }

        return x;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        Tensor<T, B> grad = grad_output;

        for (int k = depth_ - 1; k >= 0; --k) {
            // Reverse operations: FWHT -> Scale -> Permutation

            // 1. FWHT (Symmetric)
            algo::WHT::FWHT(grad);

            // 2. Scale Gradient
            // dL/dScale = grad * Input_to_Scale
            // Input_to_Scale was x_perm in forward pass.
            // We need to reconstruct x_perm.
            // x_in = intermediate_activations_[k]
            // x_perm = P(x_in)

            Tensor<T, B> x_in = intermediate_activations_[k];
            Tensor<T, B> x_perm(x_in.shape());
            permute_forward(x_in, x_perm, perms_[k]);

            // Compute dL/dD
            // Sum over batch
            Tensor<T, B> elem_grads = grad * x_perm;
            reduce_gradients(elem_grads, grad_scales_[k]);

            // Compute Curvature (approx sum(x^2))
            reduce_curvature(x_perm, curv_scales_[k]);

            // 3. Propagate Gradient to Input
            // dL/dx_perm = grad * scale
            Tensor<T, B> grad_perm = grad * scales_[k];

            // 4. Inverse Permutation
            // dL/dx = P^T(dL/dx_perm)
            // grad = InvPerm(grad_perm)
            permute_forward(grad_perm, grad, inv_perms_[k]); // Reuse permute logic with inv_perm
        }

        // If we padded input, we should slice gradient?
        // Usually backprop assumes shape matches forward input.
        // If original input was smaller, we slice.
        if (cached_input_raw_.size() > 0 && cached_input_raw_.shape().back() < dim_) {
            return grad.slice_last_dim(cached_input_raw_.shape().back());
        }

        return grad;
    }

    std::vector<Tensor<T, B>*> parameters() override {
        std::vector<Tensor<T, B>*> params;
        for (auto& s : scales_) params.push_back(&s);
        return params;
    }

    std::vector<Tensor<T, B>*> gradients() override {
        std::vector<Tensor<T, B>*> grads;
        for (auto& g : grad_scales_) grads.push_back(&g);
        return grads;
    }

    std::vector<Tensor<T, B>*> curvatures() override {
        std::vector<Tensor<T, B>*> curvs;
        for (auto& c : curv_scales_) curvs.push_back(&c);
        return curvs;
    }

    std::string name() const override { return "DeepSpectralLinear"; }

    // Manually set permutation for a specific layer index
    void set_permutation(size_t index, const std::vector<size_t>& p) {
        if (index >= depth_) {
            throw std::out_of_range("Layer index out of range in DeepSpectralLinear");
        }
        if (p.size() != dim_) {
            throw std::invalid_argument("Permutation size mismatch");
        }
        perms_[index] = p;

        // Recompute inverse permutation
        for (size_t i = 0; i < dim_; ++i) {
            inv_perms_[index][p[i]] = i;
        }
    }

private:
    void permute_forward(const Tensor<T, B>& input, Tensor<T, B>& output, const std::vector<size_t>& p) {
        // Output shape matches input
        size_t last_dim = dim_;
        size_t total = input.size();
        size_t outer = total / last_dim;

        const T* in_ptr = input.data();
        T* out_ptr = output.data();

        #pragma omp parallel for
        for (long i = 0; i < (long)outer; ++i) {
            size_t offset = i * last_dim;
            for (size_t j = 0; j < last_dim; ++j) {
                out_ptr[offset + j] = in_ptr[offset + p[j]];
            }
        }
    }

    void reduce_gradients(const Tensor<T, B>& elem_grads, Tensor<T, B>& grad_scale) {
        size_t last_dim = dim_;
        size_t total = elem_grads.size();
        size_t outer = total / last_dim;

        const T* ptr = elem_grads.data();
        grad_scale.fill(0);
        T* g_ptr = grad_scale.data();

        #pragma omp parallel
        {
            std::vector<T> local(last_dim, 0);
            #pragma omp for
            for (long i = 0; i < (long)outer; ++i) {
                size_t offset = i * last_dim;
                for (size_t j = 0; j < last_dim; ++j) {
                    local[j] += ptr[offset + j];
                }
            }
            #pragma omp critical
            {
                for (size_t j = 0; j < last_dim; ++j) {
                    g_ptr[j] += local[j];
                }
            }
        }
    }

    void reduce_curvature(const Tensor<T, B>& input, Tensor<T, B>& curv_scale) {
        size_t last_dim = dim_;
        size_t total = input.size();
        size_t outer = total / last_dim;

        const T* ptr = input.data();
        curv_scale.fill(0);
        T* c_ptr = curv_scale.data();

        #pragma omp parallel
        {
            std::vector<T> local(last_dim, 0);
            #pragma omp for
            for (long i = 0; i < (long)outer; ++i) {
                size_t offset = i * last_dim;
                for (size_t j = 0; j < last_dim; ++j) {
                    T v = ptr[offset + j];
                    local[j] += v * v;
                }
            }
            #pragma omp critical
            {
                for (size_t j = 0; j < last_dim; ++j) {
                    c_ptr[j] += local[j];
                }
            }
        }
    }

    size_t dim_;
    size_t depth_;

    std::vector<Tensor<T, B>> scales_;
    std::vector<Tensor<T, B>> grad_scales_;
    std::vector<Tensor<T, B>> curv_scales_;
    std::vector<std::vector<size_t>> perms_;
    std::vector<std::vector<size_t>> inv_perms_;

    // Cache for backward
    Tensor<T, B> cached_input_raw_;
    Tensor<T, B> cached_input_padded_;
    std::vector<Tensor<T, B>> intermediate_activations_;
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_DEEP_SPECTRAL_LINEAR_HPP
