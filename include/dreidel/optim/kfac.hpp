#pragma once

#include <vector>
#include <map>
#include <iostream>
#include "../core/Tensor.hpp"
#include "../layers/Layer.hpp"
#include "BlockDiagonal.hpp" // For invert_matrix

namespace dreidel {
namespace optim {

/**
 * @brief Kronecker-Factored Approximate Curvature (KFAC) Optimizer.
 *
 * Uses second-order information to precondition gradients.
 * Approximates the Fisher Information Matrix F as A (x) G.
 * A = E[aa^T] (activations covariance)
 * G = E[gg^T] (gradient covariance)
 *
 * Update rule: vec(W) = vec(W) - lr * (A^-1 (x) G^-1) * vec(grad)
 * Equivalent to: W = W - lr * A^-1 * grad * G^-1
 */
template <typename T>
class KFAC {
public:
    KFAC(T learning_rate, T damping = 0.001, T stat_decay = 0.95)
        : lr_(learning_rate), damping_(damping), stat_decay_(stat_decay) {}

    /**
     * @brief Update weights.
     *
     * @param layers List of layers to optimize.
     */
    void step(const std::vector<layers::Layer<T>*>& layers) {
        step_count_++;

        for (auto* layer : layers) {
            auto activations = layer->activations();
            auto grad_outputs = layer->grad_outputs();
            auto params = layer->parameters();
            auto grads = layer->gradients();

            // KFAC currently supports only Dense-like layers with 1 input and 1 output for A/G calc
            // and params [Weights, Bias] or [Weights].
            if (activations.empty() || grad_outputs.empty() || params.empty() || grads.empty()) {
                // Fallback to SGD if KFAC info not available
                // Apply standard SGD update: w = w - lr * grad
                for(size_t i=0; i<params.size(); ++i) {
                     T* p = params[i]->data();
                     const T* g = grads[i]->data();
                     size_t size = params[i]->size();
                     for(size_t k=0; k<size; ++k) {
                         p[k] -= lr_ * g[k];
                     }
                }
                continue;
            }

            // Assume activations[0] is 'a' (Batch, In)
            // Assume grad_outputs[0] is 'g' (Batch, Out)
            // Assume params[0] is Weights (In, Out)
            // Assume params[1] is Bias (1, Out) if exists

            Tensor<T>* a = activations[0];
            Tensor<T>* g = grad_outputs[0];
            Tensor<T>* w = params[0];
            Tensor<T>* w_grad = grads[0];

            // 1. Update Stats A and G
            update_stats(layer, *a, *g);

            // 2. Invert Factors periodically
            if (step_count_ % inverse_update_freq_ == 0) {
                compute_inverses(layer);
            }

            // 3. Precondition Gradient
            // grad_pre = A_inv * grad * G_inv
            // If bias exists, we usually append 1 to 'a' for A calculation, making A (In+1, In+1).
            // But Dense layer separates W and b.
            // Standard KFAC treats [W; b^T] as a single matrix.
            // Here we treat W separately for simplicity or we need to handle bias.

            // Current simplification: Apply KFAC to Weights only. Bias uses SGD.
            // TODO: Proper bias handling by appending 1 to activations.

            // Get Inverses
            if (cov_A_inv_.find(layer) == cov_A_inv_.end()) {
                // Initialize if not ready (first few steps)
                 compute_inverses(layer);
            }

            const Tensor<T>& A_inv = cov_A_inv_[layer];
            const Tensor<T>& G_inv = cov_G_inv_[layer];

            // Precondition W gradient: V = A_inv * W_grad * G_inv
            // W_grad: (In, Out)
            // A_inv: (In, In)
            // G_inv: (Out, Out)

            // Check dims
            if (A_inv.shape()[1] != w_grad->shape()[0] || w_grad->shape()[1] != G_inv.shape()[0]) {
                // Dimension mismatch (maybe bias handling issue), fallback to SGD
                // std::cerr << "KFAC dimension mismatch. Fallback to SGD." << std::endl;
                 T* p = w->data();
                 const T* gr = w_grad->data();
                 for(size_t k=0; k<w->size(); ++k) p[k] -= lr_ * gr[k];
            } else {
                Tensor<T> pre_grad = A_inv.matmul(*w_grad).matmul(G_inv);

                // Update W
                T* p = w->data();
                const T* pg = pre_grad.data();
                for(size_t k=0; k<w->size(); ++k) {
                    p[k] -= lr_ * pg[k]; // KFAC update
                    // p[k] -= lr_ * gr[k]; // SGD Update (for debugging)
                }
            }

            // Update Bias (SGD)
            if (params.size() > 1) {
                Tensor<T>* b = params[1];
                Tensor<T>* b_grad = grads[1];
                T* p = b->data();
                const T* gr = b_grad->data();
                for(size_t k=0; k<b->size(); ++k) {
                    p[k] -= lr_ * gr[k];
                }
            }
        }
    }

    // Support overloaded add_parameters call from Autoencoder example which just passes vectors
    // But KFAC needs Layer structure to pair activations with gradients.
    // So this method is not really usable for KFAC logic unless we map params back to layers.
    // We will assume the user calls step(layers).

    void set_inverse_update_freq(int freq) { inverse_update_freq_ = freq; }

private:
    T lr_;
    T damping_;
    T stat_decay_; // Moving average decay (e.g. 0.95)
    int step_count_ = 0;
    int inverse_update_freq_ = 20;

    // Storage for covariance matrices per layer
    std::map<layers::Layer<T>*, Tensor<T>> cov_A_;
    std::map<layers::Layer<T>*, Tensor<T>> cov_G_;

    std::map<layers::Layer<T>*, Tensor<T>> cov_A_inv_;
    std::map<layers::Layer<T>*, Tensor<T>> cov_G_inv_;

    void update_stats(layers::Layer<T>* layer, const Tensor<T>& a, const Tensor<T>& g) {
        // Compute current batch A = a^T * a / Batch
        // a: (Batch, In) -> A: (In, In)
        // g: (Batch, Out) -> G: g^T * g / Batch
        // G: (Out, Out)

        T batch_size = static_cast<T>(a.shape()[0]);
        T scale = 1.0 / batch_size;

        Tensor<T> A_batch = a.transpose().matmul(a); // (In, In)

        // Scale manually since Tensor doesn't have scalar mul (Wait, it has op*)
        // Wait, Tensor operator* is element-wise.
        // We need to implement scaling or do it element-wise.
        // Assuming tensor has broadcasting or iterate.
        T* ptr_A = A_batch.data();
        for(size_t i=0; i<A_batch.size(); ++i) ptr_A[i] *= scale;

        Tensor<T> G_batch = g.transpose().matmul(g); // (Out, Out)
        T* ptr_G = G_batch.data();
        for(size_t i=0; i<G_batch.size(); ++i) ptr_G[i] *= scale; // * batch_size?
        // Actually Fisher def is E[grad grad^T]. If loss is mean, grad scales 1/B.
        // Standard KFAC usually uses just g g^T * scale.

        if (cov_A_.find(layer) == cov_A_.end()) {
            cov_A_[layer] = A_batch;
            cov_G_[layer] = G_batch;
        } else {
            // Moving average: S_new = decay * S_old + (1-decay) * S_batch
            T alpha = stat_decay_;
            T beta = 1.0 - alpha;

            Tensor<T>& A = cov_A_[layer];
            T* pA = A.data();
            T* pAb = A_batch.data();
            for(size_t i=0; i<A.size(); ++i) pA[i] = alpha * pA[i] + beta * pAb[i];

            Tensor<T>& G = cov_G_[layer];
            T* pG = G.data();
            T* pGb = G_batch.data();
            for(size_t i=0; i<G.size(); ++i) pG[i] = alpha * pG[i] + beta * pGb[i];
        }
    }

    void compute_inverses(layers::Layer<T>* layer) {
        if (cov_A_.find(layer) == cov_A_.end()) return;

        // Add damping to diagonal
        Tensor<T> A = cov_A_[layer]; // Copy
        Tensor<T> G = cov_G_[layer]; // Copy

        T root_damping = std::sqrt(damping_);

        add_damping(A, root_damping);
        add_damping(G, root_damping);

        // Invert
        cov_A_inv_[layer] = invert(A);
        cov_G_inv_[layer] = invert(G);
    }

    void add_damping(Tensor<T>& M, T val) {
        // M is square (D, D)
        size_t dim = M.shape()[0];
        T* ptr = M.data();
        for(size_t i=0; i<dim; ++i) {
            ptr[i * dim + i] += val;
        }
    }

    Tensor<T> invert(const Tensor<T>& M) {
        // Use Gauss-Jordan from BlockDiagonal.hpp if available or implement simple one.
        // BlockDiagonal.hpp has `invert_matrix`.
        // We need to check if it's accessible.
        // Let's assume we can include it or copy it.
        // Since I included BlockDiagonal.hpp, I can use dreidel::optim::invert_matrix?
        // No, BlockDiagonal might store it as private or helper.
        // Let's implement a simple Gauss-Jordan here to be safe and dependency-free.

        size_t n = M.shape()[0];
        Tensor<T> inv({n, n});
        inv.fill(0);
        T* res = inv.data();
        const T* src = M.data();

        // Initialize res as identity
        for(size_t i=0; i<n; ++i) res[i*n+i] = 1.0;

        // Create augmented matrix logic (work on copy of M)
        std::vector<T> mat(n*n);
        for(size_t i=0; i<n*n; ++i) mat[i] = src[i];

        for (size_t i = 0; i < n; ++i) {
            // Pivot
            T pivot = mat[i * n + i];
            // If singular, add epsilon (simple regularization)
            if (std::abs(pivot) < 1e-6) {
                pivot = 1e-6;
                mat[i * n + i] = pivot;
            }

            // Normalize row i
            T inv_pivot = 1.0 / pivot;
            for (size_t j = 0; j < n; ++j) {
                mat[i * n + j] *= inv_pivot;
                res[i * n + j] *= inv_pivot;
            }

            // Eliminate other rows
            for (size_t k = 0; k < n; ++k) {
                if (k != i) {
                    T factor = mat[k * n + i];
                    for (size_t j = 0; j < n; ++j) {
                        mat[k * n + j] -= factor * mat[i * n + j];
                        res[k * n + j] -= factor * res[i * n + j];
                    }
                }
            }
        }
        return inv;
    }
};

} // namespace optim
} // namespace dreidel
