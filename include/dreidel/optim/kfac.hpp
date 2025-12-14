#pragma once

#include <vector>
#include "../core/tensor.hpp"
#include "../layers/layer.hpp"

namespace dreidel {
namespace optim {

/**
 * @brief Kronecker-Factored Approximate Curvature (KFAC) Optimizer.
 *
 * Uses second-order information to precondition gradients.
 * Requires storing and inverting covariance matrices of activations and gradients.
 */
class KFAC {
public:
    KFAC(float learning_rate, float damping = 0.001)
        : lr_(learning_rate), damping_(damping) {}

    /**
     * @brief Update weights.
     *
     * @param layers List of layers to optimize.
     */
    template <typename T>
    void step(std::vector<std::shared_ptr<layers::Layer<T>>>& layers) {
        // TODO:
        // 1. Accumulate curvature stats (A and G matrices) from layers.
        // 2. Periodically invert these matrices (or their factors).
        // 3. Precondition gradients: grad = inv(Curv) * grad.
        // 4. Apply standard update: w = w - lr * grad.
    }

private:
    float lr_;
    float damping_;
    // TODO: Storage for curvature matrices
};

} // namespace optim
} // namespace dreidel
