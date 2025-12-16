#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "dreidel/core/Tensor.hpp"
#include "dreidel/optim/DiagonalNewton.hpp"

using namespace dreidel;

// Test on Diagonal Quadratic Function (Ill-conditioned)
// f(x, y) = 0.5 * (a*x^2 + b*y^2)
// Gradient: [a*x, b*y]
// Hessian: Diag([a, b])
//
// With SGD, this takes many steps if b >> a.
// With Diagonal Newton, it should take 1 step.

void test_diagonal_quadratic() {
    std::cout << "Running Diagonal Quadratic Test (Ill-conditioned)..." << std::endl;

    // Parameters (High Condition Number)
    float a = 1.0f;
    float b = 1000.0f;

    // Initial Guess
    Tensor<float> params({2});
    params.data()[0] = 10.0f;
    params.data()[1] = 10.0f;

    // Containers
    Tensor<float> grads({2});
    Tensor<float> curvs({2});

    // Optimizer with LR=1.0 (Exact Newton Step)
    optim::DiagonalNewton<float> optimizer(1.0f);
    optimizer.add_parameters({&params}, {&grads}, {&curvs});

    int steps = 0;
    float tolerance = 1e-6;
    bool converged = false;

    std::cout << "Initial Loss: " << 0.5f * (a*100 + b*100) << std::endl;

    for (steps = 0; steps < 100; ++steps) {
        float x = params.data()[0];
        float y = params.data()[1];

        // Loss
        float loss = 0.5f * (a * x * x + b * y * y);

        if (loss < tolerance) {
            converged = true;
            break;
        }

        // Gradients
        grads.data()[0] = a * x;
        grads.data()[1] = b * y;

        // Curvatures (Exact Hessian Diagonal)
        curvs.data()[0] = a;
        curvs.data()[1] = b;

        // Step
        optimizer.step();

        std::cout << "Step " << steps+1 << ": Loss=" << loss << std::endl;
    }

    if (converged) {
        std::cout << "Converged in " << steps << " steps." << std::endl;
        std::cout << "Final Position: (" << params.data()[0] << ", " << params.data()[1] << ")" << std::endl;
        if (steps <= 2) {
             std::cout << "PASS (Instant Convergence)" << std::endl;
        } else {
             std::cout << "PASS (Converged)" << std::endl;
        }
    } else {
        std::cout << "FAIL: Did not converge." << std::endl;
        exit(1);
    }
}

int main() {
    test_diagonal_quadratic();
    return 0;
}
