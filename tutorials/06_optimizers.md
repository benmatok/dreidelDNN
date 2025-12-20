# Optimizers: SGD and DiagonalNewton

dreidelDNN provides two main optimizers. Choosing the right one is crucial for convergence, especially with spectral layers.

## SGD (Stochastic Gradient Descent)

Standard SGD updates parameters using the gradient:
$$ w \leftarrow w - \eta \cdot \nabla L $$

### Usage
```cpp
#include <dreidel/optim/SGD.hpp>

float learning_rate = 0.01f;
optim::SGD<float> sgd(learning_rate);

// In training loop
sgd.step(model.parameters(), model.gradients());
```

Best for: Standard layers like `Dense` or `Bias`.

## DiagonalNewton

Spectral layers (`LinearWHT`, `DeepSpectralLinear`) often have ill-conditioned curvature due to the repeated application of FWHT (which scales values). `DiagonalNewton` uses second-order information (curvature) to precondition the gradient, acting like an adaptive learning rate per parameter.

Update rule:
$$ w \leftarrow w - \eta \frac{\nabla L}{|h| + \epsilon} $$
Where $h$ is the diagonal curvature estimate.

### Usage
```cpp
#include <dreidel/optim/DiagonalNewton.hpp>

float learning_rate = 1.0f; // Often needs to be higher (1.0 - 10.0)
float epsilon = 1e-4f;
optim::DiagonalNewton<float> newton(learning_rate, epsilon);

// In training loop
// Note: Requires passing curvatures()
newton.step(model.parameters(), model.gradients(), model.curvatures());
```

### Why use it?
- **Faster Convergence**: Essential for `DeepSpectralLinear` layers where gradients can vanish or explode relative to the scale parameters.
- **Handling Scale**: The curvature term $h \approx \sum x^2$ normalizes the update based on the signal energy at each frequency.

## Mixing Optimizers

In complex models like Spectral ViT, we often use different optimizers for different parts, or rely on `DiagonalNewton`'s ability to fallback to SGD if curvature is not provided (e.g., for Bias layers which return `nullptr` for curvature).

```cpp
// DiagonalNewton automatically handles layers without curvature by doing a standard SGD update
newton.step(model.parameters(), model.gradients(), model.curvatures());
```
