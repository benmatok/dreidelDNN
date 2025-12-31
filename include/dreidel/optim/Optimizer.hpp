#ifndef DREIDEL_OPTIM_OPTIMIZER_HPP
#define DREIDEL_OPTIM_OPTIMIZER_HPP

#include "../core/Tensor.hpp"
#include <vector>

namespace dreidel {
namespace optim {

template <typename T, BackendType B = BackendType::CPU>
class Optimizer {
public:
    virtual ~Optimizer() = default;

    // Common interface for adding parameters
    virtual void add_parameters(std::vector<Tensor<T, B>*> params, std::vector<Tensor<T, B>*> grads) = 0;

    // Update weights given gradients
    virtual void step() = 0;

    // Zero out gradients
    virtual void zero_grad() = 0;
};

} // namespace optim
} // namespace dreidel

#endif // DREIDEL_OPTIM_OPTIMIZER_HPP
