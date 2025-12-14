#ifndef DREIDEL_DIST_COMMUNICATOR_HPP
#define DREIDEL_DIST_COMMUNICATOR_HPP

#include "../core/Tensor.hpp"

namespace dreidel {
namespace dist {

template <typename T, BackendType B = BackendType::CPU>
class Communicator {
public:
    virtual ~Communicator() = default;

    // AllReduce: Sum gradients across all nodes
    virtual void all_reduce(Tensor<T, B>& tensor) = 0;

    // Broadcast: Send parameters from root to all nodes
    virtual void broadcast(Tensor<T, B>& tensor, int root_rank) = 0;

    virtual int rank() const = 0;
    virtual int size() const = 0;
};

} // namespace dist
} // namespace dreidel

#endif // DREIDEL_DIST_COMMUNICATOR_HPP
