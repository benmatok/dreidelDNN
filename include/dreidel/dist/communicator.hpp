#pragma once

#include <vector>
#include "../core/tensor.hpp"

namespace dreidel {
namespace dist {

/**
 * @brief Abstract Communicator for Distributed Training.
 *
 * Supports operations like AllReduce for synchronizing gradients across nodes.
 */
class Communicator {
public:
    virtual ~Communicator() = default;

    /**
     * @brief Initialize the communication backend (e.g., MPI Init).
     */
    virtual void init() = 0;

    /**
     * @brief Rank of the current node.
     */
    virtual int rank() = 0;

    /**
     * @brief Total number of nodes.
     */
    virtual int size() = 0;

    /**
     * @brief AllReduce (Sum) operation for gradients.
     *
     * @param tensor The tensor to reduce.
     */
    template <typename T>
    void all_reduce(core::Tensor<T>& tensor) {
        // Mock implementation: do nothing (single node)
        // Real implementation: MPI_Allreduce
    }
};

} // namespace dist
} // namespace dreidel
