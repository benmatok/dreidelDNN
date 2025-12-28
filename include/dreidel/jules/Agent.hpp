#ifndef DREIDEL_JULES_AGENT_HPP
#define DREIDEL_JULES_AGENT_HPP

#include "../core/TensorView.hpp"
#include "../core/Arena.hpp"
#include "../algo/WHT.hpp"
#include "../quant/APoT.hpp"
#include "../hal/ops.hpp"
#include "SparseBlock.hpp"
#include "DenseLUT.hpp"
#include <cmath>
#include <iostream>

namespace dreidel {
namespace jules {

/**
 * @brief Base class for a "Jules" Agent.
 */
class Agent {
public:
    virtual void init() = 0;
    virtual void step(const float* input, float* output) = 0;
    virtual ~Agent() = default;
};

// Example Spectral Block Implementation (Layer)
// y = FWHT(x) * D
template <size_t DIM>
class SpectralBlock {
public:
    float scales[DIM]; // Diagonal scales (Weights)

    SpectralBlock() {
        for(size_t i=0; i<DIM; ++i) scales[i] = 1.0f;
    }

    void forward(core::TensorView<float>& input, core::TensorView<float>& output) {
        float* in_ptr = input.data();
        float* out_ptr = output.data();

        // Copy
        std::copy(in_ptr, in_ptr + DIM, out_ptr);

        core::TensorView<float> out_view(out_ptr, DIM);

        // In-place FWHT
        algo::WHT::FWHT(out_view);

        // Element-wise Scale
        #pragma omp simd
        for(size_t i=0; i<DIM; ++i) {
            out_ptr[i] *= scales[i];
        }
    }
};

/**
 * @brief The Concrete Jules Agent.
 *
 * Pipeline:
 * 1. SpectralBlock<1024>
 * 2. SparseBlock<1024, 1024, 4> (4 Experts)
 * 3. DenseLUT<1024, 16>
 */
template <size_t INPUT_DIM, size_t HIDDEN_DIM, size_t OUTPUT_DIM>
class AgentJules : public Agent {
public:
    // Layer 1: Spectral Mixing
    SpectralBlock<INPUT_DIM> layer1;

    // Layer 2: Sparse Expert Selection
    // Assume Hidden Dim is same as Input for Sparse Block I/O for simplicity here,
    // or SparseBlock maps Input->Hidden.
    SparseBlock<INPUT_DIM, HIDDEN_DIM, 4> layer2; // 4 Experts

    // Layer 3: Dense LUT Output
    DenseLUT<HIDDEN_DIM, OUTPUT_DIM> layer3;

    core::Arena* workspace_;

    AgentJules(core::Arena* workspace) : workspace_(workspace) {}

    void init() override {
        quant::APoT::init();
        // Force LUT init
        DenseLUT<HIDDEN_DIM, OUTPUT_DIM>::init_lut();
    }

    void step(const float* input_ptr, float* output_ptr) override {
        workspace_->reset();

        // Allocations are pointer bumps (Zero syscalls)
        float* buf_a = workspace_->allocate<float>(INPUT_DIM);
        float* buf_b = workspace_->allocate<float>(HIDDEN_DIM);

        // 0. Copy Input -> A
        std::copy(input_ptr, input_ptr + INPUT_DIM, buf_a);

        // Define Views (Shape array is on stack, cheap)
        core::TensorView<float> view_a(buf_a, INPUT_DIM);
        core::TensorView<float> view_b(buf_b, HIDDEN_DIM);

        // 1. Layer 1 (Spectral): A -> A (In-place logic inside layer or copy)
        // SpectralBlock implementation copies Input to Output then modifies Output.
        // We can optimize to do A -> B, but let's stick to A -> A (using B as scratch if needed).
        // Actually SpectralBlock::forward takes Input and Output.
        // Let's do A -> B
        layer1.forward(view_a, view_b);

        // Now B holds result of L1.
        // 2. Layer 2 (Sparse): B -> A
        // We reuse A for output of L2.
        // If HIDDEN_DIM != INPUT_DIM, we might need reallocation if A is too small,
        // but here INPUT_DIM=1024, HIDDEN=1024.

        core::TensorView<float> view_b_in(buf_b, HIDDEN_DIM); // Correct view
        core::TensorView<float> view_a_out(buf_a, HIDDEN_DIM); // Reuse A

        layer2.forward(view_b_in, view_a_out);

        // Now A holds result of L2.
        // 3. Layer 3 (DenseLUT): A -> Output Ptr
        // DenseLUT takes View Input, View Output
        // We can wrap output_ptr directly.
        core::TensorView<float> view_final_in(buf_a, HIDDEN_DIM);
        core::TensorView<float> view_final_out(output_ptr, OUTPUT_DIM);

        layer3.forward(view_final_in, view_final_out);
    }
};

} // namespace jules
} // namespace dreidel

#endif // DREIDEL_JULES_AGENT_HPP
