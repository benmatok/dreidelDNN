#ifndef DREIDEL_JULES_FUSEDJULESBLOCK_HPP
#define DREIDEL_JULES_FUSEDJULESBLOCK_HPP

#include "SparseBlock.hpp"
#include "SoftPermutation.hpp"
#include "DenseLUT.hpp"
#include "../algo/WHT.hpp"
#include "../core/Arena.hpp"

namespace dreidel {
namespace jules {

/**
 * @brief Fused Jules Block.
 *
 * Composition:
 * Input -> [ Branch 1: ALSH (Sparse) ] + [ Branch 2: Spectral (FWHT + SoftPerm) ] -> Output
 *
 * Branch 1 (Depth-wise ALSH):
 * - Input -> SparseBlock -> Out1
 *
 * Branch 2 (Point-wise Spectral):
 * - Input -> FWHT -> SoftPermutation -> DenseLUT (Mixing/Scale) -> Out2
 *
 * Output = Out1 + Out2
 */
template <size_t DIM, size_t HIDDEN_DIM, size_t NUM_EXPERTS = 4>
class FusedJulesBlock {
public:
    // Branch 1: Sparse
    SparseBlock<DIM, HIDDEN_DIM, NUM_EXPERTS> sparse_branch;

    // Branch 2: Spectral
    // Step A: FWHT (In-place on scratch)
    // Step B: Soft Permutation
    SoftPermutation<DIM> soft_perm;
    // Step C: DenseLUT (Mixing/Scaling to match HIDDEN_DIM)
    // If DIM != HIDDEN_DIM, we need this to project.
    // If DIM == HIDDEN_DIM, this acts as the "APoT" weighting for the spectral branch.
    DenseLUT<DIM, HIDDEN_DIM> spectral_mix;

    FusedJulesBlock() {
        // Init happens in members
    }

    // Forward needs workspace for intermediate branches
    // We assume input and output are distinct buffers provided by caller.
    // Workspace provided for internal scratch.
    void forward(core::TensorView<float>& input, core::TensorView<float>& output, core::Arena* arena) {
        // We need to sum two branches.
        // Output will accumulate, so we assume Output is zeroed or we write to it?
        // Let's write Branch 1 to Output, then add Branch 2.

        // --- Branch 1: Sparse ---
        sparse_branch.forward(input, output);

        // --- Branch 2: Spectral ---
        // Need scratch for FWHT result
        float* scratch_ptr = arena->allocate<float>(DIM);
        core::TensorView<float> scratch(scratch_ptr, DIM);

        // 1. Copy Input -> Scratch
        std::copy(input.data(), input.data() + DIM, scratch_ptr);

        // 2. FWHT
        algo::WHT::FWHT(scratch);

        // 3. Soft Permutation (Scratch -> Scratch)
        // Note: SoftPerm implementation supports in-place if in==out pointers
        soft_perm.forward(scratch, scratch);

        // 4. Mixing (DenseLUT) -> Add to Output
        // DenseLUT overwrites usually?
        // DenseLUT::forward currently does: output.fill(0); ... accumulate ...
        // We need it to ADD to existing output.
        // Let's allocate a temp buffer for Branch 2 result, then add.
        // Or modify DenseLUT to support "accumulate" mode.
        // For standard "AgentJules", DenseLUT was final layer.
        // Let's Alloc temp.

        float* branch2_ptr = arena->allocate<float>(HIDDEN_DIM);
        core::TensorView<float> branch2_out(branch2_ptr, HIDDEN_DIM);

        spectral_mix.forward(scratch, branch2_out);

        // Sum: Output += Branch2
        float* out_data = output.data();
        for(size_t i=0; i<HIDDEN_DIM; ++i) {
            out_data[i] += branch2_ptr[i];
        }
    }
};

} // namespace jules
} // namespace dreidel

#endif // DREIDEL_JULES_FUSEDJULESBLOCK_HPP
