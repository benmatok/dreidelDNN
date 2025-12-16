#ifndef DREIDEL_ALGO_WHT_HASHER_HPP
#define DREIDEL_ALGO_WHT_HASHER_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "../core/Tensor.hpp"
#include "WHT.hpp"

namespace dreidel {
namespace algo {

/**
 * @brief WHT Hasher for Large-Scale Filter Pruning.
 *
 * Generates binary codes using Sign(FWHT(x)).
 * These codes can be used to estimate cosine similarity or identify redundant filters.
 */
class WHTHasher {
public:
    /**
     * @brief Compute binary codes for a given input tensor.
     *
     * @tparam T Data type.
     * @tparam B Backend type.
     * @param input The input tensor (Batch, ..., Dim).
     * @return std::vector<uint64_t> Packed binary codes.
     *         Each row of the input produces ceil(Dim / 64) uint64_t values.
     *         The result is a flat vector of size Batch * ceil(Dim / 64).
     */
    template <typename T, BackendType B>
    static std::vector<uint64_t> compute_codes(const Tensor<T, B>& input) {
        // 1. Copy input to avoid modifying the original
        Tensor<T, B> temp = input;

        // 2. Apply FWHT
        WHT::FWHT(temp);

        // 3. Binarize and pack
        size_t last_dim = temp.shape().back();
        size_t num_vectors = temp.size() / last_dim;
        size_t num_uint64s = (last_dim + 63) / 64;

        std::vector<uint64_t> codes(num_vectors * num_uint64s, 0);

        const T* data = temp.data();

        // Parallelize over vectors
        #pragma omp parallel for
        for (long i = 0; i < (long)num_vectors; ++i) {
            const T* vec_ptr = data + i * last_dim;
            uint64_t* code_ptr = codes.data() + i * num_uint64s;

            for (size_t j = 0; j < last_dim; ++j) {
                if (vec_ptr[j] >= 0) {
                    size_t word_idx = j / 64;
                    size_t bit_idx = j % 64;
                    code_ptr[word_idx] |= (uint64_t(1) << bit_idx);
                }
            }
        }

        return codes;
    }

    /**
     * @brief Calculate Hamming distance between two packed codes.
     */
    static int hamming_distance(const std::vector<uint64_t>& code1, const std::vector<uint64_t>& code2) {
        if (code1.size() != code2.size()) {
             throw std::invalid_argument("Code sizes mismatch.");
        }
        int dist = 0;
        for (size_t i = 0; i < code1.size(); ++i) {
            uint64_t xor_val = code1[i] ^ code2[i];
            // Portable bit count
            // Kernighan's algorithm or std::bitset
            // std::bitset needs compile time size, so loop:
            while (xor_val) {
                xor_val &= (xor_val - 1);
                dist++;
            }
        }
        return dist;
    }
};

} // namespace algo
} // namespace dreidel

#endif // DREIDEL_ALGO_WHT_HASHER_HPP
