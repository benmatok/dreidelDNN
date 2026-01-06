#ifndef DREIDEL_ALGO_SEQUENCY_HPP
#define DREIDEL_ALGO_SEQUENCY_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace dreidel {
namespace algo {

class Sequency {
public:
    // Bit reversal of 'v' assuming 'bits' number of bits.
    static inline uint32_t bit_reverse(uint32_t v, int bits) {
        uint32_t r = 0;
        for (int i = 0; i < bits; ++i) {
            r = (r << 1) | (v & 1);
            v >>= 1;
        }
        return r;
    }

    // Binary to Gray code: g = b ^ (b >> 1)
    static inline uint32_t binary_to_gray(uint32_t b) {
        return b ^ (b >> 1);
    }

    // Gray to Binary (not needed for the standard map, but good to have)
    static inline uint32_t gray_to_binary(uint32_t g) {
        uint32_t mask = g >> 1;
        while (mask != 0) {
            g = g ^ mask;
            mask = mask >> 1;
        }
        return g;
    }

    // Generates map where map[k] = Natural Index of the k-th Sequency
    // Output: Sequency Ordered Buffer[k] = Natural Ordered Buffer[map[k]]
    static std::vector<int32_t> compute_to_natural_map(size_t N) {
        if ((N & (N - 1)) != 0) {
            throw std::invalid_argument("N must be a power of 2");
        }

        std::vector<int32_t> map(N);
        int bits = 0;
        while ((1UL << bits) < N) bits++;

        for (size_t k = 0; k < N; ++k) {
            // Formula: The k-th sequency index corresponds to natural index:
            // BitReverse(Gray(k))?
            // Let's re-verify with k=0..3 example.
            // k=0 (Seq 0) -> Gray(0)=0 -> BR(0)=0. Natural 0. (Correct)
            // k=1 (Seq 1) -> Gray(1)=1 -> BR(1)=2. Natural 2. (Correct)
            // k=2 (Seq 2) -> Gray(2)=3 -> BR(3)=3. Natural 3. (Correct)
            // k=3 (Seq 3) -> Gray(3)=2 -> BR(2)=1. Natural 1. (Correct)

            uint32_t gray = binary_to_gray((uint32_t)k);
            uint32_t natural = bit_reverse(gray, bits);
            map[k] = (int32_t)natural;
        }
        return map;
    }

    // Generates map where map[i] = Sequency Index of the i-th Natural Index
    // Useful for inverse permutation if we iterate naturally?
    // Actually we usually just use the same map backwards or scatter/gather.
    static std::vector<int32_t> compute_to_sequency_map(size_t N) {
        auto to_nat = compute_to_natural_map(N);
        std::vector<int32_t> to_seq(N);
        for (size_t k = 0; k < N; ++k) {
            to_seq[to_nat[k]] = (int32_t)k;
        }
        return to_seq;
    }
};

} // namespace algo
} // namespace dreidel

#endif // DREIDEL_ALGO_SEQUENCY_HPP
