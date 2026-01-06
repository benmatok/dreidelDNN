#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/algo/WHT.hpp"
#include "../include/dreidel/algo/Sequency.hpp"

using namespace dreidel;

// Helper to generate a sawtooth wave
std::vector<float> generate_sawtooth(size_t N) {
    std::vector<float> signal(N);
    for (size_t i = 0; i < N; ++i) {
        signal[i] = (float)i / N;
    }
    return signal;
}

// Calculate energy concentration in the first K coefficients
float calculate_concentration(const std::vector<float>& signal, size_t K) {
    float energy_total = 0.0f;
    float energy_low = 0.0f;
    for (size_t i = 0; i < signal.size(); ++i) {
        float sq = signal[i] * signal[i];
        energy_total += sq;
        if (i < K) energy_low += sq;
    }
    return (energy_total > 0) ? (energy_low / energy_total) : 0.0f;
}

void test_sequency_permutation() {
    size_t N = 64; // Small power of 2
    auto signal = generate_sawtooth(N);

    // Copy for FWHT
    std::vector<float> buf = signal;

    // 1. FWHT (Natural)
    algo::WHT::fwht_1d(buf.data(), N);

    // Check concentration in Natural Order
    // A sawtooth wave has energy decaying as 1/k^2? In Walsh domain?
    // In Natural order, low sequency components are scattered.
    float conc_nat = calculate_concentration(buf, N/4);
    std::cout << "Energy Concentration (Natural, Top 25%): " << conc_nat * 100.0f << "%" << std::endl;

    // 2. Permute to Sequency
    auto seq_map = algo::Sequency::compute_to_natural_map(N);
    std::vector<float> seq_buf(N);
    for (size_t k = 0; k < N; ++k) {
        seq_buf[k] = buf[seq_map[k]];
    }

    // Check concentration in Sequency Order
    // Should be higher for low indices
    float conc_seq = calculate_concentration(seq_buf, N/4);
    std::cout << "Energy Concentration (Sequency, Top 25%): " << conc_seq * 100.0f << "%" << std::endl;

    if (conc_seq <= conc_nat) {
        std::cerr << "FAIL: Sequency ordering did not improve concentration." << std::endl;
        exit(1);
    } else {
        std::cout << "PASS: Sequency ordering improved concentration." << std::endl;
    }

    // Verify specifically that index 1 (high magnitude in Sequency) corresponds to
    // an index in Natural that is likely scattered.
    // Index 1 in Sequency (1 crossing) maps to Natural index ?
    // Map[1] = Natural index of Sequency 1.
    // K=1 -> Gray(1)=1 -> BR(1, 6)= 32 (if N=64).
    // Let's check map[1].
    std::cout << "Map[1] (Nat Index for Seq 1): " << seq_map[1] << std::endl;
    std::cout << "Value at Seq[1]: " << seq_buf[1] << std::endl;
    std::cout << "Value at Nat[" << seq_map[1] << "]: " << buf[seq_map[1]] << std::endl;
}

void test_zenith_integration() {
    size_t N = 1;
    size_t H = 4;
    size_t W = 4;
    size_t C = 64;

    Tensor<float> input({N, H, W, C});
    input.random(0.0f, 1.0f);

    // Create ZenithBlock with Sequency Ordering enabled
    layers::ZenithBlock<float> block_seq(C, 3, C, true, false, false, true, true); // use_slm=true, use_sequency=true

    // Manually set weights to Identity for Eyes and Mixer to isolate permutation effect?
    // Actually, we just want to ensure it runs and backward passes gradients correctly.
    // The permutation happens internally.
    // If SLM is enabled, weights are random.

    Tensor<float> output = block_seq.forward(input);
    Tensor<float> grad_out({N, H, W, C});
    grad_out.random(-0.1f, 0.1f);

    Tensor<float> grad_in = block_seq.backward(grad_out);

    std::cout << "ZenithBlock (Sequency) Forward/Backward ran successfully." << std::endl;
}

int main() {
    try {
        test_sequency_permutation();
        test_zenith_integration();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
