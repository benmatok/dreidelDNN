#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <numeric>

#include "../include/dreidel/layers/ZenithBlock.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/hal/ops.hpp"

using namespace dreidel;

// Helper for timing
template<typename Func>
double time_execution(Func f, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<iterations; ++i) {
        f();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// ----------------------------------------------------------------------------
// Test 1: The "L1 Resident" Eye Test
// Goal: Verify vpermb throughput.
// ----------------------------------------------------------------------------
void test_l1_resident_eye() {
    std::cout << "\n[Test 1] L1 Resident Eye Test (vpermb throughput)" << std::endl;

    size_t C = 64; // Small channel count to fit in registers
    size_t K = 3;
    size_t H = 8, W = 8; // Small spatial to fit in L1 (8*8*64*1 = 4KB for int8)

    // Create Zenith Block (Int8)
    // C, C, K, C (Cin=C, Cout=C, K=K, Spec=C)
    layers::ZenithBlock zenith(C, C, K, C, 1024*1024, false);

    // Input Tensor (Random Int8)
    Tensor<int8_t> input({1, H, W, C});
    input.fill(1); // Dummy data

    // Run Loop
    int iterations = 10000;

    double duration = time_execution([&]() {
        zenith.forward(input); // Forward only
    }, iterations);

    double ops_per_forward = (double)H * W * K * K * C; // Rough MACs equivalent
    double total_ops = ops_per_forward * iterations;
    double gflops = (total_ops / duration) / 1e9;

    std::cout << "  - Iterations: " << iterations << std::endl;
    std::cout << "  - Time: " << duration << "s" << std::endl;
    std::cout << "  - Throughput: " << (iterations / duration) << " passes/sec" << std::endl;
    std::cout << "  - Est. GOps/s (Equiv): " << gflops << std::endl;
}

// ----------------------------------------------------------------------------
// Test 2: The "Memory Wall" Test
// Goal: Measure impact of Phase 6 (Z-Curve) by blowing out L3 cache.
// ----------------------------------------------------------------------------
void test_memory_wall() {
    std::cout << "\n[Test 2] Memory Wall Test (100MB Image)" << std::endl;

    // 100MB ~= 25M floats.
    // Let C=64. Then H*W = 25M / 64 ~= 390k pixels.
    // Let's use 512x512 = 262k pixels -> ~67MB. Good enough.

    size_t C = 64;
    size_t H = 512;
    size_t W = 512;
    size_t K = 3;

    layers::ZenithBlock zenith(C, C, K, C, 10*1024*1024, false); // Larger arena

    Tensor<int8_t> input({1, H, W, C});
    input.fill(1);

    int iterations = 10;

    double duration = time_execution([&]() {
        zenith.forward(input);
    }, iterations);

    double throughput_mb = (double)(input.size() * sizeof(int8_t) * iterations) / (1024.0*1024.0) / duration;

    std::cout << "  - Input Size: " << (input.size() * sizeof(int8_t) / (1024.0*1024.0)) << " MB" << std::endl;
    std::cout << "  - Time: " << duration << "s" << std::endl;
    std::cout << "  - Bandwidth: " << throughput_mb << " MB/s (Read)" << std::endl; // Rough estimate
}

// ----------------------------------------------------------------------------
// Test 3: The "Sparsity" Breakeven Analysis
// Goal: Find the "Breakeven Point" for the Oracle.
// ----------------------------------------------------------------------------
void test_sparsity_breakeven() {
    std::cout << "\n[Test 3] Sparsity Breakeven Analysis" << std::endl;

    size_t C = 64;
    size_t H = 64;
    size_t W = 64;
    size_t K = 3;

    // We can't easily control "active blocks" from outside without modifying ZenithBlock internals
    // or crafting specific inputs.
    // However, ZenithBlock uses `use_gating` flag.
    // And it computes hash based on input.
    // To simulate different sparsity levels, we might need to modify ZenithBlock or just run with random data
    // and see what happens, or skip this specific precise control for now and just compare Gating ON vs OFF on random noise.
    // Random noise usually has low sparsity (high entropy), so Oracle might not skip much or might skip 50%.
    // The current Zenith implementation: `if (dist > (int)(C/2)) continue;`
    // Random vectors have dist ~ C/2. So it's a toss up.

    std::cout << "  Comparing Gating ON vs OFF on Random Noise (Approx 50% sparsity)" << std::endl;

    Tensor<int8_t> input({1, H, W, C});
    // Fill with random int8
    for(size_t i=0; i<input.size(); ++i) input.data()[i] = (rand() % 256) - 128;

    // Gating OFF
    layers::ZenithBlock zenith_off(C, C, K, C, 1024*1024, false);
    double t_off = time_execution([&]() { zenith_off.forward(input); }, 100);

    // Gating ON
    layers::ZenithBlock zenith_on(C, C, K, C, 1024*1024, true);
    double t_on = time_execution([&]() { zenith_on.forward(input); }, 100);

    std::cout << "  - Gating OFF Time: " << t_off << "s" << std::endl;
    std::cout << "  - Gating ON Time:  " << t_on << "s" << std::endl;
    std::cout << "  - Speedup: " << (t_off / t_on) << "x" << std::endl;

    if (t_on > t_off) {
        std::cout << "  -> Oracle overhead is higher than savings (or sparsity low)." << std::endl;
    } else {
        std::cout << "  -> Oracle is providing speedup." << std::endl;
    }
}

int main() {
    std::cout << "=== Alien Technology Benchmarks ===" << std::endl;

    test_l1_resident_eye();
    test_memory_wall();
    test_sparsity_breakeven();

    return 0;
}
