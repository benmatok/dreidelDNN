#include <dreidel/dreidel.hpp>
#include <dreidel/algo/WHTHasher.hpp>
#include <dreidel/hal/ops.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

using namespace dreidel;

void test_wht_hasher() {
    std::cout << "--- Testing WHT Hasher ---" << std::endl;
    // Dimension 64
    size_t dim = 64;
    Tensor<float> t1({1, dim});
    Tensor<float> t2({1, dim});

    // Randomize t1
    t1.random(0, 1);

    // t2 = t1 + small noise
    float* data1 = t1.data();
    float* data2 = t2.data();
    for (size_t i = 0; i < dim; ++i) {
        data2[i] = data1[i] + 0.01f * ((float)rand() / RAND_MAX - 0.5f);
    }

    auto codes1 = algo::WHTHasher::compute_codes(t1);
    auto codes2 = algo::WHTHasher::compute_codes(t2);

    int dist = algo::WHTHasher::hamming_distance(codes1, codes2);
    std::cout << "Hamming Distance (Similar vectors): " << dist << std::endl;

    // Expect distance to be small
    if (dist < 10) { // arbitrary threshold for 64 bits
        std::cout << "PASS: Hasher preserves locality." << std::endl;
    } else {
        std::cout << "FAIL: Distance too large for similar vectors (" << dist << ")" << std::endl;
        exit(1);
    }
}

void test_sparse_gather() {
    std::cout << "--- Testing Sparse Gather ---" << std::endl;
    size_t N = 1000;
    std::vector<float> base(N);
    std::iota(base.begin(), base.end(), 0.0f); // 0, 1, 2...

    size_t K = 128;
    std::vector<int> indices(K);
    for (size_t i = 0; i < K; ++i) {
        indices[i] = rand() % N;
    }

    std::vector<float> out(K);
    hal::sparse_gather(base.data(), indices.data(), out.data(), K);

    // Verify
    for (size_t i = 0; i < K; ++i) {
        if (out[i] != base[indices[i]]) {
            std::cout << "FAIL: Mismatch at index " << i << ". Expected " << base[indices[i]] << ", got " << out[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "PASS: Sparse Gather correct." << std::endl;
}

void benchmark_pruning() {
    std::cout << "--- Benchmarking Filter Pruning (Simulation) ---" << std::endl;
    // 4096 channels
    size_t C = 4096;
    size_t K = C / 10; // Select 10% (approx 400)

    // Setup Data
    std::vector<float> input(C);
    std::vector<float> weights(C);
    std::iota(input.begin(), input.end(), 1.0f);
    std::fill(weights.begin(), weights.end(), 0.5f);

    std::vector<int> indices(K);
    for(size_t i=0; i<K; ++i) indices[i] = (i * 10) % C;

    std::vector<float> sparse_weights(K);

    volatile float result = 0;

    // Benchmark Sparse Compute: Gather + Dot
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        // 1. Gather weights
        hal::sparse_gather(weights.data(), indices.data(), sparse_weights.data(), K);

        // 2. Gather input (assuming we also only need corresponding inputs,
        // usually indices select neurons, so we compute dot(w_subset, x_subset))
        // Actually usually we gather weights W[:, idx] but input is same...
        // Let's assume dot product of two sparse vectors.
        // We need to gather input too.
        std::vector<float> sparse_input(K);
        hal::sparse_gather(input.data(), indices.data(), sparse_input.data(), K);

        // 3. Dot
        float sum = 0;
        for(size_t j=0; j<K; ++j) sum += sparse_weights[j] * sparse_input[j];
        result = sum;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double sparse_time = std::chrono::duration<double>(end - start).count();

    // Benchmark Dense Compute: Full Dot Product
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        float sum = 0;
        // In AVX2 this would be faster, but simple loop for fair comparison with simple loop above
        // unless we used optimized GEMM.
        // Let's use std::inner_product or loop
        for(size_t j=0; j<C; ++j) sum += weights[j] * input[j];
        result = sum;
    }
    end = std::chrono::high_resolution_clock::now();
    double dense_time = std::chrono::duration<double>(end - start).count();

    std::cout << "Sparse Compute Time (Gather+Dot): " << sparse_time << " s" << std::endl;
    std::cout << "Dense Compute Time (Full Dot): " << dense_time << " s" << std::endl;

    double speedup = dense_time / sparse_time;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    if (speedup > 1.5) { // Lower threshold than 5x because we are doing double gather here overhead
        std::cout << "PASS: Speedup observed." << std::endl;
    } else {
         std::cout << "WARNING: Low speedup (Check optimization/AVX)." << std::endl;
    }
}

int main() {
    test_wht_hasher();
    test_sparse_gather();
    benchmark_pruning();
    return 0;
}
