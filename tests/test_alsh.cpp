#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <string>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/algo/alsh.hpp"

using namespace dreidel;

// Helper to compute exact top-k
std::vector<int> get_exact_top_k(const Tensor<float>& query, const Tensor<float>& weights, int k) {
    size_t input_dim = weights.shape()[0];
    size_t output_dim = weights.shape()[1];

    std::vector<std::pair<float, int>> scores;
    scores.reserve(output_dim);

    const float* w_data = weights.data();
    const float* q_data = query.data();

    for(size_t j=0; j<output_dim; ++j) {
        float score = 0;
        for(size_t i=0; i<input_dim; ++i) {
            score += q_data[i] * w_data[i * output_dim + j];
        }
        scores.push_back({score, (int)j});
    }

    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b){
        return a.first > b.first;
    });

    std::vector<int> top_k;
    for(int i=0; i<k; ++i) top_k.push_back(scores[i].second);
    return top_k;
}

void run_test(std::string name, std::function<bool()> test) {
    std::cout << "[TEST] " << name << "... ";
    if (test()) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
        exit(1);
    }
}

int main() {
    std::cout << "Running Comprehensive ALSH Validation..." << std::endl;

    // Test 1: Sanity Check (Identity)
    // A vector should always find itself if it's in the index.
    run_test("Identity Retrieval", []() {
        size_t dim = 64;
        size_t N = 100;
        Tensor<float> weights({dim, N});
        weights.random(0.0f, 1.0f);

        algo::ALSHParams params;
        params.num_hashes = 4;
        params.num_tables = 5;
        params.seed = 123;

        algo::ALSH<float> alsh(params);
        alsh.build_index(weights);

        // Pick item 0
        Tensor<float> query({dim});
        for(size_t i=0; i<dim; ++i) query.data()[i] = weights.data()[i * N + 0];

        auto candidates = alsh.query(query);

        // Check if 0 is in candidates
        for(int idx : candidates) {
            if(idx == 0) return true;
        }
        return false;
    });

    // Test 2: Structured Data Recall
    // Create clusters. Query near cluster center. Should retrieve neighbors.
    run_test("Structured Data Recall", []() {
        size_t dim = 64;
        size_t N = 1000;
        Tensor<float> weights({dim, N});

        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // 10 clusters
        std::vector<std::vector<float>> centroids(10, std::vector<float>(dim));
        for(auto& c : centroids) for(auto& v : c) v = dist(gen);

        float* w = weights.data();
        for(size_t i=0; i<N; ++i) {
            int c = i % 10;
            for(size_t d=0; d<dim; ++d) {
                // Low noise to ensure high correlation within cluster
                w[d * N + i] = centroids[c][d] + dist(gen) * 0.1f;
            }
        }

        algo::ALSHParams params;
        params.num_hashes = 8;
        params.num_tables = 10;

        algo::ALSH<float> alsh(params);
        alsh.build_index(weights);

        // Query: Centroid 0
        Tensor<float> query({dim});
        for(size_t d=0; d<dim; ++d) query.data()[d] = centroids[0][d];

        auto candidates = alsh.query(query);
        auto exact_top_10 = get_exact_top_k(query, weights, 10);

        int hits = 0;
        for(int target : exact_top_10) {
            for(int c : candidates) {
                if(c == target) { hits++; break; }
            }
        }

        std::cout << "(Recall: " << hits << "/10) ";
        return hits >= 8; // Expect decent recall
    });

    // Test 3: Noise Sensitivity
    // As noise increases, recall should likely drop or require more tables.
    // We just verify that with high noise, we don't get 100% recall trivially,
    // but ALSH still functions (returns candidates).
    run_test("High Noise Robustness", []() {
        size_t dim = 64;
        size_t N = 1000;
        Tensor<float> weights({dim, N});

        std::mt19937 gen(99);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> centroid(dim);
        for(float& v : centroid) v = dist(gen);

        float* w = weights.data();
        // All points are noisy versions of ONE centroid
        for(size_t i=0; i<N; ++i) {
            for(size_t d=0; d<dim; ++d) {
                // High noise
                w[d * N + i] = centroid[d] + dist(gen) * 1.5f;
            }
        }

        algo::ALSHParams params;
        params.num_hashes = 6;
        params.num_tables = 10;

        algo::ALSH<float> alsh(params);
        alsh.build_index(weights);

        Tensor<float> query({dim});
        for(size_t d=0; d<dim; ++d) query.data()[d] = centroid[d];

        auto candidates = alsh.query(query);
        // We just ensure we return something and not everything
        // High noise means points are scattered, but still somewhat related to query.
        // candidates shouldn't be empty, but ideally not full N (if buckets split space)

        return !candidates.empty() && candidates.size() < N;
    });

    // Test 4: Orthogonal / Hard Negative
    // Query orthogonal to data should ideally return few candidates (or random ones),
    // but definitely shouldn't break.
    run_test("Orthogonal Query", []() {
        size_t dim = 2; // Simple 2D
        size_t N = 100;
        Tensor<float> weights({dim, N});

        // Data on X axis: [1, 0], [2, 0]...
        float* w = weights.data();
        for(size_t i=0; i<N; ++i) {
            w[0 * N + i] = 1.0f + (float)i/N;
            w[1 * N + i] = 0.0f;
        }

        algo::ALSHParams params;
        params.num_hashes = 2; // 4 buckets
        params.num_tables = 2;

        algo::ALSH<float> alsh(params);
        alsh.build_index(weights);

        // Query on Y axis: [0, 1]
        // Orthogonal. Dot product is 0.
        // MIPS transform might map them differently.
        Tensor<float> query({dim});
        query.data()[0] = 0.0f;
        query.data()[1] = 1.0f;

        auto candidates = alsh.query(query);
        // Just ensure it runs.
        return true;
    });

    std::cout << "All validation tests passed." << std::endl;
    return 0;
}
