#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <chrono>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/algo/alsh.hpp"

using namespace dreidel;

// Helper to compute exact dot product
template<typename T>
T dot_product(const std::vector<T>& a, const std::vector<T>& b) {
    T sum = 0;
    for(size_t i=0; i<a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

int main() {
    std::cout << "Running ALSH Test..." << std::endl;

    // Parameters
    size_t input_dim = 128;
    size_t output_dim = 10000; // Number of items
    int K = 10;
    int L = 5;

    // Create random weights (items)
    // Weights are (InputDim, OutputDim)
    Tensor<float> weights({input_dim, output_dim});

    // Create structured data for test to ensure recall > 0
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int num_clusters = 100;
    std::vector<std::vector<float>> centroids(num_clusters, std::vector<float>(input_dim));
    for(auto& c : centroids) for(auto& v : c) v = dist(gen);

    float* w_data = weights.data();
    for(size_t i=0; i<output_dim; ++i) {
        int c_idx = i % num_clusters;
        for(size_t d=0; d<input_dim; ++d) {
            w_data[d * output_dim + i] = centroids[c_idx][d] + dist(gen) * 0.1f;
        }
    }

    // Build Index
    algo::ALSHParams params;
    params.num_hashes = K;
    params.num_tables = L;

    algo::ALSH<float> alsh(params);

    auto start_build = std::chrono::high_resolution_clock::now();
    alsh.build_index(weights);
    auto end_build = std::chrono::high_resolution_clock::now();
    std::cout << "Index build time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count()
              << " ms" << std::endl;

    // Create a random query
    Tensor<float> query({input_dim});
    float* q_data = query.data();
    for(size_t d=0; d<input_dim; ++d) q_data[d] = centroids[0][d] + dist(gen) * 0.1f;

    // 1. Exact Search (Brute Force)
    std::vector<std::pair<float, int>> exact_scores;
    exact_scores.reserve(output_dim);

    auto start_bf = std::chrono::high_resolution_clock::now();
    for(size_t j=0; j<output_dim; ++j) {
        float score = 0;
        for(size_t i=0; i<input_dim; ++i) {
            score += q_data[i] * w_data[i * output_dim + j];
        }
        exact_scores.push_back({score, (int)j});
    }
    // Sort descending
    std::sort(exact_scores.begin(), exact_scores.end(), [](const auto& a, const auto& b){
        return a.first > b.first;
    });
    auto end_bf = std::chrono::high_resolution_clock::now();

    // Top 10
    int top_k = 10;
    std::vector<int> exact_top_k;
    for(int i=0; i<top_k; ++i) exact_top_k.push_back(exact_scores[i].second);

    // 2. ALSH Search
    auto start_query = std::chrono::high_resolution_clock::now();
    std::vector<int> candidates = alsh.query(query);

    // Re-rank candidates
    std::vector<std::pair<float, int>> approx_scores;
    for(int idx : candidates) {
        float score = 0;
        for(size_t i=0; i<input_dim; ++i) {
            score += q_data[i] * w_data[i * output_dim + idx];
        }
        approx_scores.push_back({score, idx});
    }
    std::sort(approx_scores.begin(), approx_scores.end(), [](const auto& a, const auto& b){
        return a.first > b.first;
    });
    auto end_query = std::chrono::high_resolution_clock::now();

    // Check recall @ 10
    // How many of exact_top_k are in the candidates?
    int hits = 0;
    for(int target : exact_top_k) {
        bool found = false;
        for(const auto& p : approx_scores) {
            if(p.second == target) {
                found = true;
                break;
            }
        }
        if(found) hits++;
    }

    std::cout << "Brute Force Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_bf - start_bf).count() << " us" << std::endl;
    std::cout << "ALSH Query Time (incl. re-rank): " << std::chrono::duration_cast<std::chrono::microseconds>(end_query - start_query).count() << " us" << std::endl;
    std::cout << "Candidates count: " << candidates.size() << " / " << output_dim << std::endl;
    std::cout << "Recall @ " << top_k << ": " << hits << " / " << top_k << " (" << (float)hits/top_k * 100 << "%)" << std::endl;

    // Verify retrieval accuracy
    if (hits > 0) {
        std::cout << "Test Passed: Retrieved at least some relevant items." << std::endl;
        return 0;
    } else {
        std::cout << "Test Failed: Zero recall." << std::endl;
        return 1;
    }
}
