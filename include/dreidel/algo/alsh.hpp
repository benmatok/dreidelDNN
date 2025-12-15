#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include "../core/Tensor.hpp"

namespace dreidel {
namespace algo {

/**
 * @brief Parameters for LSH.
 */
struct ALSHParams {
    int num_hashes; // K (number of bits per table)
    int num_tables; // L (number of tables)
    int bucket_width; // for scalar quantization if needed (unused for SRP)
    int seed = 42; // Random seed for reproducibility
};

/**
 * @brief Asymmetric Locality Sensitive Hashing (ALSH) Engine.
 *
 * Implements MIPS (Maximum Inner Product Search) to NNS (Nearest Neighbor Search)
 * transformation and bucket management.
 *
 * Uses Signed Random Projections (SRP) for Cosine Similarity.
 * MIPS transform: P(x) = [x, sqrt(M^2 - ||x||^2)], Q(q) = [q, 0]
 */
template <typename T, BackendType B = BackendType::CPU>
class ALSH {
public:
    ALSH(const ALSHParams& params) : params_(params) {
        if (params_.num_hashes > 20) {
             throw std::invalid_argument("ALSH: num_hashes > 20 is too large for direct addressing.");
        }
        if (params_.num_hashes < 1) {
             throw std::invalid_argument("ALSH: num_hashes must be >= 1.");
        }
    }

    /**
     * @brief Preprocess weights and build hash tables.
     *
     * @param weights The weight matrix from a layer (InputDim x OutputDim).
     *                Items are columns.
     */
    void build_index(const Tensor<T, B>& weights) {
        if (weights.shape().size() != 2) {
            throw std::invalid_argument("ALSH: Weights must be 2D tensor.");
        }
        size_t input_dim = weights.shape()[0];
        size_t output_dim = weights.shape()[1];

        orig_dim_ = input_dim;
        item_count_ = output_dim;

        // Initialize deduplication structures
        visited_token_.assign(item_count_, 0);
        current_token_ = 0;

        // 1. Calculate norms of each item (column) to find max norm M
        std::vector<T> norms(output_dim, 0);
        const T* w_data = weights.data();

        // Accessing column-wise: w_data[i * output_dim + j]
        for (size_t j = 0; j < output_dim; ++j) {
            T sq_sum = 0;
            for (size_t i = 0; i < input_dim; ++i) {
                T val = w_data[i * output_dim + j];
                sq_sum += val * val;
            }
            norms[j] = std::sqrt(sq_sum);
        }

        max_norm_ = 0;
        for (T n : norms) {
            if (n > max_norm_) max_norm_ = n;
        }

        // 2. Generate random projections
        // Dimension is input_dim + 1 for the MIPS transform
        size_t proj_dim = input_dim + 1;
        generate_projections(proj_dim);

        // 3. Initialize buckets (Direct Addressing)
        // buckets_[table_idx][hash_val] -> vector of item indices
        size_t num_buckets = 1 << params_.num_hashes;
        buckets_.clear();
        buckets_.resize(params_.num_tables);
        for(auto& table : buckets_) {
            table.resize(num_buckets);
        }

        // 4. Hash each item
        // P(x) = [x, sqrt(M^2 - ||x||^2)]
        std::vector<T> temp_vec(proj_dim);

        for (size_t j = 0; j < output_dim; ++j) {
            // Copy x
            for (size_t i = 0; i < input_dim; ++i) {
                temp_vec[i] = w_data[i * output_dim + j];
            }
            // Add extra dimension
            T diff = (max_norm_ * max_norm_) - (norms[j] * norms[j]);
            temp_vec[input_dim] = (diff > 0) ? std::sqrt(diff) : 0;

            // Compute hash for each table
            for (int l = 0; l < params_.num_tables; ++l) {
                size_t h = compute_hash(temp_vec, l);
                buckets_[l][h].push_back(static_cast<int>(j));
            }
        }
    }

    /**
     * @brief Query the index for active neurons.
     *
     * @param query_vec Input activation vector (1D or 1xInputDim).
     * @return std::vector<int> Indices of potential top-k neurons (candidates).
     */
    std::vector<int> query(const Tensor<T, B>& query_vec) {
        size_t q_size = query_vec.size();
        if (q_size != orig_dim_) {
            throw std::invalid_argument("ALSH: Query dimension mismatch.");
        }

        // Q(q) = [q, 0]
        std::vector<T> temp_vec(orig_dim_ + 1);
        const T* q_data = query_vec.data();
        for (size_t i = 0; i < orig_dim_; ++i) {
            temp_vec[i] = q_data[i];
        }
        temp_vec[orig_dim_] = 0;

        // Increment generation token for deduplication
        current_token_++;
        // Handle wrap-around (very rare, but safe)
        if (current_token_ == 0) {
            std::fill(visited_token_.begin(), visited_token_.end(), 0);
            current_token_ = 1;
        }

        std::vector<int> candidates;
        // Reserve some space to avoid reallocations (heuristic)
        candidates.reserve(item_count_ / 100);

        for (int l = 0; l < params_.num_tables; ++l) {
            size_t h = compute_hash(temp_vec, l);
            // Direct access
            const auto& bucket_indices = buckets_[l][h];

            for (int idx : bucket_indices) {
                if (visited_token_[idx] != current_token_) {
                    visited_token_[idx] = current_token_;
                    candidates.push_back(idx);
                }
            }
        }

        return candidates;
    }

    /**
     * @brief Update index (for Mongoose/Learnable LSH).
     */
    void update() {
        // TODO: Re-hash or adjust hash functions.
    }

    // Accessor for max_norm (useful for verification if needed)
    T get_max_norm() const { return max_norm_; }

private:
    ALSHParams params_;
    size_t orig_dim_ = 0;
    size_t item_count_ = 0;
    T max_norm_ = 0;

    // Deduplication optimization
    std::vector<unsigned int> visited_token_;
    unsigned int current_token_ = 0;

    // Projections: [table][hash_index][dimension]
    // Flattened as vector of vectors
    std::vector<std::vector<T>> projections_;

    // Buckets: [table][hash_key] -> list of indices
    // Replaced unordered_map with vector for direct addressing
    std::vector<std::vector<std::vector<int>>> buckets_;

    void generate_projections(size_t dim) {
        size_t total_hashes = params_.num_tables * params_.num_hashes;
        projections_.resize(total_hashes);

        // Use fixed seed from params or default for reproducibility
        std::mt19937 gen(params_.seed);
        std::normal_distribution<T> d(0.0, 1.0);

        for (auto& proj : projections_) {
            proj.resize(dim);
            for (auto& val : proj) {
                val = d(gen);
            }
        }
    }

    size_t compute_hash(const std::vector<T>& vec, int table_idx) {
        size_t hash = 0;
        size_t dim = vec.size();

        // Start index for this table's projections
        size_t start_proj = table_idx * params_.num_hashes;

        for (int k = 0; k < params_.num_hashes; ++k) {
            const auto& proj = projections_[start_proj + k];

            // Dot product
            T dot = 0;
            for (size_t i = 0; i < dim; ++i) {
                dot += vec[i] * proj[i];
            }

            if (dot >= 0) {
                hash |= (size_t(1) << k);
            }
        }
        return hash;
    }
};

} // namespace algo
} // namespace dreidel
