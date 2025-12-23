#ifndef DREIDEL_LAYERS_MULTI_HEAD_ATTENTION_SPECTRAL_HPP
#define DREIDEL_LAYERS_MULTI_HEAD_ATTENTION_SPECTRAL_HPP

#include "Layer.hpp"
#include "LinearWHT.hpp"
#include "DeepSpectralLinear.hpp"
#include "Softmax.hpp"
#include <vector>
#include <cmath>

namespace dreidel {
namespace layers {

template <typename T, BackendType B = BackendType::CPU>
class MultiHeadAttentionSpectral : public Layer<T, B> {
public:
    // Helper for Po2
    static size_t next_power_of_2(size_t n) {
        if (n == 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n++;
        return n;
    }

    MultiHeadAttentionSpectral(size_t dim, size_t num_heads, bool use_deep_spectral = true)
        : dim_(dim), num_heads_(num_heads), head_dim_(dim / num_heads), use_deep_spectral_(use_deep_spectral)
    {
        if (dim % num_heads != 0) {
            throw std::invalid_argument("Dimension must be divisible by number of heads.");
        }

        // Q, K, V, O projections
        // For simplicity, we use the same type for all
        if (use_deep_spectral) {
            size_t s_dim = next_power_of_2(dim);
            q_proj_ = std::make_shared<DeepSpectralLinear<T, B>>(s_dim);
            k_proj_ = std::make_shared<DeepSpectralLinear<T, B>>(s_dim);
            v_proj_ = std::make_shared<DeepSpectralLinear<T, B>>(s_dim);
            o_proj_ = std::make_shared<DeepSpectralLinear<T, B>>(s_dim);
        } else {
            q_proj_ = std::make_shared<LinearWHT<T, B>>(dim);
            k_proj_ = std::make_shared<LinearWHT<T, B>>(dim);
            v_proj_ = std::make_shared<LinearWHT<T, B>>(dim);
            o_proj_ = std::make_shared<LinearWHT<T, B>>(dim);
        }
    }

    // Forward for Self-Attention
    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        return forward_attention(input, input, input);
    }

    // Forward for Cross-Attention (Query from Decoder, Key/Value from Encoder)
    Tensor<T, B> forward(const Tensor<T, B>& query, const Tensor<T, B>& key_value) {
        return forward_attention(query, key_value, key_value);
    }

    Tensor<T, B> forward_attention(const Tensor<T, B>& query, const Tensor<T, B>& key, const Tensor<T, B>& value, bool mask = false) {
        // Input shapes: (Batch, Seq, Dim)

        // 1. Projections
        Tensor<T, B> Q = q_proj_->forward(query);
        Tensor<T, B> K = k_proj_->forward(key);
        Tensor<T, B> V = v_proj_->forward(value);

        // Slice back if padded by DeepSpectralLinear
        if (use_deep_spectral_ && Q.shape().back() > dim_) {
            Q = Q.slice_last_dim(dim_);
            K = K.slice_last_dim(dim_);
            V = V.slice_last_dim(dim_);
        }

        // 2. Reshape and Transpose for Multi-Head
        // (Batch, Seq, Dim) -> (Batch, Seq, NumHeads, HeadDim) -> (Batch, NumHeads, Seq, HeadDim)

        // Split heads logic
        size_t batch = query.shape()[0];
        size_t seq_q = query.shape()[1];
        size_t seq_k = key.shape()[1];

        // Q: (Batch, NumHeads, SeqQ, HeadDim)
        Tensor<T, B> Q_h = split_heads(Q, batch, seq_q);
        Tensor<T, B> K_h = split_heads(K, batch, seq_k);
        Tensor<T, B> V_h = split_heads(V, batch, seq_k);

        // Attention Scores: Q * K^T / sqrt(HeadDim)
        // Q_h: (Batch, NumHeads, SeqQ, HeadDim)
        // K_h: (Batch, NumHeads, SeqK, HeadDim)
        // Result: (Batch, NumHeads, SeqQ, SeqK)

        // Tensor matmul currently supports 2D. We need to loop over Batch and Heads.
        Tensor<T, B> scores({batch, num_heads_, seq_q, seq_k});

        T scale = 1.0 / std::sqrt(static_cast<T>(head_dim_));

        T* scores_data = scores.data();
        const T* Q_data = Q_h.data();
        const T* K_data = K_h.data();

        // This is slow, but correct. For high performance, we would need batched matmul in Tensor.
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads_; ++h) {
                // Extract Q_bh: (SeqQ, HeadDim)
                // Extract K_bh: (SeqK, HeadDim)
                // Scores = Q_bh * K_bh^T

                size_t offset_bh_scores = b * num_heads_ * seq_q * seq_k + h * seq_q * seq_k;
                size_t offset_bh_q = b * num_heads_ * seq_q * head_dim_ + h * seq_q * head_dim_;
                size_t offset_bh_k = b * num_heads_ * seq_k * head_dim_ + h * seq_k * head_dim_;

                for (size_t i = 0; i < seq_q; ++i) {
                    for (size_t j = 0; j < seq_k; ++j) {
                        T dot = 0;
                        for (size_t d = 0; d < head_dim_; ++d) {
                            dot += Q_data[offset_bh_q + i * head_dim_ + d] * K_data[offset_bh_k + j * head_dim_ + d];
                        }
                        scores_data[offset_bh_scores + i * seq_k + j] = dot * scale;
                    }
                }
            }
        }

        // Masking (Causal) if enabled
        if (mask) {
             for (size_t b = 0; b < batch; ++b) {
                for (size_t h = 0; h < num_heads_; ++h) {
                    size_t offset_bh_scores = b * num_heads_ * seq_q * seq_k + h * seq_q * seq_k;
                    for (size_t i = 0; i < seq_q; ++i) {
                        for (size_t j = 0; j < seq_k; ++j) {
                            if (j > i) { // Simple causal mask assuming seq_q == seq_k
                                scores_data[offset_bh_scores + i * seq_k + j] = -1e9;
                            }
                        }
                    }
                }
             }
        }

        // Softmax over last dimension (SeqK)
        // scores: (Batch, NumHeads, SeqQ, SeqK)
        // We can flatten last dim or use a Softmax layer that supports axis.
        // Our Softmax layer assumes last dim.
        Softmax<T, B> softmax;
        // Reshape to (Batch * NumHeads * SeqQ, SeqK)
        // We assume Softmax layer operates on the last dimension of whatever shape we pass.
        // The Tensor Softmax logic just needs last dim size.
        scores = softmax.forward(scores);

        // Weighted Sum: Scores * V
        // Scores: (Batch, NumHeads, SeqQ, SeqK)
        // V_h: (Batch, NumHeads, SeqK, HeadDim)
        // Result: (Batch, NumHeads, SeqQ, HeadDim)

        Tensor<T, B> context({batch, num_heads_, seq_q, head_dim_});
        context.fill(0);
        T* context_data = context.data();
        scores_data = scores.data(); // Reload after softmax (potentially new tensor)
        const T* V_data = V_h.data();

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads_; ++h) {
                size_t offset_bh_scores = b * num_heads_ * seq_q * seq_k + h * seq_q * seq_k;
                size_t offset_bh_v = b * num_heads_ * seq_k * head_dim_ + h * seq_k * head_dim_;
                size_t offset_bh_ctx = b * num_heads_ * seq_q * head_dim_ + h * seq_q * head_dim_;

                for (size_t i = 0; i < seq_q; ++i) {
                    for (size_t d = 0; d < head_dim_; ++d) {
                         T sum = 0;
                         for (size_t j = 0; j < seq_k; ++j) {
                             sum += scores_data[offset_bh_scores + i * seq_k + j] * V_data[offset_bh_v + j * head_dim_ + d];
                         }
                         context_data[offset_bh_ctx + i * head_dim_ + d] = sum;
                    }
                }
            }
        }

        // Merge Heads
        // (Batch, NumHeads, SeqQ, HeadDim) -> (Batch, SeqQ, NumHeads, HeadDim) -> (Batch, SeqQ, Dim)
        Tensor<T, B> output_merged = merge_heads(context, batch, seq_q);

        // Output Projection
        Tensor<T, B> out = o_proj_->forward(output_merged);
        if (use_deep_spectral_ && out.shape().back() > dim_) {
            out = out.slice_last_dim(dim_);
        }
        return out;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // Not implemented for Phase 2, usually handled by autograd or manual chain rule
        // For SpectralViT it was manually implemented block-wise.
        return grad_output;
    }

    std::vector<Tensor<T, B>*> parameters() override {
        std::vector<Tensor<T, B>*> params;
        auto p_q = q_proj_->parameters(); params.insert(params.end(), p_q.begin(), p_q.end());
        auto p_k = k_proj_->parameters(); params.insert(params.end(), p_k.begin(), p_k.end());
        auto p_v = v_proj_->parameters(); params.insert(params.end(), p_v.begin(), p_v.end());
        auto p_o = o_proj_->parameters(); params.insert(params.end(), p_o.begin(), p_o.end());
        return params;
    }

    std::vector<Tensor<T, B>*> gradients() override {
        // ... similar to parameters
         std::vector<Tensor<T, B>*> grads;
        auto g_q = q_proj_->gradients(); grads.insert(grads.end(), g_q.begin(), g_q.end());
        auto g_k = k_proj_->gradients(); grads.insert(grads.end(), g_k.begin(), g_k.end());
        auto g_v = v_proj_->gradients(); grads.insert(grads.end(), g_v.begin(), g_v.end());
        auto g_o = o_proj_->gradients(); grads.insert(grads.end(), g_o.begin(), g_o.end());
        return grads;
    }

    std::vector<Tensor<T, B>*> curvatures() override {
         std::vector<Tensor<T, B>*> curvs;
        auto c_q = q_proj_->curvatures(); curvs.insert(curvs.end(), c_q.begin(), c_q.end());
        auto c_k = k_proj_->curvatures(); curvs.insert(curvs.end(), c_k.begin(), c_k.end());
        auto c_v = v_proj_->curvatures(); curvs.insert(curvs.end(), c_v.begin(), c_v.end());
        auto c_o = o_proj_->curvatures(); curvs.insert(curvs.end(), c_o.begin(), c_o.end());
        return curvs;
    }

    std::string name() const override { return "MultiHeadAttentionSpectral"; }

private:
    size_t dim_;
    size_t num_heads_;
    size_t head_dim_;
    bool use_deep_spectral_;

    std::shared_ptr<Layer<T, B>> q_proj_;
    std::shared_ptr<Layer<T, B>> k_proj_;
    std::shared_ptr<Layer<T, B>> v_proj_;
    std::shared_ptr<Layer<T, B>> o_proj_;

    // Helper to split heads: (Batch, Seq, Dim) -> (Batch, NumHeads, Seq, HeadDim)
    Tensor<T, B> split_heads(const Tensor<T, B>& input, size_t batch, size_t seq) {
        Tensor<T, B> output({batch, num_heads_, seq, head_dim_});
        const T* in_ptr = input.data();
        T* out_ptr = output.data();

        // Input layout: Batch -> Seq -> NumHeads -> HeadDim (Dim = NumHeads * HeadDim)
        // Output layout: Batch -> NumHeads -> Seq -> HeadDim

        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                for (size_t h = 0; h < num_heads_; ++h) {
                    for (size_t d = 0; d < head_dim_; ++d) {
                        // Input index: b * Seq * Dim + s * Dim + h * HeadDim + d
                        // Output index: b * Heads * Seq * HeadDim + h * Seq * HeadDim + s * HeadDim + d
                        size_t in_idx = b * seq * dim_ + s * dim_ + h * head_dim_ + d;
                        size_t out_idx = b * num_heads_ * seq * head_dim_ + h * seq * head_dim_ + s * head_dim_ + d;
                        out_ptr[out_idx] = in_ptr[in_idx];
                    }
                }
            }
        }
        return output;
    }

    // Helper to merge heads: (Batch, NumHeads, Seq, HeadDim) -> (Batch, Seq, Dim)
    Tensor<T, B> merge_heads(const Tensor<T, B>& input, size_t batch, size_t seq) {
        Tensor<T, B> output({batch, seq, dim_});
        const T* in_ptr = input.data();
        T* out_ptr = output.data();

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads_; ++h) {
                for (size_t s = 0; s < seq; ++s) {
                    for (size_t d = 0; d < head_dim_; ++d) {
                         size_t in_idx = b * num_heads_ * seq * head_dim_ + h * seq * head_dim_ + s * head_dim_ + d;
                         size_t out_idx = b * seq * dim_ + s * dim_ + h * head_dim_ + d;
                         out_ptr[out_idx] = in_ptr[in_idx];
                    }
                }
            }
        }
        return output;
    }
};

} // namespace layers
} // namespace dreidel

#endif // DREIDEL_LAYERS_MULTI_HEAD_ATTENTION_SPECTRAL_HPP
