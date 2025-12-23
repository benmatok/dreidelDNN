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

    Tensor<T, B> forward(const Tensor<T, B>& input) override {
        return forward_attention(input, input, input);
    }

    Tensor<T, B> forward(const Tensor<T, B>& query, const Tensor<T, B>& key_value) {
        return forward_attention(query, key_value, key_value);
    }

    Tensor<T, B> forward_attention(const Tensor<T, B>& query, const Tensor<T, B>& key, const Tensor<T, B>& value, bool mask = false) {
        cached_v_shape_ = value.shape();

        // 1. Projections
        Tensor<T, B> Q = q_proj_->forward(query);
        Tensor<T, B> K = k_proj_->forward(key);
        Tensor<T, B> V = v_proj_->forward(value);

        if (use_deep_spectral_ && Q.shape().back() > dim_) {
            Q = Q.slice_last_dim(dim_);
            K = K.slice_last_dim(dim_);
            V = V.slice_last_dim(dim_);
        }

        size_t batch = query.shape()[0];
        size_t seq_q = query.shape()[1];
        size_t seq_k = key.shape()[1];

        Tensor<T, B> Q_h = split_heads(Q, batch, seq_q);
        Tensor<T, B> K_h = split_heads(K, batch, seq_k);
        Tensor<T, B> V_h = split_heads(V, batch, seq_k);

        Tensor<T, B> scores({batch, num_heads_, seq_q, seq_k});
        T scale = 1.0 / std::sqrt(static_cast<T>(head_dim_));

        T* scores_data = scores.data();
        const T* Q_data = Q_h.data();
        const T* K_data = K_h.data();

        #pragma omp parallel for collapse(2)
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < num_heads_; ++h) {
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

        if (mask) {
             #pragma omp parallel for collapse(2)
             for (size_t b = 0; b < batch; ++b) {
                for (size_t h = 0; h < num_heads_; ++h) {
                    size_t offset_bh_scores = b * num_heads_ * seq_q * seq_k + h * seq_q * seq_k;
                    for (size_t i = 0; i < seq_q; ++i) {
                        for (size_t j = 0; j < seq_k; ++j) {
                            if (j > i) {
                                scores_data[offset_bh_scores + i * seq_k + j] = -1e9;
                            }
                        }
                    }
                }
             }
        }

        Softmax<T, B> softmax;
        scores = softmax.forward(scores);

        Tensor<T, B> context({batch, num_heads_, seq_q, head_dim_});
        context.fill(0);
        T* context_data = context.data();
        scores_data = scores.data();
        const T* V_data = V_h.data();

        #pragma omp parallel for collapse(2)
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

        Tensor<T, B> output_merged = merge_heads(context, batch, seq_q);
        Tensor<T, B> out = o_proj_->forward(output_merged);
        if (use_deep_spectral_ && out.shape().back() > dim_) {
            out = out.slice_last_dim(dim_);
        }
        return out;
    }

    Tensor<T, B> backward(const Tensor<T, B>& grad_output) override {
        // Approximate Backward:
        // 1. Backprop through O_proj
        Tensor<T, B> grad_context = grad_output;

        size_t o_dim = o_proj_->parameters()[0]->shape()[1];
        if (grad_context.shape().back() < o_dim) {
            grad_context = grad_context.pad_last_dim(o_dim);
        }

        grad_context = o_proj_->backward(grad_context);

        if (grad_context.shape().back() > dim_) {
            grad_context = grad_context.slice_last_dim(dim_);
        }

        // 2. Backprop through V (Only if shapes match)
        // Check against cached V shape
        bool shape_match = true;
        if (grad_context.shape().size() != cached_v_shape_.size()) shape_match = false;
        else {
            for(size_t i=0; i<grad_context.shape().size(); ++i) {
                // Ignore last dim mismatch if we are about to pad?
                // cached_v_shape_ last dim is dim_. grad_context last dim is dim_.
                // Dimensions 0...Rank-2 should match (Batch, Seq).
                if (i < grad_context.shape().size() - 1) {
                    if (grad_context.shape()[i] != cached_v_shape_[i]) shape_match = false;
                }
            }
        }

        Tensor<T, B> grad_v = grad_context;

        if (shape_match) {
            size_t v_dim = v_proj_->parameters()[0]->shape()[1];
            if (grad_v.shape().back() < v_dim) {
                 grad_v = grad_v.pad_last_dim(v_dim);
            }

            grad_v = v_proj_->backward(grad_v);

            if (grad_v.shape().back() > dim_) {
                grad_v = grad_v.slice_last_dim(dim_);
            }
        } else {
            // Mismatch (e.g. Cross Attn). Stop V training.
            // Return Zeros?
            // If we return grad_v (context), it flows back to Query.
            // If we stop V training, we should still flow back to Query?
            // dL/dQ comes from dL/dScores.
            // We ignored dL/dScores.
            // So dL/dQ = 0.
            // We return grad_v as "dL/dInput".
            // If Self Attn: Input=Q=V. dL/dInput += dL/dV.
            // If Cross Attn: Input=Q. V=KeyVal.
            // dL/dQuery = 0 (in our approx).
            // So we should return 0 if mismatch?
            // Or return grad_context if we assume O_proj output aligns with Q?
            // Usually O_proj(Context) aligns with Q in terms of residual.
            // So passing grad_context back as grad_q is a rough approx "Identity Attention".

            // Safe bet: If mismatch, we are in Cross Attn. We return gradient wrt Query.
            // Context has same shape as Query.
            // So we can return grad_context (unprocessed by V_proj).
            // This assumes "Attention passes Query through".
            // Which is reasonable for initialization (near identity).

            // So we just return grad_v (which is grad_context).
        }

        return grad_v;
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

    std::shared_ptr<DeepSpectralLinear<T, B>> q_proj_public() { return std::dynamic_pointer_cast<DeepSpectralLinear<T, B>>(q_proj_); }
    std::shared_ptr<DeepSpectralLinear<T, B>> k_proj_public() { return std::dynamic_pointer_cast<DeepSpectralLinear<T, B>>(k_proj_); }
    std::shared_ptr<DeepSpectralLinear<T, B>> v_proj_public() { return std::dynamic_pointer_cast<DeepSpectralLinear<T, B>>(v_proj_); }
    std::shared_ptr<DeepSpectralLinear<T, B>> o_proj_public() { return std::dynamic_pointer_cast<DeepSpectralLinear<T, B>>(o_proj_); }

private:
    size_t dim_;
    size_t num_heads_;
    size_t head_dim_;
    bool use_deep_spectral_;

    std::shared_ptr<Layer<T, B>> q_proj_;
    std::shared_ptr<Layer<T, B>> k_proj_;
    std::shared_ptr<Layer<T, B>> v_proj_;
    std::shared_ptr<Layer<T, B>> o_proj_;

    std::vector<size_t> cached_v_shape_;

    Tensor<T, B> split_heads(const Tensor<T, B>& input, size_t batch, size_t seq) {
        Tensor<T, B> output({batch, num_heads_, seq, head_dim_});
        const T* in_ptr = input.data();
        T* out_ptr = output.data();

        #pragma omp parallel for collapse(2)
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                for (size_t h = 0; h < num_heads_; ++h) {
                    for (size_t d = 0; d < head_dim_; ++d) {
                        size_t in_idx = b * seq * dim_ + s * dim_ + h * head_dim_ + d;
                        size_t out_idx = b * num_heads_ * seq * head_dim_ + h * seq * head_dim_ + s * head_dim_ + d;
                        out_ptr[out_idx] = in_ptr[in_idx];
                    }
                }
            }
        }
        return output;
    }

    Tensor<T, B> merge_heads(const Tensor<T, B>& input, size_t batch, size_t seq) {
        Tensor<T, B> output({batch, seq, dim_});
        const T* in_ptr = input.data();
        T* out_ptr = output.data();

        #pragma omp parallel for collapse(2)
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
