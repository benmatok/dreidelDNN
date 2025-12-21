#ifndef DREIDEL_MODELS_SPECTRAL_WHISPER_HPP
#define DREIDEL_MODELS_SPECTRAL_WHISPER_HPP

#include "../layers/LinearWHT.hpp"
#include "../layers/DeepSpectralLinear.hpp"
#include "../layers/Bias.hpp"
#include "../layers/GELU.hpp"
#include "../layers/Softmax.hpp"
#include "../layers/Layer.hpp"
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace dreidel {
namespace models {

template <typename T, BackendType B = BackendType::CPU>
class SpectralWhisper {
public:
    SpectralWhisper() : num_encoder_layers_(0), num_decoder_layers_(0) {}

    void load_weights(const std::string& filepath) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open weights file: " + filepath);
        }

        // Magic
        char magic[4];
        f.read(magic, 4);
        if (std::string(magic, 4) != "DRDL") {
            throw std::runtime_error("Invalid file format");
        }

        // Version
        uint32_t version;
        f.read(reinterpret_cast<char*>(&version), 4);
        if (version != 1) {
            throw std::runtime_error("Unsupported version");
        }

        // Number of layers
        uint32_t num_layers;
        f.read(reinterpret_cast<char*>(&num_layers), 4);

        for (uint32_t i = 0; i < num_layers; ++i) {
            // Name
            uint32_t name_len;
            f.read(reinterpret_cast<char*>(&name_len), 4);
            std::vector<char> name_buf(name_len);
            f.read(name_buf.data(), name_len);
            std::string name(name_buf.begin(), name_buf.end());

            // Type
            uint32_t type_len;
            f.read(reinterpret_cast<char*>(&type_len), 4);
            std::vector<char> type_buf(type_len);
            f.read(type_buf.data(), type_len);
            std::string type(type_buf.begin(), type_buf.end());

            // Dim
            uint32_t dim;
            f.read(reinterpret_cast<char*>(&dim), 4);

            if (type == "DeepSpectralLinear") {
                uint32_t depth;
                f.read(reinterpret_cast<char*>(&depth), 4);

                auto layer = std::make_shared<layers::DeepSpectralLinear<T, B>>(dim, depth);
                auto params = layer->parameters();

                for (size_t k = 0; k < depth; ++k) {
                    read_tensor(f, params[k]);
                    uint32_t perm_size;
                    f.read(reinterpret_cast<char*>(&perm_size), 4);
                    if (perm_size != dim) throw std::runtime_error("Permutation size mismatch.");

                    std::vector<uint64_t> p_data(perm_size);
                    f.read(reinterpret_cast<char*>(p_data.data()), perm_size * 8);

                    std::vector<size_t> p(perm_size);
                    for(size_t j=0; j<perm_size; ++j) p[j] = static_cast<size_t>(p_data[j]);
                    layer->set_permutation(k, p);
                }
                layers_[name] = layer;

                bool has_bias;
                f.read(reinterpret_cast<char*>(&has_bias), 1);
                if (has_bias) {
                    auto bias_layer = std::make_shared<layers::Bias<T, B>>(dim);
                    read_tensor(f, bias_layer->parameters()[0]);
                    layers_[name + ".bias"] = bias_layer;
                }
            } else {
                 // Skip other types if any, but we only generated DeepSpectralLinear
                 // Or implement LinearWHT loading if reused from ViT
                 std::cerr << "Unknown or unsupported layer type: " << type << std::endl;
                 // Need to skip data
                 // But wait, the file format is sequential. We cannot easily skip unless we know structure.
                 // Assuming only DeepSpectralLinear for now as per my script.
                 throw std::runtime_error("Unknown layer type encountered.");
            }
        }

        // Detect num layers
        int max_enc = -1;
        int max_dec = -1;
        for (auto const& [key, val] : layers_) {
            if (key.find("encoder.layers.") == 0) {
                 size_t first_dot = 15; // len("encoder.layers.")
                 size_t second_dot = key.find('.', first_dot);
                 if (second_dot != std::string::npos) {
                     std::string num_str = key.substr(first_dot, second_dot - first_dot);
                     try {
                         int num = std::stoi(num_str);
                         if (num > max_enc) max_enc = num;
                     } catch(...) {}
                 }
            }
            if (key.find("decoder.layers.") == 0) {
                 size_t first_dot = 15; // len("decoder.layers.")
                 size_t second_dot = key.find('.', first_dot);
                 if (second_dot != std::string::npos) {
                     std::string num_str = key.substr(first_dot, second_dot - first_dot);
                     try {
                         int num = std::stoi(num_str);
                         if (num > max_dec) max_dec = num;
                     } catch(...) {}
                 }
            }
        }
        num_encoder_layers_ = max_enc + 1;
        num_decoder_layers_ = max_dec + 1;
        std::cout << "Detected " << num_encoder_layers_ << " encoder layers and " << num_decoder_layers_ << " decoder layers." << std::endl;
    }

    Tensor<T, B> forward_encoder(const Tensor<T, B>& input) {
        Tensor<T, B> x = input;
        for (int i = 0; i < num_encoder_layers_; ++i) {
            x = forward_encoder_block(i, x);
        }
        // Note: Whisper has LayerNorm after encoder layers usually.
        // We are only recasting Linear layers, so we assume LayerNorm is handled outside or skipped for spectral approximation (often spectral layers absorb normalization or rely on instance norm).
        return x;
    }

    Tensor<T, B> forward_encoder_block(int i, const Tensor<T, B>& input) {
        Tensor<T, B> x = input;
        std::string prefix = "encoder.layers." + std::to_string(i) + ".";

        // Self Attention
        auto q_layer = get_layer(prefix + "self_attn.q_proj");
        auto k_layer = get_layer(prefix + "self_attn.k_proj");
        auto v_layer = get_layer(prefix + "self_attn.v_proj");

        Tensor<T, B> q = forward_linear(q_layer, x, prefix + "self_attn.q_proj");
        Tensor<T, B> k = forward_linear(k_layer, x, prefix + "self_attn.k_proj");
        Tensor<T, B> v = forward_linear(v_layer, x, prefix + "self_attn.v_proj");

        // Attention Mechanism (Simplified for Benchmark)
        // NOTE: This implementation currently skips the actual Dot-Product Attention (Softmax(QK^T)V) and LayerNorms.
        // It pipes the Query projection directly into the Output projection.
        // This is sufficient for verifying the throughput and memory of the Spectral Linear layers but invalid for ASR inference.
        // Full attention requires implementing 4D MatMul and Softmax.

        auto out_layer = get_layer(prefix + "self_attn.out_proj");
        // For benchmark/test purposes, we feed `q` into `out_proj`.
        Tensor<T, B> attn_output = forward_linear(out_layer, q, prefix + "self_attn.out_proj");

        if (attn_output.shape().back() > x.shape().back()) {
            attn_output = attn_output.slice_last_dim(x.shape().back());
        }
        x = x + attn_output; // Residual

        // MLP
        auto fc1 = get_layer(prefix + "fc1");
        Tensor<T, B> hidden = forward_linear(fc1, x, prefix + "fc1");

        layers::GELU<T, B> gelu;
        hidden = gelu.forward(hidden);

        auto fc2 = get_layer(prefix + "fc2");
        Tensor<T, B> mlp_out = forward_linear(fc2, hidden, prefix + "fc2");

        if (mlp_out.shape().back() > x.shape().back()) {
            mlp_out = mlp_out.slice_last_dim(x.shape().back());
        }
        x = x + mlp_out; // Residual

        return x;
    }

    Tensor<T, B> forward_decoder_block(int i, const Tensor<T, B>& input, const Tensor<T, B>& encoder_output) {
        Tensor<T, B> x = input;
        std::string prefix = "decoder.layers." + std::to_string(i) + ".";

        // Self Attention
        auto q_layer = get_layer(prefix + "self_attn.q_proj");
        // k, v
        Tensor<T, B> q = forward_linear(q_layer, x, prefix + "self_attn.q_proj");

        auto out_layer = get_layer(prefix + "self_attn.out_proj");
        Tensor<T, B> attn_out = forward_linear(out_layer, q, prefix + "self_attn.out_proj");
        if (attn_out.shape().back() > x.shape().back()) attn_out = attn_out.slice_last_dim(x.shape().back());
        x = x + attn_out;

        // Encoder Attention (Cross)
        // In real Transformer: Q from x, K,V from encoder_output.
        auto enc_q_layer = get_layer(prefix + "encoder_attn.q_proj");
        // auto enc_k_layer = get_layer(prefix + "encoder_attn.k_proj"); // acts on encoder_output
        // auto enc_v_layer = get_layer(prefix + "encoder_attn.v_proj"); // acts on encoder_output

        Tensor<T, B> enc_q = forward_linear(enc_q_layer, x, prefix + "encoder_attn.q_proj");

        auto enc_out_layer = get_layer(prefix + "encoder_attn.out_proj");
        Tensor<T, B> enc_attn_out = forward_linear(enc_out_layer, enc_q, prefix + "encoder_attn.out_proj");
        if (enc_attn_out.shape().back() > x.shape().back()) enc_attn_out = enc_attn_out.slice_last_dim(x.shape().back());
        x = x + enc_attn_out;

        // MLP
        auto fc1 = get_layer(prefix + "fc1");
        Tensor<T, B> hidden = forward_linear(fc1, x, prefix + "fc1");

        layers::GELU<T, B> gelu;
        hidden = gelu.forward(hidden);

        auto fc2 = get_layer(prefix + "fc2");
        Tensor<T, B> mlp_out = forward_linear(fc2, hidden, prefix + "fc2");
        if (mlp_out.shape().back() > x.shape().back()) mlp_out = mlp_out.slice_last_dim(x.shape().back());
        x = x + mlp_out;

        return x;
    }

private:
    std::map<std::string, std::shared_ptr<layers::Layer<T, B>>> layers_;
    int num_encoder_layers_;
    int num_decoder_layers_;

    std::shared_ptr<layers::Layer<T, B>> get_layer(const std::string& name) {
        if (layers_.find(name) == layers_.end()) {
            throw std::runtime_error("Layer not found: " + name);
        }
        return layers_[name];
    }

    Tensor<T, B> forward_linear(std::shared_ptr<layers::Layer<T, B>> layer, const Tensor<T, B>& input, const std::string& name) {
        Tensor<T, B> out = layer->forward(input);
        std::string bias_name = name + ".bias";
        if (layers_.find(bias_name) != layers_.end()) {
            out = layers_[bias_name]->forward(out);
        }
        return out;
    }

    void read_tensor(std::ifstream& f, Tensor<T, B>* t) {
        uint32_t rank;
        f.read(reinterpret_cast<char*>(&rank), 4);
        std::vector<size_t> shape(rank);
        for (uint32_t i = 0; i < rank; ++i) {
            uint32_t d;
            f.read(reinterpret_cast<char*>(&d), 4);
            shape[i] = d;
        }
        size_t num_elements = 1;
        for (auto s : shape) num_elements *= s;
        std::vector<float> data(num_elements);
        f.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));
        T* t_data = t->data();
        if (t->size() != num_elements) throw std::runtime_error("Tensor size mismatch.");
        for(size_t i=0; i<num_elements; ++i) t_data[i] = static_cast<T>(data[i]);
    }
};

} // namespace models
} // namespace dreidel

#endif // DREIDEL_MODELS_SPECTRAL_WHISPER_HPP
