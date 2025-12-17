#ifndef DREIDEL_MODELS_SPECTRAL_VIT_HPP
#define DREIDEL_MODELS_SPECTRAL_VIT_HPP

#include "../layers/LinearWHT.hpp"
#include "../layers/DeepSpectralLinear.hpp"
#include "../layers/Bias.hpp"
#include "../layers/ReLU.hpp"
#include "../layers/Softmax.hpp"
#include "../layers/Layer.hpp"
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>

namespace dreidel {
namespace models {

template <typename T, BackendType B = BackendType::CPU>
class SpectralViT {
public:
    SpectralViT() : num_blocks_(0) {}

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

            if (type == "LinearWHT") {
                auto layer = std::make_shared<layers::LinearWHT<T, B>>(dim);
                read_tensor(f, layer->parameters()[0]);
                layers_[name] = layer;

                bool has_bias;
                f.read(reinterpret_cast<char*>(&has_bias), 1);
                if (has_bias) {
                    auto bias_layer = std::make_shared<layers::Bias<T, B>>(dim);
                    read_tensor(f, bias_layer->parameters()[0]);
                    layers_[name + ".bias"] = bias_layer;
                }
            } else if (type == "DeepSpectralLinear") {
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
                std::cerr << "Unknown layer type: " << type << ". Skipping." << std::endl;
                throw std::runtime_error("Unknown layer type encountered.");
            }
        }

        // Detect num blocks
        int max_layer = -1;
        for (auto const& [key, val] : layers_) {
            if (key.find("encoder.layer.") == 0) {
                 size_t first_dot = 14;
                 size_t second_dot = key.find('.', first_dot);
                 if (second_dot != std::string::npos) {
                     std::string num_str = key.substr(first_dot, second_dot - first_dot);
                     try {
                         int num = std::stoi(num_str);
                         if (num > max_layer) max_layer = num;
                     } catch(...) {}
                 }
            }
        }
        num_blocks_ = max_layer + 1;
        std::cout << "Detected " << num_blocks_ << " encoder blocks." << std::endl;
    }

    int get_num_blocks() const { return num_blocks_; }

    std::vector<Tensor<T, B>*> parameters() {
        std::vector<Tensor<T, B>*> params;
        for (auto& kv : layers_) {
            auto layer_params = kv.second->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    std::vector<Tensor<T, B>*> gradients() {
        std::vector<Tensor<T, B>*> grads;
        for (auto& kv : layers_) {
            auto layer_grads = kv.second->gradients();
            grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
        }
        return grads;
    }

    std::vector<Tensor<T, B>*> curvatures() {
        std::vector<Tensor<T, B>*> curvs;
        for (auto& kv : layers_) {
            auto layer_curvs = kv.second->curvatures();
            curvs.insert(curvs.end(), layer_curvs.begin(), layer_curvs.end());
        }
        return curvs;
    }

    Tensor<T, B> forward(const Tensor<T, B>& input) {
        Tensor<T, B> x = input;
        for (int i = 0; i < num_blocks_; ++i) {
            x = forward_block(i, x);
        }
        x = forward_pooler(x);
        return x;
    }

    Tensor<T, B> forward_block(int i, const Tensor<T, B>& input) {
        Tensor<T, B> x = input;
        std::string prefix = "encoder.layer." + std::to_string(i) + ".";

        auto q_layer = get_layer(prefix + "attention.attention.query");
        auto k_layer = get_layer(prefix + "attention.attention.key");
        auto v_layer = get_layer(prefix + "attention.attention.value");

        Tensor<T, B> q = forward_linear(q_layer, x, prefix + "attention.attention.query");
        Tensor<T, B> k = forward_linear(k_layer, x, prefix + "attention.attention.key");
        Tensor<T, B> v = forward_linear(v_layer, x, prefix + "attention.attention.value");

        auto attn_out_layer = get_layer(prefix + "attention.output.dense");
        Tensor<T, B> attn_output = forward_linear(attn_out_layer, q, prefix + "attention.output.dense");

        if (attn_output.shape().back() > x.shape().back()) {
            attn_output = attn_output.slice_last_dim(x.shape().back());
        }

        x = x + attn_output;

        auto inter_layer = get_layer(prefix + "intermediate.dense");
        Tensor<T, B> mlp_hidden = forward_linear(inter_layer, x, prefix + "intermediate.dense");

        layers::ReLU<T, B> relu;
        mlp_hidden = relu.forward(mlp_hidden);

        auto out_layer = get_layer(prefix + "output.dense");
        Tensor<T, B> mlp_out = forward_linear(out_layer, mlp_hidden, prefix + "output.dense");

        if (mlp_out.shape().back() > x.shape().back()) {
            mlp_out = mlp_out.slice_last_dim(x.shape().back());
        }

        x = x + mlp_out;
        return x;
    }

    Tensor<T, B> forward_pooler(const Tensor<T, B>& input) {
        // Input is (Batch, Seq, Dim)
        // We need to extract the CLS token (index 0 along Seq dim)
        // Result should be (Batch, Dim)

        size_t Batch = input.shape()[0];
        size_t Seq = input.shape()[1];
        size_t Dim = input.shape()[2];

        Tensor<T, B> cls_token({Batch, Dim});

        const T* in_data = input.data();
        T* cls_data = cls_token.data();

        for (size_t b = 0; b < Batch; ++b) {
            // Copy Dim elements from start of each sequence
            const T* src = in_data + b * Seq * Dim;
            T* dst = cls_data + b * Dim;
            std::copy(src, src + Dim, dst);
        }

        auto pooler = get_layer("pooler.dense");
        Tensor<T, B> out = forward_linear(pooler, cls_token, "pooler.dense");
        // Slice if needed to match input dim (which corresponds to embedding dim)
        if (out.shape().back() > Dim) {
            out = out.slice_last_dim(Dim);
        }
        return out;
    }

    void backward_block(int i, const Tensor<T, B>& grad_output) {
        Tensor<T, B> grad = grad_output;
        std::string prefix = "encoder.layer." + std::to_string(i) + ".";

        // We need to know the spectral dimensions to pad gradients correctly.
        // We can inspect the layers to find their spectral dimensions.
        auto out_layer = get_layer(prefix + "output.dense");
        size_t spectral_dim_out = out_layer->parameters()[0]->size(); // scale is size dim

        auto inter_layer = get_layer(prefix + "intermediate.dense");
        size_t spectral_dim_inter = inter_layer->parameters()[0]->size();

        auto attn_out_layer = get_layer(prefix + "attention.output.dense");
        size_t spectral_dim_attn = attn_out_layer->parameters()[0]->size();

        auto q_layer = get_layer(prefix + "attention.attention.query");
        size_t spectral_dim_q = q_layer->parameters()[0]->size();

        Tensor<T, B> grad_mlp_out = grad;
        Tensor<T, B> grad_skip_2 = grad;

        Tensor<T, B> grad_mlp_hidden = backward_linear(out_layer, grad_mlp_out, prefix + "output.dense", spectral_dim_out);

        // Skip ReLU backward (identity)

        Tensor<T, B> grad_inter = backward_linear(inter_layer, grad_mlp_hidden, prefix + "intermediate.dense", spectral_dim_inter);

        grad = grad_skip_2 + grad_inter;

        Tensor<T, B> grad_attn_out = grad;
        Tensor<T, B> grad_skip_1 = grad;

        Tensor<T, B> grad_q = backward_linear(attn_out_layer, grad_attn_out, prefix + "attention.output.dense", spectral_dim_attn);

        Tensor<T, B> grad_x_q = backward_linear(q_layer, grad_q, prefix + "attention.attention.query", spectral_dim_q);

        Tensor<T, B> dummy_grad(grad_q.shape());
        dummy_grad.fill(0);
        auto k_layer = get_layer(prefix + "attention.attention.key");
        backward_linear(k_layer, dummy_grad, prefix + "attention.attention.key", spectral_dim_q);

        auto v_layer = get_layer(prefix + "attention.attention.value");
        backward_linear(v_layer, dummy_grad, prefix + "attention.attention.value", spectral_dim_q);

        grad = grad_skip_1 + grad_x_q;
    }

    void backward_pooler(const Tensor<T, B>& grad_output) {
        auto pooler = get_layer("pooler.dense");
        size_t spectral_dim = pooler->parameters()[0]->size();
        backward_linear(pooler, grad_output, "pooler.dense", spectral_dim);
    }

    std::vector<Tensor<T, B>*> parameters_block(int i) {
        std::vector<Tensor<T, B>*> params;
        std::string prefix = "encoder.layer." + std::to_string(i) + ".";

        std::vector<std::string> sublayers = {
            "attention.attention.query", "attention.attention.key", "attention.attention.value",
            "attention.output.dense", "intermediate.dense", "output.dense"
        };

        for (const auto& sub : sublayers) {
            std::string name = prefix + sub;
            auto p = get_layer(name)->parameters();
            params.insert(params.end(), p.begin(), p.end());

             if (layers_.find(name + ".bias") != layers_.end()) {
                auto pb = layers_[name + ".bias"]->parameters();
                params.insert(params.end(), pb.begin(), pb.end());
            }
        }
        return params;
    }

    std::vector<Tensor<T, B>*> gradients_block(int i) {
        std::vector<Tensor<T, B>*> grads;
        std::string prefix = "encoder.layer." + std::to_string(i) + ".";
        std::vector<std::string> sublayers = {
            "attention.attention.query", "attention.attention.key", "attention.attention.value",
            "attention.output.dense", "intermediate.dense", "output.dense"
        };
        for (const auto& sub : sublayers) {
            std::string name = prefix + sub;
            auto p = get_layer(name)->gradients();
            grads.insert(grads.end(), p.begin(), p.end());
             if (layers_.find(name + ".bias") != layers_.end()) {
                auto pb = layers_[name + ".bias"]->gradients();
                grads.insert(grads.end(), pb.begin(), pb.end());
            }
        }
        return grads;
    }

    std::vector<Tensor<T, B>*> curvatures_block(int i) {
        std::vector<Tensor<T, B>*> curvs;
        std::string prefix = "encoder.layer." + std::to_string(i) + ".";
        std::vector<std::string> sublayers = {
            "attention.attention.query", "attention.attention.key", "attention.attention.value",
            "attention.output.dense", "intermediate.dense", "output.dense"
        };
        for (const auto& sub : sublayers) {
            std::string name = prefix + sub;
            auto p = get_layer(name)->curvatures();
            curvs.insert(curvs.end(), p.begin(), p.end());
             if (layers_.find(name + ".bias") != layers_.end()) {
                auto pb = layers_[name + ".bias"]->curvatures();
                curvs.insert(curvs.end(), pb.begin(), pb.end());
            }
        }
        return curvs;
    }

    std::vector<Tensor<T, B>*> parameters_pooler() {
        auto p = get_layer("pooler.dense")->parameters();
        if (layers_.find("pooler.dense.bias") != layers_.end()) {
             auto pb = layers_["pooler.dense.bias"]->parameters();
             p.insert(p.end(), pb.begin(), pb.end());
        }
        return p;
    }

    std::vector<Tensor<T, B>*> gradients_pooler() {
        auto p = get_layer("pooler.dense")->gradients();
        if (layers_.find("pooler.dense.bias") != layers_.end()) {
             auto pb = layers_["pooler.dense.bias"]->gradients();
             p.insert(p.end(), pb.begin(), pb.end());
        }
        return p;
    }

    std::vector<Tensor<T, B>*> curvatures_pooler() {
        auto p = get_layer("pooler.dense")->curvatures();
        if (layers_.find("pooler.dense.bias") != layers_.end()) {
             auto pb = layers_["pooler.dense.bias"]->curvatures();
             p.insert(p.end(), pb.begin(), pb.end());
        }
        return p;
    }

private:
    std::map<std::string, std::shared_ptr<layers::Layer<T, B>>> layers_;
    int num_blocks_;

    std::shared_ptr<layers::Layer<T, B>> get_layer(const std::string& name) {
        if (layers_.find(name) == layers_.end()) {
            throw std::runtime_error("Layer not found: " + name);
        }
        return layers_[name];
    }

    Tensor<T, B> forward_linear(std::shared_ptr<layers::Layer<T, B>> layer, const Tensor<T, B>& input, const std::string& name) {
        Tensor<T, B> out = layer->forward(input);

        // Apply bias if exists
        std::string bias_name = name + ".bias";
        if (layers_.find(bias_name) != layers_.end()) {
            out = layers_[bias_name]->forward(out);
        }

        return out;
    }

    Tensor<T, B> backward_linear(std::shared_ptr<layers::Layer<T, B>> layer, const Tensor<T, B>& grad_output, const std::string& name, size_t expected_dim = 0) {
        Tensor<T, B> grad = grad_output;

        // Backward bias if exists
        std::string bias_name = name + ".bias";
        if (layers_.find(bias_name) != layers_.end()) {
            grad = layers_[bias_name]->backward(grad);
        }

        // Pad gradient if needed
        if (expected_dim > 0 && grad.shape().back() < expected_dim) {
            grad = grad.pad_last_dim(expected_dim);
        }

        return layer->backward(grad);
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
        if (t->size() != num_elements) {
             throw std::runtime_error("Tensor size mismatch loading weights.");
        }

        for(size_t i=0; i<num_elements; ++i) {
            t_data[i] = static_cast<T>(data[i]);
        }
    }
};

} // namespace models
} // namespace dreidel

#endif // DREIDEL_MODELS_SPECTRAL_VIT_HPP
