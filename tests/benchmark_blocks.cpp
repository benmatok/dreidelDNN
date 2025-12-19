#include "../include/dreidel/layers/Dense.hpp"
#include "../include/dreidel/layers/LinearWHT.hpp"
#include "../include/dreidel/layers/DeepSpectralLinear.hpp"
#include "../include/dreidel/layers/ReLU.hpp"
#include "../include/dreidel/optim/SGD.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace dreidel;

// Abstract Block
template <typename T>
class Block {
public:
    virtual Tensor<T> forward(const Tensor<T>& input) = 0;
    virtual Tensor<T> backward(const Tensor<T>& grad_output) = 0;
    virtual std::vector<Tensor<T>*> parameters() = 0;
    virtual std::vector<Tensor<T>*> gradients() = 0;
    virtual size_t param_count() = 0;
    virtual std::string name() = 0;
    virtual ~Block() {}
};

// Standard Dense MLP: Linear(d_in, d_mid) -> ReLU -> Linear(d_mid, d_out)
// Modifying to handle 3D inputs (B, S, D) by flattening to (B*S, D)
template <typename T>
class DenseMLP : public Block<T> {
    layers::Dense<T> l1, l2;
    layers::ReLU<T> act;
    Tensor<T> hidden;
    std::vector<size_t> original_shape;
public:
    DenseMLP(size_t d_in, size_t d_mid, size_t d_out)
        : l1(d_in, d_mid), l2(d_mid, d_out) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        original_shape = input.shape();
        Tensor<T> flat_input = input;

        // Flatten to 2D if rank > 2
        if (original_shape.size() > 2) {
            size_t last_dim = original_shape.back();
            size_t total = input.size();
            size_t rows = total / last_dim;

            // We can just construct a new tensor header with shared data if we had views,
            // but here we do a copy/reshape via constructor or just hack the shape if it was mutable public.
            // Since Tensor copies in constructor, let's just make a new one with correct shape.
            flat_input = Tensor<T>({rows, last_dim});
            std::copy(input.data(), input.data() + total, flat_input.data());
        }

        hidden = act.forward(l1.forward(flat_input));
        Tensor<T> out = l2.forward(hidden);

        // Reshape output back to (..., d_out)
        if (original_shape.size() > 2) {
            std::vector<size_t> out_shape = original_shape;
            out_shape.back() = out.shape().back(); // d_out

            Tensor<T> reshaped_out(out_shape);
            std::copy(out.data(), out.data() + out.size(), reshaped_out.data());
            return reshaped_out;
        }

        return out;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> flat_grad = grad_output;

        // Flatten grad if needed
        if (grad_output.shape().size() > 2) {
            size_t last_dim = grad_output.shape().back();
            size_t total = grad_output.size();
            size_t rows = total / last_dim;
            flat_grad = Tensor<T>({rows, last_dim});
            std::copy(grad_output.data(), grad_output.data() + total, flat_grad.data());
        }

        Tensor<T> d_hidden = l2.backward(flat_grad);
        Tensor<T> d_act = act.backward(d_hidden);
        Tensor<T> d_in_grad = l1.backward(d_act);

        // Reshape grad input back
        if (original_shape.size() > 2) {
             Tensor<T> reshaped_grad(original_shape);
             std::copy(d_in_grad.data(), d_in_grad.data() + d_in_grad.size(), reshaped_grad.data());
             return reshaped_grad;
        }

        return d_in_grad;
    }

    std::vector<Tensor<T>*> parameters() override {
        auto p1 = l1.parameters();
        auto p2 = l2.parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }

    std::vector<Tensor<T>*> gradients() override {
        auto g1 = l1.gradients();
        auto g2 = l2.gradients();
        g1.insert(g1.end(), g2.begin(), g2.end());
        return g1;
    }

    size_t param_count() override {
        size_t c = 0;
        for(auto* p : parameters()) c += p->size();
        return c;
    }
    std::string name() override { return "Dense MLP"; }
};

// Spectral MLP: LinearWHT(d_mid) -> ReLU -> LinearWHT(d_mid)
// Note: Input is padded to d_mid if smaller. Output sliced if needed.
template <typename T>
class SpectralMLP : public Block<T> {
    layers::LinearWHT<T> l1, l2;
    layers::ReLU<T> act;
    size_t d_in, d_out;
    Tensor<T> hidden;
public:
    SpectralMLP(size_t d_in, size_t d_mid, size_t d_out)
        : l1(d_mid), l2(d_mid), d_in(d_in), d_out(d_out) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        // LinearWHT auto-pads input to d_mid
        hidden = act.forward(l1.forward(input));
        Tensor<T> out = l2.forward(hidden);
        // Slice if needed
        if (out.shape().back() > d_out) {
            out = out.slice_last_dim(d_out);
        }
        return out;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> grad = grad_output;
        // Pad gradient to d_mid
        if (grad.shape().back() < l2.parameters()[0]->size()) {
             grad = grad.pad_last_dim(l2.parameters()[0]->size());
        }
        Tensor<T> d_hidden = l2.backward(grad);
        Tensor<T> d_act = act.backward(d_hidden);
        Tensor<T> d_in_grad = l1.backward(d_act);

        // Slice gradient to input dim
        if (d_in_grad.shape().back() > d_in) {
            d_in_grad = d_in_grad.slice_last_dim(d_in);
        }
        return d_in_grad;
    }

    std::vector<Tensor<T>*> parameters() override {
        auto p1 = l1.parameters();
        auto p2 = l2.parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }

    std::vector<Tensor<T>*> gradients() override {
        auto g1 = l1.gradients();
        auto g2 = l2.gradients();
        g1.insert(g1.end(), g2.begin(), g2.end());
        return g1;
    }
    size_t param_count() override {
        size_t c = 0;
        for(auto* p : parameters()) c += p->size();
        return c;
    }
    std::string name() override { return "Spectral MLP (LinearWHT)"; }
};

// Deep Spectral MLP
template <typename T>
class DeepSpectralMLP : public Block<T> {
    layers::DeepSpectralLinear<T> l1, l2;
    layers::ReLU<T> act;
    size_t d_in, d_out;
    Tensor<T> hidden;
public:
    DeepSpectralMLP(size_t d_in, size_t d_mid, size_t d_out, size_t depth=4)
        : l1(d_mid, depth), l2(d_mid, depth), d_in(d_in), d_out(d_out) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        hidden = act.forward(l1.forward(input));
        Tensor<T> out = l2.forward(hidden);
        if (out.shape().back() > d_out) {
            out = out.slice_last_dim(d_out);
        }
        return out;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        Tensor<T> grad = grad_output;
        if (grad.shape().back() < l2.parameters()[0]->size()) { // size of scale is d_mid
             grad = grad.pad_last_dim(l2.parameters()[0]->size());
        }
        Tensor<T> d_hidden = l2.backward(grad);
        Tensor<T> d_act = act.backward(d_hidden);
        Tensor<T> d_in_grad = l1.backward(d_act);
        if (d_in_grad.shape().back() > d_in) {
            d_in_grad = d_in_grad.slice_last_dim(d_in);
        }
        return d_in_grad;
    }

    std::vector<Tensor<T>*> parameters() override {
        auto p1 = l1.parameters();
        auto p2 = l2.parameters();
        p1.insert(p1.end(), p2.begin(), p2.end());
        return p1;
    }

    std::vector<Tensor<T>*> gradients() override {
        auto g1 = l1.gradients();
        auto g2 = l2.gradients();
        g1.insert(g1.end(), g2.begin(), g2.end());
        return g1;
    }
    size_t param_count() override {
        size_t c = 0;
        for(auto* p : parameters()) c += p->size();
        return c;
    }
    std::string name() override { return "Deep Spectral MLP"; }
};

int main() {
    // Config
    size_t BATCH = 4;
    size_t SEQ = 64;
    size_t D_IN = 768;
    size_t D_MID_DENSE = 3072;
    size_t D_MID_SPECTRAL = 4096; // Next power of 2 >= 3072
    size_t D_OUT = 768;

    // Synthetic Data
    Tensor<float> input({BATCH, SEQ, D_IN});
    input.random(0, 1);

    // Generate Target using a Teacher Dense MLP
    // We want a non-trivial target that depends on input non-linearly.
    DenseMLP<float> teacher(D_IN, D_MID_DENSE, D_OUT);
    Tensor<float> target = teacher.forward(input); // Deterministic random target

    // Models to Benchmark
    std::vector<Block<float>*> models;
    models.push_back(new DenseMLP<float>(D_IN, D_MID_DENSE, D_OUT));
    models.push_back(new SpectralMLP<float>(D_IN, D_MID_SPECTRAL, D_OUT));
    models.push_back(new DeepSpectralMLP<float>(D_IN, D_MID_SPECTRAL, D_OUT, 4));

    std::cout << "--- Benchmark: Efficiency vs Discrimination ---" << std::endl;
    std::cout << "Input: (" << BATCH << ", " << SEQ << ", " << D_IN << ")" << std::endl;
    std::cout << "Teacher: Dense MLP (" << D_IN << "->" << D_MID_DENSE << "->" << D_OUT << ")" << std::endl;
    std::cout << std::endl;
    std::cout << std::left << std::setw(30) << "Model"
              << std::setw(15) << "Params"
              << std::setw(15) << "Time/Step(ms)"
              << std::setw(15) << "Initial Loss"
              << std::setw(15) << "Final Loss"
              << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    for (auto* model : models) {
        optim::SGD<float> optimizer(0.01); // Simple SGD
        optimizer.add_parameters(model->parameters(), model->gradients());

        // Measure Params
        size_t params = model->param_count();

        // Training Loop
        int steps = 10; // Reduced steps for speed
        double total_time = 0;
        float initial_loss = 0;
        float final_loss = 0;

        for (int i = 0; i < steps; ++i) {
            auto start = std::chrono::high_resolution_clock::now();

            // Forward
            Tensor<float> out = model->forward(input);

            // Loss (MSE) and Grad
            float loss = 0;
            size_t n = out.size();
            Tensor<float> grad_out(out.shape());

            // Manual loop for MSE and grad
            float* o_ptr = out.data();
            float* t_ptr = target.data();
            float* g_ptr = grad_out.data();

            #pragma omp parallel for reduction(+:loss)
            for(long j=0; j<(long)n; ++j) {
                float d = o_ptr[j] - t_ptr[j];
                loss += d * d;
                g_ptr[j] = 2.0f * d / n;
            }
            loss /= n;

            if (i == 0) initial_loss = loss;
            final_loss = loss;

            // Backward
            optimizer.zero_grad();
            model->backward(grad_out);
            optimizer.step();

            auto end = std::chrono::high_resolution_clock::now();
            if (i > 0) { // Skip first iter warmup
                 total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            }
        }

        double avg_time = total_time / (steps - 1);

        std::cout << std::left << std::setw(30) << model->name()
                  << std::setw(15) << params
                  << std::setw(15) << avg_time
                  << std::setw(15) << initial_loss
                  << std::setw(15) << final_loss
                  << std::endl;

        delete model;
    }

    return 0;
}
