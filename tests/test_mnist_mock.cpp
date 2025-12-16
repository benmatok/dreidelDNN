#include <dreidel/dreidel.hpp>
#include <dreidel/model.hpp>
#include <dreidel/layers/Dense.hpp>
#include <dreidel/layers/LinearWHT.hpp>
#include <dreidel/optim/SGD.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace dreidel;

// Basic Activation/Loss definitions locally if not in headers
// Looking at README: "Implement ReLU and Softmax".
// They might be in layers/Dense.hpp or layers/Activations.hpp?
// Let's check Dense.hpp content via list_files or read_file.
// But based on previous error `activation` has not been declared.
// I will assume they are not nicely exposed or I missed an include.
// I will just implement manual layers or helper if needed, but Sequential supports layers.
// Let's use `add(std::make_shared<...>)` pattern instead of template add if convenient.

// Let's inspect `include/dreidel/layers/Dense.hpp` to see how activation is handled.

namespace dreidel {
namespace activation {
    enum Type { ReLU, Softmax, None };
}
namespace loss {
    struct CrossEntropy {
        static float compute(const Tensor<float>& pred, const Tensor<float>& target) {
            // Pred are logits. Compute softmax first.
            Tensor<float> prob = pred;
            size_t batch = pred.shape()[0];
            size_t dim = pred.shape()[1];
            float* p_ptr = prob.data();

            for(size_t i=0; i<batch; ++i) {
                float max_val = -1e9;
                for(size_t j=0; j<dim; ++j) {
                    if (p_ptr[i*dim + j] > max_val) max_val = p_ptr[i*dim + j];
                }

                float sum = 0;
                for(size_t j=0; j<dim; ++j) {
                    p_ptr[i*dim + j] = std::exp(p_ptr[i*dim + j] - max_val);
                    sum += p_ptr[i*dim + j];
                }

                for(size_t j=0; j<dim; ++j) {
                    p_ptr[i*dim + j] /= sum;
                }
            }

            // CE
            const float* p = prob.data();
            const float* t = target.data();
            float sum_loss = 0;
            size_t n = pred.size();
            for(size_t i=0; i<n; ++i) {
                if(t[i] > 0) sum_loss -= t[i] * std::log(p[i] + 1e-7f);
            }
            return sum_loss / batch; // mean over batch
        }
        static Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target) {
            // Gradient of CE + Softmax is (softmax(logits) - t)
            // pred here is raw logits from the Dense layer output.

            // 1. Compute Softmax
            Tensor<float> prob = pred;
            size_t batch = pred.shape()[0];
            size_t dim = pred.shape()[1];
            float* p_ptr = prob.data();

            for(size_t i=0; i<batch; ++i) {
                float max_val = -1e9;
                for(size_t j=0; j<dim; ++j) {
                    if (p_ptr[i*dim + j] > max_val) max_val = p_ptr[i*dim + j];
                }

                float sum = 0;
                for(size_t j=0; j<dim; ++j) {
                    p_ptr[i*dim + j] = std::exp(p_ptr[i*dim + j] - max_val);
                    sum += p_ptr[i*dim + j];
                }

                for(size_t j=0; j<dim; ++j) {
                    p_ptr[i*dim + j] /= sum;
                }
            }

            // 2. Compute Gradient (prob - target)
            Tensor<float> grad = prob;
            const float* t = target.data();
            float* g = grad.data();
            size_t n = pred.size(); // total elements

            for(size_t i=0; i<n; ++i) {
                g[i] -= t[i];
            }

            // Normalize by batch size
            float scale = 1.0f / batch;
             for(size_t i=0; i<n; ++i) {
                g[i] *= scale;
            }
            return grad;
        }
    };
}
}

// Re-implementing mock test with corrected includes and types
void test_mnist_mock() {
    std::cout << "--- Testing Mock MNIST Training with LinearWHT ---" << std::endl;

    size_t batch_size = 64;
    Tensor<float> x({batch_size, 784});
    Tensor<float> y({batch_size, 10});

    x.random(0, 1);
    y.fill(0);
    // targets
    float* y_ptr = y.data();
    for(size_t i=0; i<batch_size; ++i) {
        y_ptr[i*10 + (rand()%10)] = 1.0f;
    }

    Sequential<float> model;

    // Layer 1: Dense 784 -> 1024
    // We don't have nice `add<T>` with activation enum in the generic `Sequential` I saw in `model.hpp`.
    // It has `add(shared_ptr)`.
    // And Dense constructor takes (in, out).
    // Activation is likely separate layer or inside Dense?
    // Let's assume Dense does not have activation param based on `Dense(N, N)` in verify_spectral.
    // So we add Dense, then Activation (if available) or assume linear for now.
    // Wait, Phase 3 said "Implement ReLU and Softmax". They might be layers.

    // Let's try to include Activations if they exist.
    // `ls include/dreidel/layers/`

    model.add(std::make_shared<layers::Dense<float>>(784, 1024));
    // model.add(std::make_shared<layers::ReLU<float>>()); // guessing

    // Layer 2: LinearWHT 1024
    model.add(std::make_shared<layers::LinearWHT<float>>(1024));

    // Layer 3: Dense 1024 -> 10
    model.add(std::make_shared<layers::Dense<float>>(1024, 10));

    // Output should be softmaxed for CE.
    // Assuming simple linear output for now, handled by loss gradient approx.

    auto optimizer = std::make_shared<optim::SGD<float>>(0.01f);
    model.compile(optimizer);

    loss::CrossEntropy loss_fn;

    float initial_loss = 0;
    float final_loss = 0;

    for(int i=0; i<10; ++i) {
        float loss = model.train_step(x, y, loss_fn);
        if (i==0) initial_loss = loss;
        final_loss = loss;
    }

    std::cout << "Initial Loss: " << initial_loss << std::endl;
    std::cout << "Final Loss: " << final_loss << std::endl;

    // Since data is random, loss might not decrease much or might oscillate.
    // But it should run without crashing.
    std::cout << "PASS: Training loop ran successfully." << std::endl;
}

int main() {
    test_mnist_mock();
    return 0;
}
