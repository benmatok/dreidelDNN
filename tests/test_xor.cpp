#include <dreidel/dreidel.hpp>
#include <dreidel/model.hpp>
#include <dreidel/layers/Dense.hpp>
#include <dreidel/layers/ReLU.hpp>
#include <dreidel/layers/Softmax.hpp>
#include <dreidel/optim/SGD.hpp>

#include <iostream>
#include <vector>
#include <cassert>
#include <limits>

using namespace dreidel;

// Cross Entropy Loss
struct CELoss {
    float compute(const Tensor<float>& pred, const Tensor<float>& target) {
        float loss = 0;
        size_t n = pred.size(); // Total elements (Batch * Classes)
        // Usually averaged over batch
        size_t batch_size = pred.shape()[0];

        for (size_t i = 0; i < n; ++i) {
            float y = pred.data()[i];
            float t = target.data()[i];
            // Clamp y to avoid log(0)
            if (y < 1e-7f) y = 1e-7f;
            if (y > 1.0f - 1e-7f) y = 1.0f - 1e-7f;

            loss -= t * std::log(y);
        }
        return loss / batch_size;
    }

    Tensor<float> gradient(const Tensor<float>& pred, const Tensor<float>& target) {
        Tensor<float> grad = pred; // Copy shape
        size_t n = pred.size();
        size_t batch_size = pred.shape()[0];

        for (size_t i = 0; i < n; ++i) {
            float y = pred.data()[i];
            float t = target.data()[i];
            if (y < 1e-7f) y = 1e-7f;
            if (y > 1.0f - 1e-7f) y = 1.0f - 1e-7f;

            // dL/dy = -t/y
            grad.data()[i] = -t / y;

            // Normalize by batch size if loss is averaged
            grad.data()[i] /= batch_size;
        }
        return grad;
    }
};

void test_xor() {
    std::cout << "Training XOR Network..." << std::endl;

    // XOR Data
    Tensor<float> inputs({4, 2}, {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    });

    Tensor<float> targets({4, 2}, {
        1, 0,
        0, 1,
        0, 1,
        1, 0
    });

    // Model
    Sequential<float> model;
    model.add(std::make_shared<layers::Dense<float>>(2, 8)); // 8 Hidden units
    model.add(std::make_shared<layers::ReLU<float>>());
    model.add(std::make_shared<layers::Dense<float>>(8, 2));
    model.add(std::make_shared<layers::Softmax<float>>());

    // Higher learning rate
    auto optimizer = std::make_shared<optim::SGD<float>>(0.1f);
    model.compile(optimizer);

    CELoss loss_fn;

    // Train
    for (int epoch = 0; epoch < 10000; ++epoch) {
        float loss = model.train_step(inputs, targets, loss_fn);
        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }

    // Verify
    Tensor<float> output = model.forward(inputs);
    std::cout << "Predictions:" << std::endl;

    int correct = 0;
    for (int i = 0; i < 4; ++i) {
        float p0 = output.data()[i*2];
        float p1 = output.data()[i*2+1];

        float t0 = targets.data()[i*2];

        int pred_class = (p1 > p0) ? 1 : 0;
        int true_class = (t0 > 0.5) ? 0 : 1;

        std::cout << "Input [" << inputs.data()[i*2] << ", " << inputs.data()[i*2+1] << "] -> "
                  << "Prob [0]: " << p0 << ", [1]: " << p1
                  << " | Pred: " << pred_class << " True: " << true_class << std::endl;

        if (pred_class == true_class) correct++;
    }

    std::cout << "Accuracy: " << correct << "/4" << std::endl;
    if (correct == 4) std::cout << "PASS" << std::endl;
    else std::cout << "FAIL" << std::endl;
}

int main() {
    test_xor();
    return 0;
}
