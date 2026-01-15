#ifndef DREIDEL_MODELS_ZENITHDISCRIMINATOR_HPP
#define DREIDEL_MODELS_ZENITHDISCRIMINATOR_HPP

#include "../core/Model.hpp"
#include "../layers/Conv2D.hpp"
#include "../layers/LeakyReLU.hpp"
#include "../layers/Bias.hpp" // If needed separately, but Conv2D usually has bias
#include <vector>

namespace dreidel {
namespace models {

// Simple PatchGAN Discriminator (70x70 style)
// Input: 512x512x3
// C64 (s2) -> C128 (s2) -> C256 (s2) -> C512 (s1) -> C1 (s1)
// Actually standard PatchGAN:
// 1. Conv 64, k=4, s=2, p=1 + LeakyReLU
// 2. Conv 128, k=4, s=2, p=1 + LeakyReLU
// 3. Conv 256, k=4, s=2, p=1 + LeakyReLU
// 4. Conv 512, k=4, s=1, p=1 + LeakyReLU
// 5. Conv 1, k=4, s=1, p=1 (Logits)
// Note: dreidel Conv2D does generic convolution.
class ZenithDiscriminator : public Model<float> {
public:
    ZenithDiscriminator() {
        // Layer 1: 3 -> 64, stride 2
        conv1 = new layers::Conv2D<float>(3, 64, 4, 2, 1);
        act1 = new layers::LeakyReLU<float>(0.2f);

        // Layer 2: 64 -> 128, stride 2
        conv2 = new layers::Conv2D<float>(64, 128, 4, 2, 1);
        act2 = new layers::LeakyReLU<float>(0.2f);

        // Layer 3: 128 -> 256, stride 2
        conv3 = new layers::Conv2D<float>(128, 256, 4, 2, 1);
        act3 = new layers::LeakyReLU<float>(0.2f);

        // Layer 4: 256 -> 512, stride 1
        conv4 = new layers::Conv2D<float>(256, 512, 4, 1, 1);
        act4 = new layers::LeakyReLU<float>(0.2f);

        // Layer 5: 512 -> 1, stride 1
        conv5 = new layers::Conv2D<float>(512, 1, 4, 1, 1);

        // Register modules
        add_layer(conv1); add_layer(act1);
        add_layer(conv2); add_layer(act2);
        add_layer(conv3); add_layer(act3);
        add_layer(conv4); add_layer(act4);
        add_layer(conv5);
    }

    Tensor<float> forward(const Tensor<float>& input) override {
        auto x = conv1->forward(input);
        x = act1->forward(x);

        x = conv2->forward(x);
        x = act2->forward(x);

        x = conv3->forward(x);
        x = act3->forward(x);

        x = conv4->forward(x);
        x = act4->forward(x);

        x = conv5->forward(x);
        return x;
    }

    Tensor<float> backward(const Tensor<float>& grad_output) override {
        // Reverse order
        auto dx = conv5->backward(grad_output);

        dx = act4->backward(dx);
        dx = conv4->backward(dx);

        dx = act3->backward(dx);
        dx = conv3->backward(dx);

        dx = act2->backward(dx);
        dx = conv2->backward(dx);

        dx = act1->backward(dx);
        dx = conv1->backward(dx);

        return dx;
    }

    ~ZenithDiscriminator() {
        delete conv1; delete act1;
        delete conv2; delete act2;
        delete conv3; delete act3;
        delete conv4; delete act4;
        delete conv5;
    }

private:
    layers::Conv2D<float>* conv1;
    layers::LeakyReLU<float>* act1;

    layers::Conv2D<float>* conv2;
    layers::LeakyReLU<float>* act2;

    layers::Conv2D<float>* conv3;
    layers::LeakyReLU<float>* act3;

    layers::Conv2D<float>* conv4;
    layers::LeakyReLU<float>* act4;

    layers::Conv2D<float>* conv5;
};

} // namespace models
} // namespace dreidel

#endif // DREIDEL_MODELS_ZENITHDISCRIMINATOR_HPP
