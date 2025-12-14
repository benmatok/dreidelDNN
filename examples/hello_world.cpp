#include <dreidel/dreidel.hpp>
#include <iostream>
#include <vector>

using namespace dreidel;

// Mock classes for verification
template <typename T>
class MockLayer : public layers::Layer<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) override {
        std::cout << "MockLayer Forward Pass" << std::endl;
        return input;
    }

    Tensor<T> backward(const Tensor<T>& grad_output) override {
        std::cout << "MockLayer Backward Pass" << std::endl;
        return grad_output;
    }

    std::string name() const override { return "MockLayer"; }
};

int main() {
    std::cout << "--- dreidelDNN Hello World ---" << std::endl;

    // 1. Instantiate Tensor
    std::vector<size_t> shape = {2, 2};
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
    Tensor<float> t(shape, data);

    std::cout << "Tensor created with shape: [" << t.shape()[0] << ", " << t.shape()[1] << "]" << std::endl;
    std::cout << "Data[0]: " << t[0] << std::endl;

    // 2. Instantiate Mock Layer
    MockLayer<float> layer;
    std::cout << "Layer: " << layer.name() << std::endl;

    // 3. Forward Pass
    Tensor<float> out = layer.forward(t);

    // 4. IO Mock
    io::SimpleSerializer serializer;
    serializer.save("test_model.weights", t);

    std::cout << "--- Verification Complete ---" << std::endl;
    return 0;
}
