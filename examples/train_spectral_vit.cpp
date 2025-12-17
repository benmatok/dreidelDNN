
#include "../include/dreidel/models/SpectralViT.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

using namespace dreidel;

// Helper to read tensor from file
template<typename T>
void read_tensor_util(std::ifstream& f, Tensor<T>& t) {
    uint32_t rank;
    f.read(reinterpret_cast<char*>(&rank), 4);

    std::vector<size_t> shape(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        uint32_t d;
        f.read(reinterpret_cast<char*>(&d), 4);
        shape[i] = d;
    }

    t = Tensor<T>(shape);
    size_t num_elements = t.size();
    f.read(reinterpret_cast<char*>(t.data()), num_elements * sizeof(T));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights_path> <distillation_data_path>" << std::endl;
        return 1;
    }

    std::string weights_path = argv[1];
    std::string data_path = argv[2];

    try {
        std::cout << "--- Spectral ViT Block-wise Distillation ---" << std::endl;

        models::SpectralViT<float> model;
        std::cout << "Loading Student Weights from " << weights_path << "..." << std::endl;
        model.load_weights(weights_path);

        std::cout << "Loading Layer-wise Data from " << data_path << "..." << std::endl;
        std::ifstream f(data_path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open data file");

        uint32_t num_blocks;
        f.read(reinterpret_cast<char*>(&num_blocks), 4);

        // Loop over blocks
        for (int i = 0; i < 12; ++i) {
            Tensor<float> inputs;
            Tensor<float> targets;

            read_tensor_util(f, inputs);
            read_tensor_util(f, targets);

            std::cout << "--- Training Block " << i << " ---" << std::endl;
            std::cout << "Input: "; for(auto d : inputs.shape()) std::cout << d << " ";
            std::cout << " Target: "; for(auto d : targets.shape()) std::cout << d << " ";
            std::cout << std::endl;

            float lr = 1e-5;
            optim::DiagonalNewton<float> optimizer(lr);
            optimizer.add_parameters(model.parameters_block(i), model.gradients_block(i), model.curvatures_block(i));

            for(int epoch=0; epoch<3; ++epoch) {
                 auto start = std::chrono::high_resolution_clock::now();
                 optimizer.zero_grad();

                 // Forward Block
                 // We use teacher input 'inputs' as input (Teacher Forcing)
                 Tensor<float> output = model.forward_block(i, inputs);

                 // Compute Loss & Gradient
                 size_t total = output.size();
                 Tensor<float> grad_output(output.shape());
                 float loss = 0;

                 float* out_ptr = output.data();
                 float* tgt_ptr = targets.data();
                 float* grad_ptr = grad_output.data();

                 #pragma omp parallel for reduction(+:loss)
                 for(size_t k=0; k<total; ++k) {
                     float diff = out_ptr[k] - tgt_ptr[k];
                     loss += diff * diff;
                     grad_ptr[k] = 2.0f * diff / total;
                 }
                 loss /= total;

                 // Backward Block
                 model.backward_block(i, grad_output);

                 optimizer.step();

                 auto end = std::chrono::high_resolution_clock::now();
                 double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                 std::cout << "  Epoch " << epoch+1 << ": Loss " << loss << " (" << ms << " ms)" << std::endl;
            }
        }

        // Pooler
        {
            Tensor<float> inputs;
            Tensor<float> targets;
            read_tensor_util(f, inputs);
            read_tensor_util(f, targets);

            std::cout << "--- Training Pooler ---" << std::endl;

            float lr = 1e-5;
            optim::DiagonalNewton<float> optimizer(lr);
            optimizer.add_parameters(model.parameters_pooler(), model.gradients_pooler(), model.curvatures_pooler());

            for(int epoch=0; epoch<3; ++epoch) {
                 optimizer.zero_grad();
                 Tensor<float> output = model.forward_pooler(inputs);

                 size_t total = output.size();
                 Tensor<float> grad_output(output.shape());
                 float loss = 0;

                 float* out_ptr = output.data();
                 float* tgt_ptr = targets.data();
                 float* grad_ptr = grad_output.data();

                 #pragma omp parallel for reduction(+:loss)
                 for(size_t k=0; k<total; ++k) {
                     float diff = out_ptr[k] - tgt_ptr[k];
                     loss += diff * diff;
                     grad_ptr[k] = 2.0f * diff / total;
                 }
                 loss /= total;

                 model.backward_pooler(grad_output);
                 optimizer.step();
                 std::cout << "  Epoch " << epoch+1 << ": Loss " << loss << std::endl;
            }
        }

        std::cout << "Distillation Complete." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
