
#include "../include/dreidel/models/SpectralViT.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <string>

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
    std::string weights_path;
    std::string data_path;
    int target_block = -1;
    int epochs = 20; // Increased default epochs
    float lr = 1.0; // Increased default LR for DiagonalNewton

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--data" && i + 1 < argc) data_path = argv[++i];
        else if (arg == "--block" && i + 1 < argc) target_block = std::stoi(argv[++i]);
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (weights_path.empty()) weights_path = arg; // Backwards compat
        else if (data_path.empty()) data_path = arg;     // Backwards compat
    }

    if (weights_path.empty() || data_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --weights <path> --data <path> [--block <id>] [--epochs <n>] [--lr <val>]" << std::endl;
        return 1;
    }

    try {
        std::cout << "--- Spectral ViT Block-wise Distillation ---" << std::endl;
        std::cout << "Weights: " << weights_path << std::endl;
        std::cout << "Data: " << data_path << std::endl;
        std::cout << "Block: " << (target_block == -1 ? "All" : std::to_string(target_block)) << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "LR: " << lr << std::endl;

        models::SpectralViT<float> model;
        std::cout << "Loading Student Weights..." << std::endl;
        model.load_weights(weights_path);

        std::cout << "Loading Layer-wise Data..." << std::endl;
        std::ifstream f(data_path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open data file");

        uint32_t num_blocks_in_data;
        f.read(reinterpret_cast<char*>(&num_blocks_in_data), 4);

        int num_encoder_blocks = num_blocks_in_data - 1;

        if (num_encoder_blocks != model.get_num_blocks()) {
             std::cerr << "Warning: Model has " << model.get_num_blocks() << " blocks but data has " << num_encoder_blocks << ". Using minimum." << std::endl;
        }

        int loop_count = std::min(num_encoder_blocks, model.get_num_blocks());

        // Loop over encoder blocks
        for (int i = 0; i < loop_count; ++i) {
            Tensor<float> inputs;
            Tensor<float> targets;

            read_tensor_util(f, inputs);
            read_tensor_util(f, targets);

            if (target_block != -1 && i != target_block) {
                continue;
            }

            std::cout << "--- Training Block " << i << " ---" << std::endl;
            std::cout << "Input: "; for(auto d : inputs.shape()) std::cout << d << " ";
            std::cout << " Target: "; for(auto d : targets.shape()) std::cout << d << " ";
            std::cout << std::endl;

            optim::DiagonalNewton<float> optimizer(lr);
            optimizer.add_parameters(model.parameters_block(i), model.gradients_block(i), model.curvatures_block(i));

            for(int epoch=0; epoch<epochs; ++epoch) {
                 auto start = std::chrono::high_resolution_clock::now();
                 optimizer.zero_grad();

                 // Forward Block
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

                 if (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 10 == 0) {
                     std::cout << "  Epoch " << epoch+1 << ": Loss " << loss << " (" << ms << " ms)" << std::endl;
                 }
            }
        }

        // Pooler
        {
            Tensor<float> inputs;
            Tensor<float> targets;
            read_tensor_util(f, inputs);
            read_tensor_util(f, targets);

            // Pooler ID is loop_count
            if (target_block == -1 || target_block == loop_count) {
                std::cout << "--- Training Pooler ---" << std::endl;

                optim::DiagonalNewton<float> optimizer(lr);
                optimizer.add_parameters(model.parameters_pooler(), model.gradients_pooler(), model.curvatures_pooler());

                for(int epoch=0; epoch<epochs; ++epoch) {
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

                     if (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 10 == 0) {
                         std::cout << "  Epoch " << epoch+1 << ": Loss " << loss << std::endl;
                     }
                }
            }
        }

        std::cout << "Distillation Complete." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
