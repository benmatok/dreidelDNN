
#include "../include/dreidel/models/SpectralWhisper.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <string>
#include <fstream>
#include <cmath>

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
    std::string weights_path = "whisper_spectral_weights.bin";
    std::string data_path = "whisper_layer_data.bin";
    int epochs = 1000;
    float lr = 0.1; // Reduced from 1.0 to prevent divergence

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--data" && i + 1 < argc) data_path = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
    }

    try {
        std::cout << "--- Spectral Whisper Distillation ---" << std::endl;
        std::cout << "Weights: " << weights_path << std::endl;
        std::cout << "Data: " << data_path << std::endl;
        std::cout << "Epochs: " << epochs << ", LR: " << lr << std::endl;

        // Initialize Model
        models::WhisperConfig config;
        config.n_audio_state = 384;
        config.n_audio_head = 6;
        config.n_audio_layer = 4;
        config.n_text_state = 384;
        config.n_text_head = 6;
        config.n_text_layer = 4;

        models::SpectralWhisper<float> model(config);

        std::cout << "Loading Weights..." << std::endl;
        model.load_weights(weights_path);

        std::cout << "Loading Distillation Data..." << std::endl;
        std::ifstream f(data_path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open data file");

        uint32_t enc_layers_in_data, dec_layers_in_data;
        f.read(reinterpret_cast<char*>(&enc_layers_in_data), 4);
        f.read(reinterpret_cast<char*>(&dec_layers_in_data), 4);

        // Skip global inputs
        Tensor<float> mel_input;
        read_tensor_util(f, mel_input);
        Tensor<float> dec_input;
        read_tensor_util(f, dec_input);

        // Track convergence
        float total_initial_mse = 0;
        float total_final_mse = 0;

        // ENCODER DISTILLATION
        std::cout << "\n--- Distilling Encoders (" << enc_layers_in_data << " blocks) ---" << std::endl;
        for (int i = 0; i < enc_layers_in_data; ++i) {
            Tensor<float> inputs;
            Tensor<float> targets;
            read_tensor_util(f, inputs);
            read_tensor_util(f, targets);

            if (inputs.size() == 0) continue;

            std::cout << "Block " << i << " | Input: " << inputs.shape()[0] << "x" << inputs.shape()[1] << "x" << inputs.shape()[2] << std::endl;

            optim::DiagonalNewton<float> optimizer(lr);
            optimizer.add_parameters(model.parameters_encoder_block(i), model.gradients_encoder_block(i), model.curvatures_encoder_block(i));

            float initial_loss = 0;
            float final_loss = 0;

            for(int epoch=0; epoch<epochs; ++epoch) {
                 optimizer.zero_grad();
                 Tensor<float> output = model.forward_encoder_block(i, inputs);

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

                 if (epoch == 0) initial_loss = loss;
                 final_loss = loss;

                 model.backward_encoder_block(i, grad_output);
                 optimizer.step();
            }
            std::cout << "  Block " << i << " MSE: " << initial_loss << " -> " << final_loss << std::endl;
            total_initial_mse += initial_loss;
            total_final_mse += final_loss;
        }

        // DECODER DISTILLATION
        std::cout << "\n--- Distilling Decoders (" << dec_layers_in_data << " blocks) ---" << std::endl;
        for (int i = 0; i < dec_layers_in_data; ++i) {
            Tensor<float> inputs;
            Tensor<float> enc_out;
            Tensor<float> targets;

            read_tensor_util(f, inputs);
            read_tensor_util(f, enc_out);
            read_tensor_util(f, targets);

            if (inputs.size() == 0) continue;

            std::cout << "Block " << i << " | Input: " << inputs.shape()[0] << "x" << inputs.shape()[1] << "x" << inputs.shape()[2] << std::endl;

            optim::DiagonalNewton<float> optimizer(lr);
            optimizer.add_parameters(model.parameters_decoder_block(i), model.gradients_decoder_block(i), model.curvatures_decoder_block(i));

            float initial_loss = 0;
            float final_loss = 0;

            for(int epoch=0; epoch<epochs; ++epoch) {
                 optimizer.zero_grad();
                 Tensor<float> output = model.forward_decoder_block(i, inputs, enc_out);

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

                 if (epoch == 0) initial_loss = loss;
                 final_loss = loss;

                 model.backward_decoder_block(i, grad_output);
                 optimizer.step();
            }
            std::cout << "  Block " << i << " MSE: " << initial_loss << " -> " << final_loss << std::endl;
            total_initial_mse += initial_loss;
            total_final_mse += final_loss;
        }

        std::cout << "\nTotal Initial MSE: " << total_initial_mse << std::endl;
        std::cout << "Total Final MSE: " << total_final_mse << std::endl;
        std::cout << "Distillation Complete." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
