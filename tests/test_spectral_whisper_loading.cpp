
#include "../include/dreidel/models/SpectralWhisper.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <string>

using namespace dreidel;

int main() {
    try {
        std::cout << "Testing SpectralWhisper loading..." << std::endl;

        models::SpectralWhisper<float> model;

        std::string weights_path = "whisper_spectral_weights.bin";
        std::cout << "Loading weights from " << weights_path << "..." << std::endl;
        model.load_weights(weights_path);

        std::cout << "Successfully loaded weights." << std::endl;

        // Test basic forward pass structure (dummy inputs)
        // Whisper Large dim is 1280.
        size_t dim = 1280;
        size_t seq = 10;
        size_t batch = 1;

        Tensor<float> input({batch, seq, dim});
        input.fill(0.1f);

        std::cout << "Running Encoder forward block 0..." << std::endl;
        Tensor<float> enc_out = model.forward_encoder_block(0, input);

        std::cout << "Encoder Output shape: " << enc_out.shape()[0] << " " << enc_out.shape()[1] << " " << enc_out.shape()[2] << std::endl;

        std::cout << "Running Decoder forward block 0..." << std::endl;
        Tensor<float> dec_out = model.forward_decoder_block(0, input, enc_out);

        std::cout << "Decoder Output shape: " << dec_out.shape()[0] << " " << dec_out.shape()[1] << " " << dec_out.shape()[2] << std::endl;

        std::cout << "Test passed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
