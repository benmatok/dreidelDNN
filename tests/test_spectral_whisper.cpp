#include "../include/dreidel/models/SpectralWhisper.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace dreidel;

void test_spectral_whisper() {
    std::cout << "Testing SpectralWhisper..." << std::endl;

    // Config for small test model
    models::WhisperConfig config;
    config.n_mels = 80;
    config.n_fft = 400;
    config.hop_length = 160;
    config.n_audio_state = 128; // Needs to be power of 2 for DeepSpectralLinear
    config.n_audio_head = 4;
    config.n_audio_layer = 1;
    config.n_text_state = 128; // Power of 2
    config.n_text_head = 4;
    config.n_text_layer = 1;

    models::SpectralWhisper<float> model(config);

    // Create dummy input audio (1 batch, 16000 samples)
    size_t samples = 16000;
    Tensor<float> audio_input({1, samples});
    audio_input.random(0.0f, 0.1f);

    // Create dummy decoder embeddings (1 batch, 10 tokens, 128 dim)
    Tensor<float> decoder_input({1, 10, 128});
    decoder_input.random(0.0f, 1.0f);

    try {
        Tensor<float> output = model.forward(audio_input, decoder_input);

        std::cout << "Forward Pass Successful." << std::endl;
        std::cout << "Output shape: ";
        for(auto s : output.shape()) std::cout << s << " ";
        std::cout << std::endl;

        if (output.shape()[0] != 1 || output.shape()[1] != 10 || output.shape()[2] != 128) {
            std::cerr << "Output shape mismatch!" << std::endl;
            exit(1);
        }
    } catch (const std::exception& e) {
        std::cerr << "Forward Pass Failed: " << e.what() << std::endl;
        exit(1);
    }
}

int main() {
    test_spectral_whisper();
    return 0;
}
