#include "dreidel/models/TAESD.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>

using namespace dreidel::taesd;

int main(int argc, char** argv) {
    std::string encoder_path = "taesd_encoder.bin";
    std::string decoder_path = "taesd_decoder.bin";

    if (argc > 1) encoder_path = argv[1];
    if (argc > 2) decoder_path = argv[2];

    std::cout << "Loading Encoder from " << encoder_path << "..." << std::endl;
    Encoder encoder;
    try {
        encoder.load_from_file(encoder_path.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load encoder: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Loading Decoder from " << decoder_path << "..." << std::endl;
    Decoder decoder;
    try {
        decoder.load_from_file(decoder_path.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load decoder: " << e.what() << std::endl;
        return 1;
    }

    // Test Image: 512x512x3
    Tensor image(512, 512, 3);
    // Fill with gradient
    for(int y=0; y<512; ++y) {
        for(int x=0; x<512; ++x) {
            float* p = &image.data[(y*512+x)*3];
            p[0] = (float)x / 512.0f;
            p[1] = (float)y / 512.0f;
            p[2] = 0.5f;
        }
    }

    Tensor latent(64, 64, 4);
    Tensor recon(512, 512, 3);

    std::cout << "Running Encoding..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    encoder.forward(image, latent);

    auto mid = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> enc_time = mid - start;
    std::cout << "Encoding time: " << enc_time.count() << " s" << std::endl;

    std::cout << "Running Decoding..." << std::endl;
    decoder.forward(latent, recon);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dec_time = end - mid;
    std::cout << "Decoding time: " << dec_time.count() << " s" << std::endl;

    std::cout << "Total time: " << (end - start).count() << " s" << std::endl;

    // Check reconstruction (basic sanity check)
    // It won't be perfect, but shouldn't be garbage (NaN or Inf)
    float sample = recon.data[0];
    std::cout << "Reconstruction sample (0,0): R=" << sample << std::endl;

    if (std::isnan(sample) || std::isinf(sample)) {
        std::cerr << "Error: Output contains NaN or Inf" << std::endl;
        return 1;
    }

    return 0;
}
