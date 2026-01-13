#include "dreidel/models/TAESD.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

// Minimal STB Image Write for PNG
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace dreidel::taesd;

// Simple Wavelet Generation Logic (adapted to avoid linking issues)
void generate_wavelets(Tensor& img) {
    int H = img.h;
    int W = img.w;
    int C = img.c;

    // Clear
    std::fill(img.data.begin(), img.data.end(), 0.0f);

    // Random
    srand(42); // Fixed seed

    int num_wavelets = 20;

    for(int i=0; i<num_wavelets; ++i) {
        float cx = (float)(rand() % W);
        float cy = (float)(rand() % H);
        float s = 10.0f + (rand() % 40); // Size
        float amp = 0.5f + ((float)rand()/RAND_MAX) * 0.5f;

        // Add Gabor-like blob
        int radius = (int)(s * 3);
        for(int dy = -radius; dy <= radius; ++dy) {
            for(int dx = -radius; dx <= radius; ++dx) {
                int px = (int)cx + dx;
                int py = (int)cy + dy;

                if (px >= 0 && px < W && py >= 0 && py < H) {
                    float r2 = (float)(dx*dx + dy*dy);
                    float env = std::exp(-r2 / (2 * s * s));
                    float wave = std::cos(dx * 0.3f);
                    float val = amp * env * wave;

                    // Add to channels
                    for(int c=0; c<C; ++c) {
                        img.data[(py*W + px)*C + c] += val;
                    }
                }
            }
        }
    }

    // Normalize to 0-1
    float min_v = 1e9, max_v = -1e9;
    for(float v : img.data) {
        if(v < min_v) min_v = v;
        if(v > max_v) max_v = v;
    }
    float range = max_v - min_v;
    if(range < 1e-6) range = 1.0f;

    for(float& v : img.data) {
        v = (v - min_v) / range;
    }
}

void save_png(const char* filename, const Tensor& img) {
    int H = img.h;
    int W = img.w;
    int C = img.c;
    std::vector<unsigned char> bytes(H*W*C);

    for(int i=0; i<H*W*C; ++i) {
        float v = img.data[i];
        v = std::max(0.0f, std::min(1.0f, v));
        bytes[i] = (unsigned char)(v * 255.0f);
    }

    stbi_write_png(filename, W, H, C, bytes.data(), W*C);
    std::cout << "Saved " << filename << std::endl;
}

int main(int argc, char** argv) {
    std::string encoder_path = "taesd_encoder.bin";
    std::string decoder_path = "taesd_decoder.bin";

    if (argc > 1) encoder_path = argv[1];
    if (argc > 2) decoder_path = argv[2];

    Encoder encoder;
    Decoder decoder;

    try {
        std::cout << "Loading models..." << std::endl;
        encoder.load_from_file(encoder_path.c_str());
        decoder.load_from_file(decoder_path.c_str());
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    Tensor input(512, 512, 3);
    Tensor latent(64, 64, 4);
    Tensor output(512, 512, 3);

    std::cout << "Generating input..." << std::endl;
    generate_wavelets(input);
    save_png("taesd_input.png", input);

    std::cout << "Encoding..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    encoder.forward(input, latent);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Encoded in " << std::chrono::duration<double>(t2-t1).count() << "s" << std::endl;

    std::cout << "Decoding..." << std::endl;
    auto t3 = std::chrono::high_resolution_clock::now();
    decoder.forward(latent, output);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Decoded in " << std::chrono::duration<double>(t4-t3).count() << "s" << std::endl;

    save_png("taesd_output.png", output);

    return 0;
}
