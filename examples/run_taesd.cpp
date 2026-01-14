#include "dreidel/models/TAESD.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace dreidel::taesd;

// Debug helper
void print_stats(const std::string& name, const Tensor& t) {
    float min_v = 1e9, max_v = -1e9, sum = 0.0, sum_abs = 0.0;
    int zeros = 0;
    int count = t.h * t.w * t.c;

    for(float v : t.data) {
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        sum += v;
        sum_abs += std::abs(v);
        if (std::abs(v) < 1e-9) zeros++;
    }

    std::cout << "Stats [" << name << "]: "
              << "Shape=(" << t.h << "x" << t.w << "x" << t.c << ") "
              << "Min=" << min_v << " Max=" << max_v
              << " Mean=" << sum/count << " MeanAbs=" << sum_abs/count
              << " Zeros=" << zeros << "/" << count << " (" << (float)zeros/count*100.0 << "%)"
              << std::endl;
}

// Simple Wavelet Generation Logic
void generate_wavelets(Tensor& img) {
    int H = img.h;
    int W = img.w;
    int C = img.c;

    std::fill(img.data.begin(), img.data.end(), 0.0f);
    srand(42);

    int num_wavelets = 20;

    for(int i=0; i<num_wavelets; ++i) {
        float cx = (float)(rand() % W);
        float cy = (float)(rand() % H);
        float s = 10.0f + (rand() % 40);
        float amp = 0.5f + ((float)rand()/RAND_MAX) * 0.5f;

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

                    for(int c=0; c<C; ++c) {
                        img.data[(py*W + px)*C + c] += val;
                    }
                }
            }
        }
    }

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

// Helper to save latent (4 channels) as grid
void save_latent_png(const char* filename, const Tensor& latent) {
    int H = latent.h;
    int W = latent.w;
    int C = latent.c; // 4

    // Create 2x2 grid
    int GH = H * 2;
    int GW = W * 2;
    std::vector<unsigned char> bytes(GH*GW); // Grayscale

    for(int y=0; y<H; ++y) {
        for(int x=0; x<W; ++x) {
            for(int c=0; c<4; ++c) {
                // Map c to quadrant
                int qx = (c % 2) * W;
                int qy = (c / 2) * H;

                float v = latent.data[(y*W + x)*C + c];
                // Latents are roughly -3 to 3. Map to 0-1
                v = (v + 3.0f) / 6.0f;
                v = std::max(0.0f, std::min(1.0f, v));

                bytes[(qy + y)*GW + (qx + x)] = (unsigned char)(v * 255.0f);
            }
        }
    }
    stbi_write_png(filename, GW, GH, 1, bytes.data(), GW);
    std::cout << "Saved latent visualization " << filename << std::endl;
}

int main(int argc, char** argv) {
    std::string encoder_path = "taesd_encoder.bin";
    std::string decoder_path = "taesd_decoder.bin";
    std::string input_image_path = "";

    if (argc > 1) encoder_path = argv[1];
    if (argc > 2) decoder_path = argv[2];
    if (argc > 3) input_image_path = argv[3];

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

    if (!input_image_path.empty()) {
        std::cout << "Loading input image: " << input_image_path << std::endl;
        int w, h, c;
        unsigned char* data = stbi_load(input_image_path.c_str(), &w, &h, &c, 3);
        if (!data) {
            std::cerr << "Failed to load image" << std::endl;
            return 1;
        }

        // Resize/Crop logic? Or just accept?
        // TAESD is fully convolutional, so any size divisible by 8 works.
        // We will just pad to multiple of 8 if needed or truncate.
        int w_aligned = w - (w % 8);
        int h_aligned = h - (h % 8);

        input.resize(h_aligned, w_aligned, 3);
        for(int i=0; i<h_aligned*w_aligned; ++i) {
            input.data[i*3+0] = data[i*3+0] / 255.0f;
            input.data[i*3+1] = data[i*3+1] / 255.0f;
            input.data[i*3+2] = data[i*3+2] / 255.0f;
        }
        stbi_image_free(data);
    } else {
        std::cout << "Generating synthetic input..." << std::endl;
        generate_wavelets(input);
        save_png("taesd_input.png", input);
    }

    print_stats("Input", input);

    Tensor latent(input.h/8, input.w/8, 4);
    Tensor output(input.h, input.w, 3);

    std::cout << "Encoding..." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    encoder.forward(input, latent);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Encoded in " << std::chrono::duration<double>(t2-t1).count() << "s" << std::endl;

    print_stats("Latent", latent);
    save_latent_png("taesd_latent.png", latent);

    std::cout << "Decoding..." << std::endl;
    auto t3 = std::chrono::high_resolution_clock::now();
    decoder.forward(latent, output);
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "Decoded in " << std::chrono::duration<double>(t4-t3).count() << "s" << std::endl;

    print_stats("Output", output);
    save_png("taesd_output.png", output);

    return 0;
}
