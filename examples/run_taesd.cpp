#include "dreidel/models/TAESD.hpp"
#include <iostream>
#include <vector>
#include <chrono>

using namespace dreidel::taesd;

int main(int argc, char** argv) {
    std::string filename;
    if (argc > 1) {
        filename = argv[1];
    }

    if (filename == "dummy") {
         std::cout << "Creating dummy model file..." << std::endl;
         // We need to calculate exact size.
         // See Decoder::load_from_file logic.
         // 1 (4->64) + 9 (64->64) + 1 + 9 + 1 + 9 + 1 + 3 + 1 (64->3) = 35 layers
         // Conv 4->64: 4*64*9 w + 64 b = 2304 + 64 = 2368 floats
         // Conv 64->64: 64*64*9 w + 64 b = 36864 + 64 = 36928 floats
         // Conv 64->3: 64*3*9 w + 3 b = 1728 + 3 = 1731 floats

         // Count:
         // 1x Start: 2368
         // 33x Middle (64->64): 33 * 36928 = 1218624
         // 1x End: 1731

         std::ofstream f("dummy_taesd.bin", std::ios::binary);

         // Start
         std::vector<float> buf(2368, 0.01f);
         f.write((char*)buf.data(), buf.size()*4);

         // Middle
         std::vector<float> mid(36928, 0.001f);
         for(int i=0; i<33; ++i) f.write((char*)mid.data(), mid.size()*4);

         // End
         std::vector<float> end(1731, 0.01f);
         f.write((char*)end.data(), end.size()*4);

         f.close();
         std::cout << "Created dummy_taesd.bin" << std::endl;
         filename = "dummy_taesd.bin";
    }

    if (filename.empty()) {
        std::cout << "Usage: " << argv[0] << " <taesd_decoder.bin> or dummy" << std::endl;
        return 1;
    }

    std::cout << "Initializing Decoder..." << std::endl;
    Decoder decoder;

    try {
        decoder.load_from_file(filename.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return 1;
    }

    // Input Latent: 1x4x64x64 (H=64, W=64, C=4)
    Tensor latent(64, 64, 4);

    // Fill with some data
    for(size_t i=0; i<latent.data.size(); ++i) latent.data[i] = (float)i / latent.data.size();

    // Output Image: 512x512x3
    Tensor image(512, 512, 3);

    std::cout << "Running Forward Pass..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    decoder.forward(latent, image);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Forward pass completed in " << diff.count() << " s" << std::endl;
    std::cout << "First pixel: " << image.data[0] << ", " << image.data[1] << ", " << image.data[2] << std::endl;

    return 0;
}
