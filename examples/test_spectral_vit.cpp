
#include "../include/dreidel/models/SpectralViT.hpp"
#include <iostream>
#include <chrono>

using namespace dreidel;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <weights_path>" << std::endl;
        return 1;
    }

    std::string weights_path = argv[1];

    try {
        std::cout << "Initializing Spectral ViT..." << std::endl;
        models::SpectralViT<float> model;

        std::cout << "Loading weights from " << weights_path << "..." << std::endl;
        auto start_load = std::chrono::high_resolution_clock::now();
        model.load_weights(weights_path);
        auto end_load = std::chrono::high_resolution_clock::now();
        std::cout << "Loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count() << " ms." << std::endl;

        // Create dummy input
        // Batch=1, Seq=197 (14x14 patches + cls), Dim=1024 (padded)
        // Original ViT dim is 768, but we padded to 1024 in Recasting.
        // So input must be 1024.
        std::vector<size_t> input_shape = {1, 197, 1024};
        Tensor<float> input(input_shape);
        input.random(0, 1); // Random input

        std::cout << "Running Forward Pass..." << std::endl;
        auto start_fwd = std::chrono::high_resolution_clock::now();
        Tensor<float> output = model.forward(input);
        auto end_fwd = std::chrono::high_resolution_clock::now();

        std::cout << "Forward pass completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_fwd - start_fwd).count() << " ms." << std::endl;
        std::cout << "Output shape: ";
        for (auto d : output.shape()) std::cout << d << " ";
        std::cout << std::endl;

        // Print some stats
        float mean = 0;
        float max_val = -1e9;
        float* data = output.data();
        for(size_t i=0; i<output.size(); ++i) {
            mean += data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        mean /= output.size();
        std::cout << "Output Mean: " << mean << ", Max: " << max_val << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
