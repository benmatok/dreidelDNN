#include <dreidel/dreidel.hpp>
#include <dreidel/models/SpectralWhisper.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>

using namespace dreidel;

// Helper to read binary data
void read_tensor(std::ifstream& f, Tensor<float>& t) {
    uint32_t dims;
    f.read(reinterpret_cast<char*>(&dims), sizeof(uint32_t));
    std::vector<size_t> shape;
    for(uint32_t i=0; i<dims; ++i) {
        uint32_t d;
        f.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        shape.push_back(d);
    }
    t = Tensor<float>(shape);
    size_t size = t.size();
    f.read(reinterpret_cast<char*>(t.data()), size * sizeof(float));
}

std::string read_string(std::ifstream& f) {
    uint32_t len;
    f.read(reinterpret_cast<char*>(&len), sizeof(uint32_t));
    std::string s(len, ' ');
    f.read(&s[0], len);
    return s;
}

void read_perm(std::ifstream& f, std::vector<uint64_t>& p) {
    uint32_t size;
    f.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
    p.resize(size);
    f.read(reinterpret_cast<char*>(p.data()), size * sizeof(uint64_t));
}


int main(int argc, char** argv) {
    std::string weights_path = "whisper_spectral_weights.bin";
    std::string data_path = "whisper_layer_data.bin";

    if (argc > 1) weights_path = argv[1];
    if (argc > 2) data_path = argv[2];

    std::cout << "Loading weights from " << weights_path << "..." << std::endl;
    std::ifstream fw(weights_path, std::ios::binary);
    if (!fw.is_open()) {
        std::cerr << "Failed to open " << weights_path << std::endl;
        return 1;
    }

    char magic[5];
    fw.read(magic, 4);
    magic[4] = '\0';
    if (std::string(magic) != "DRDL") {
        std::cerr << "Invalid magic: " << magic << std::endl;
        return 1;
    }

    uint32_t version;
    fw.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    uint32_t num_layers;
    fw.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));
    std::cout << "Found " << num_layers << " layers in weights file." << std::endl;

    models::WhisperConfig config;
    config.n_audio_state = 384;
    config.n_audio_head = 6;
    config.n_audio_layer = 4;
    config.n_text_state = 384;
    config.n_text_head = 6;
    config.n_text_layer = 4;

    models::SpectralWhisper<float> model(config);

    struct WeightData {
        std::string name;
        std::string type;
        uint32_t dim;
        uint32_t depth;
        std::vector<Tensor<float>> scales;
        std::vector<std::vector<uint64_t>> perms;
        Tensor<float> bias;
        bool has_bias;
    };

    std::vector<WeightData> loaded_weights;

    for(uint32_t i=0; i<num_layers; ++i) {
        WeightData wd;
        wd.name = read_string(fw);
        wd.type = read_string(fw);
        fw.read(reinterpret_cast<char*>(&wd.dim), sizeof(uint32_t));

        if (wd.type == "DeepSpectralLinear") {
            fw.read(reinterpret_cast<char*>(&wd.depth), sizeof(uint32_t));
            for(uint32_t k=0; k<wd.depth; ++k) {
                Tensor<float> scale;
                read_tensor(fw, scale);
                wd.scales.push_back(scale);

                std::vector<uint64_t> p;
                read_perm(fw, p);
                wd.perms.push_back(p);
            }
        }

        bool has_bias;
        fw.read(reinterpret_cast<char*>(&has_bias), sizeof(bool));
        wd.has_bias = has_bias;
        if (has_bias) {
            read_tensor(fw, wd.bias);
        }

        loaded_weights.push_back(wd);
    }
    fw.close();

    std::cout << "Loading distillation data from " << data_path << "..." << std::endl;
    std::ifstream fd(data_path, std::ios::binary);
    if (!fd.is_open()) {
        std::cerr << "Failed to open " << data_path << std::endl;
        return 1;
    }

    uint32_t enc_layers, dec_layers;
    fd.read(reinterpret_cast<char*>(&enc_layers), sizeof(uint32_t));
    fd.read(reinterpret_cast<char*>(&dec_layers), sizeof(uint32_t));

    Tensor<float> mel_input;
    read_tensor(fd, mel_input);

    Tensor<float> decoder_embeds;
    read_tensor(fd, decoder_embeds);

    std::cout << "Mel Input Shape: " << mel_input.shape()[0] << "x" << mel_input.shape()[1] << "x" << mel_input.shape()[2] << std::endl;

    std::cout << "Running Forward Pass..." << std::endl;

    try {
        Tensor<float> out = model.forward(mel_input, decoder_embeds);
        std::cout << "Forward pass successful." << std::endl;
        std::cout << "Output shape: " << out.shape()[0] << "x" << out.shape()[1] << "x" << out.shape()[2] << std::endl;

        if (out.shape()[0] != mel_input.shape()[0]) {
            std::cerr << "Batch dimension mismatch!" << std::endl;
            return 1;
        }
        if (out.shape()[2] != config.n_text_state) {
             std::cerr << "Output embedding dimension mismatch! Expected " << config.n_text_state << " got " << out.shape()[2] << std::endl;
             return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Forward pass failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] SpectralWhisper initialized and ran forward pass." << std::endl;
    return 0;
}
