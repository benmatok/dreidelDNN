#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include <cstdlib>

#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/models/ZenithNano.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"

using namespace dreidel;

// Reusing wavelet generation from train_zenith_nano.cpp
void generate_wavelet_batch(Tensor<float>& data) {
    auto shape = data.shape();
    size_t batch = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    size_t C = shape[3];

    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> center_dist(0.3f, 0.7f);
    std::uniform_real_distribution<float> scale_dist(0.05f, 0.2f);

    data.fill(0.0f);
    float* ptr = data.data();

    for(size_t n=0; n<batch; ++n) {
        for(int k=0; k<4; ++k) {
            float cx = center_dist(gen) * W;
            float cy = center_dist(gen) * H;
            float s = scale_dist(gen) * W;

            float r = (float)gen() / gen.max();
            float g = (float)gen() / gen.max();
            float b = (float)gen() / gen.max();

            #pragma omp parallel for collapse(2)
            for(size_t y=0; y<H; ++y) {
                for(size_t x=0; x<W; ++x) {
                    float dx = x - cx;
                    float dy = y - cy;
                    float val = std::exp(-(dx*dx + dy*dy) / (2*s*s));
                    if (val > 0.01f) {
                        size_t idx = ((n*H + y)*W + x)*C;
                        ptr[idx + 0] += val * r;
                        ptr[idx + 1] += val * g;
                        ptr[idx + 2] += val * b;
                    }
                }
            }
        }
    }
}

// Function to load real data batch from file
bool load_real_batch(Tensor<float>& data, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open batch file " << filepath << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    if (!file) {
        std::cerr << "Error: Could not read full batch from " << filepath << std::endl;
        return false;
    }
    return true;
}

int main() {
    std::cout << "=== Training ZenithNano: Wavelet Pretraining -> Real Data Finetuning ===" << std::endl;

    // Config
    size_t batch_size = 4;
    size_t H = 512, W = 512;
    float lr = 1e-4f;

    // Phase 1: Wavelet Pretraining
    size_t pretrain_epochs = 10000;

    // Phase 2: Real Data
    size_t real_epochs = 50; // "Shortly for validation"

    // Model
    models::ZenithNano model;
    optim::SimpleAdam<float> optimizer(lr);
    optimizer.add_parameters(model.parameters(), model.gradients());

    Tensor<float> input({batch_size, H, W, 3});

    // ---------------------------------------------------------
    // PHASE 1: Wavelet Pretraining
    // ---------------------------------------------------------
    std::cout << "\n--- Phase 1: Wavelet Pretraining (" << pretrain_epochs << " epochs) ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for(size_t epoch = 0; epoch < pretrain_epochs; ++epoch) {
        generate_wavelet_batch(input);

        Tensor<float> output = model.forward(input);

        // Loss (MAE)
        float loss = 0;
        size_t size = output.size();
        const float* out_ptr = output.data();
        const float* tgt_ptr = input.data();
        Tensor<float> grad_output(output.shape());
        float* go_ptr = grad_output.data();

        #pragma omp parallel for reduction(+:loss)
        for(size_t i=0; i<size; ++i) {
            float diff = out_ptr[i] - tgt_ptr[i];
            loss += std::abs(diff);
            float sign = (diff > 0) ? 1.0f : ((diff < 0) ? -1.0f : 0.0f);
            go_ptr[i] = sign / size;
        }
        loss /= size;

        optimizer.zero_grad();
        model.backward(grad_output);
        optimizer.step();

        if (epoch % 1000 == 0 || epoch == pretrain_epochs - 1) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            std::cout << "Pretrain Epoch " << epoch << " | Time: " << std::fixed << std::setprecision(1) << elapsed << "s | Loss: " << std::setprecision(6) << loss << std::endl;
        }
    }

    // ---------------------------------------------------------
    // PHASE 2: Real Data Training
    // ---------------------------------------------------------
    std::cout << "\n--- Phase 2: Real Data Finetuning (" << real_epochs << " batches) ---" << std::endl;
    std::cout << "Downloading validation set..." << std::endl;

    // Download validation set once
    std::string val_cmd = "python3 tools/data_loader.py --batch-size " + std::to_string(batch_size) + " --output val_batch.bin --validation";
    int ret = std::system(val_cmd.c_str());
    if (ret != 0) {
        std::cerr << "Failed to download validation set." << std::endl;
        return 1;
    }

    // Training loop
    for(size_t epoch = 0; epoch < real_epochs; ++epoch) {
        // Fetch new batch
        std::string train_cmd = "python3 tools/data_loader.py --batch-size " + std::to_string(batch_size) + " --output train_batch.bin";
        // To be less verbose, redirect stdout/stderr or just let it print
        // user wants to see "download small set for validation and train on new batches each time"
        // Let's print fetching status
        std::cout << "Fetching real batch " << epoch + 1 << "/" << real_epochs << "..." << std::endl;
        ret = std::system(train_cmd.c_str());
        if (ret != 0) {
             std::cerr << "Failed to download training batch. Skipping." << std::endl;
             continue;
        }

        if (!load_real_batch(input, "train_batch.bin")) {
            continue;
        }

        Tensor<float> output = model.forward(input);

        // Loss (MAE)
        float loss = 0;
        size_t size = output.size();
        const float* out_ptr = output.data();
        const float* tgt_ptr = input.data();
        Tensor<float> grad_output(output.shape());
        float* go_ptr = grad_output.data();

        #pragma omp parallel for reduction(+:loss)
        for(size_t i=0; i<size; ++i) {
            float diff = out_ptr[i] - tgt_ptr[i];
            loss += std::abs(diff);
            float sign = (diff > 0) ? 1.0f : ((diff < 0) ? -1.0f : 0.0f);
            go_ptr[i] = sign / size;
        }
        loss /= size;

        optimizer.zero_grad();
        model.backward(grad_output);
        optimizer.step();

        std::cout << "Real Data Epoch " << epoch << " | Loss: " << loss << std::endl;

        // Validation every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            std::cout << "Validating..." << std::endl;
            if (load_real_batch(input, "val_batch.bin")) {
                 Tensor<float> val_out = model.forward(input);
                 float val_loss = 0;
                 const float* v_out = val_out.data();
                 const float* v_tgt = input.data();
                 #pragma omp parallel for reduction(+:val_loss)
                 for(size_t i=0; i<size; ++i) {
                     val_loss += std::abs(v_out[i] - v_tgt[i]);
                 }
                 val_loss /= size;
                 std::cout << ">>> Validation Loss: " << val_loss << std::endl;
            }
        }
    }

    std::cout << "Training Complete." << std::endl;
    return 0;
}
