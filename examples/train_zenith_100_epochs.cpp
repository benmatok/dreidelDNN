#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/dreidel/models/ComparativeAE.hpp"
#include "../include/dreidel/core/Tensor.hpp"
#include "../include/dreidel/utils/WaveletGen2D.hpp"
#include "../include/dreidel/optim/SimpleAdam.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace dreidel;
using namespace dreidel::models;
using namespace dreidel::utils;
using namespace dreidel::optim;

// MSE Loss
template <typename T>
T mse_loss(const Tensor<T>& pred, const Tensor<T>& target, Tensor<T>& grad_input) {
    size_t size = pred.size();
    const T* p_ptr = pred.data();
    const T* t_ptr = target.data();
    T* g_ptr = grad_input.data();

    T sum_sq = 0;
    T scale = 2.0 / size;

    #pragma omp parallel for reduction(+:sum_sq)
    for (size_t i = 0; i < size; ++i) {
        T diff = p_ptr[i] - t_ptr[i];
        sum_sq += diff * diff;
        g_ptr[i] = diff * scale;
    }
    return sum_sq / size;
}

// SVG Generator
void generate_loss_svg(const std::vector<float>& losses, size_t epochs) {
    std::ofstream svg("zenith_100_loss.svg");
    if (!svg.is_open()) return;

    double width = 800;
    double height = 600;
    double padding = 60;

    float min_loss = 1e9, max_loss = -1e9;
    for (float l : losses) {
        if (l < min_loss) min_loss = l;
        if (l > max_loss) max_loss = l;
    }

    // Log scale if range is large
    bool use_log = (max_loss / min_loss > 100);
    double y_min = use_log ? std::log10(min_loss) : min_loss;
    double y_max = use_log ? std::log10(max_loss) : max_loss;
    double y_range = y_max - y_min;
    if (y_range == 0) y_range = 1;

    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\" />\n";

    // Axes
    svg << "<line x1=\"" << padding << "\" y1=\"" << height - padding << "\" x2=\"" << width - padding << "\" y2=\"" << height - padding << "\" stroke=\"black\" />\n";
    svg << "<line x1=\"" << padding << "\" y1=\"" << height - padding << "\" x2=\"" << padding << "\" y2=\"" << padding << "\" stroke=\"black\" />\n";

    // Labels
    svg << "<text x=\"" << width/2 << "\" y=\"" << height - 10 << "\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">Steps</text>\n";
    std::string y_label = use_log ? "Log10 Loss" : "Loss";
    svg << "<text x=\"20\" y=\"" << height/2 << "\" text-anchor=\"middle\" transform=\"rotate(-90 20 " << height/2 << ")\" font-family=\"Arial\" font-size=\"14\">" << y_label << "</text>\n";

    auto map_x = [&](size_t i) {
        return padding + (double)i / (losses.size() - 1) * (width - 2*padding);
    };
    auto map_y = [&](float l) {
        double val = use_log ? std::log10(l) : l;
        return height - padding - ((val - y_min) / y_range * (height - 2*padding));
    };

    svg << "<path d=\"M";
    for (size_t i = 0; i < losses.size(); ++i) {
        svg << map_x(i) << " " << map_y(losses[i]);
        if (i < losses.size() - 1) svg << " L ";
    }
    svg << "\" fill=\"none\" stroke=\"blue\" stroke-width=\"2\" />\n";

    // Ticks
    for(int i=0; i<=5; ++i) {
        // X
        double x = padding + i * (width - 2*padding)/5.0;
        size_t step = (losses.size()-1) * i / 5;
        svg << "<text x=\"" << x << "\" y=\"" << height - padding + 20 << "\" text-anchor=\"middle\" font-size=\"12\">" << step << "</text>\n";

        // Y
        double y = height - padding - i * (height - 2*padding)/5.0;
        double val = y_min + i * y_range / 5.0;
        if (use_log) val = std::pow(10, val);
        svg << "<text x=\"" << padding - 5 << "\" y=\"" << y + 5 << "\" text-anchor=\"end\" font-size=\"12\">" << std::scientific << std::setprecision(1) << val << "</text>\n";
    }

    svg << "</svg>\n";
    svg.close();
}

// Simple Model Serializer
template <typename T>
void save_model(ZenithHierarchicalAE<T>& model, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to save model to " << filename << std::endl;
        return;
    }
    auto params = model.parameters();
    size_t count = params.size();
    out.write((char*)&count, sizeof(size_t));
    for (auto* p : params) {
        size_t sz = p->size();
        out.write((char*)&sz, sizeof(size_t));
        out.write((char*)p->data(), sz * sizeof(T));
    }
    out.close();
}

template <typename T>
void load_model(ZenithHierarchicalAE<T>& model, const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to load model from " << filename << std::endl;
        return;
    }
    auto params = model.parameters();
    size_t count;
    in.read((char*)&count, sizeof(size_t));
    if (count != params.size()) {
        std::cerr << "Model parameter count mismatch!" << std::endl;
        return;
    }
    for (auto* p : params) {
        size_t sz;
        in.read((char*)&sz, sizeof(size_t));
        if (sz != p->size()) {
             std::cerr << "Tensor size mismatch!" << std::endl;
             // Skip reading or abort? Abort for safety.
             return;
        }
        in.read((char*)p->data(), sz * sizeof(T));
    }
    in.close();
    std::cout << "Model loaded from " << filename << std::endl;
}

// Helper to save tensor as image
template <typename T>
void save_tensor_as_png(const Tensor<T>& tensor, size_t batch_idx, size_t H, size_t W, const std::string& filename) {
    // Tensor is NHWC. 3 Channels.
    std::vector<unsigned char> image(H * W * 3);
    const T* data = tensor.data();
    size_t C = 3;
    size_t offset = batch_idx * H * W * C;

    // Auto-normalization
    float min_val = 1e9, max_val = -1e9;
    for (size_t i = 0; i < H * W * C; ++i) {
        float val = data[offset + i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    // Avoid div by zero
    if (max_val - min_val < 1e-6) max_val = min_val + 1.0f;

    for (size_t i = 0; i < H * W; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            float val = data[offset + i * 3 + c];
            val = (val - min_val) / (max_val - min_val); // Normalize to [0, 1]
            val = std::max(0.0f, std::min(1.0f, val));
            image[i * 3 + c] = static_cast<unsigned char>(val * 255.0f);
        }
    }

    stbi_write_png(filename.c_str(), W, H, 3, image.data(), W * 3);
    std::cout << "Saved " << filename << std::endl;
}

int main() {
    std::cout << "=== Zenith Wavelet Denoise (100 Epochs, Best Save) ===\n";

    // Disable Fused Kernels for stability
    ZenithBlock<float>::use_fused_kernels = false;

    const size_t H = 32;
    const size_t W = 32;
    const size_t C = 16;
    const size_t BatchSize = 2;
    const size_t Epochs = 100;
    const size_t StepsPerEpoch = 5;

    WaveletGenerator2D<float> gen(H, W);
    Tensor<float> batch_input({BatchSize, H, W, 3});
    Tensor<float> batch_grad({BatchSize, H, W, 3});

    // Default Init (Scaled He 0.1)
    ZenithHierarchicalAE<float> model(C);
    SimpleAdam<float> optimizer(1e-3);
    optimizer.add_parameters(model.parameters(), model.gradients());

    std::vector<float> loss_history;
    auto start_time = std::chrono::high_resolution_clock::now();

    float best_loss = 1e9;
    std::string best_model_path = "best_zenith_model.bin";

    for (size_t epoch = 0; epoch < Epochs; ++epoch) {
        float epoch_loss = 0;
        for (size_t step = 0; step < StepsPerEpoch; ++step) {
            gen.generate_batch(batch_input, BatchSize);
            optimizer.zero_grad();
            Tensor<float> out = model.forward(batch_input);
            float loss = mse_loss(out, batch_input, batch_grad);
            model.backward(batch_grad);
            optimizer.step();

            epoch_loss += loss;
            loss_history.push_back(loss);
        }
        float avg_loss = epoch_loss / StepsPerEpoch;

        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            save_model(model, best_model_path);
            // std::cout << "New best loss: " << best_loss << " (Saved)\n";
        }

        if (epoch % 10 == 0 || epoch == Epochs - 1) {
            std::cout << "Epoch " << epoch + 1 << " Loss: " << avg_loss << " Best: " << best_loss << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Total Time: " << std::chrono::duration<double>(end_time - start_time).count() << "s\n";
    std::cout << "Final Loss: " << loss_history.back() << std::endl;
    std::cout << "Best Loss: " << best_loss << std::endl;

    generate_loss_svg(loss_history, Epochs);
    std::cout << "Saved zenith_100_loss.svg\n";

    // Load Best Model
    load_model(model, best_model_path);

    // Generate Reconstruction Example
    std::cout << "Generating Reconstruction Examples...\n";
    gen.generate_batch(batch_input, BatchSize); // Generate fresh batch
    Tensor<float> out = model.forward(batch_input);

    save_tensor_as_png(batch_input, 0, H, W, "input.png");
    save_tensor_as_png(out, 0, H, W, "reconstruction.png");

    return 0;
}
