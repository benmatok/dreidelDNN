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
#include <map>
#include <fstream>
#include <cmath>

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

// SVG Generator for Convergence Plot
void generate_convergence_svg(const std::map<std::string, std::vector<float>>& results, size_t total_steps) {
    std::ofstream svg("init_convergence.svg");
    if (!svg.is_open()) {
        std::cerr << "Error: Could not open init_convergence.svg for writing.\n";
        return;
    }

    double width = 800;
    double height = 600;
    double padding = 60;

    // Find min/max for scaling
    double max_loss = -1e9;
    double min_loss = 1e9;

    for (const auto& [scheme, losses] : results) {
        for (float l : losses) {
            if (l > max_loss) max_loss = l;
            if (l < min_loss) min_loss = l;
        }
    }

    // Log scale for Y
    // Handle 0 or negative (shouldn't happen for MSE)
    if (min_loss <= 0) min_loss = 1e-6;
    double log_min = std::log10(min_loss);
    double log_max = std::log10(max_loss);
    double log_range = log_max - log_min;
    if (log_range == 0) log_range = 1;

    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width << "\" height=\"" << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\" />\n";

    // Axes
    svg << "<line x1=\"" << padding << "\" y1=\"" << height - padding << "\" x2=\"" << width - padding << "\" y2=\"" << height - padding << "\" stroke=\"black\" />\n"; // X axis
    svg << "<line x1=\"" << padding << "\" y1=\"" << height - padding << "\" x2=\"" << padding << "\" y2=\"" << padding << "\" stroke=\"black\" />\n"; // Y axis

    // Labels
    svg << "<text x=\"" << width/2 << "\" y=\"" << height - 10 << "\" text-anchor=\"middle\">Steps</text>\n";
    svg << "<text x=\"20\" y=\"" << height/2 << "\" text-anchor=\"middle\" transform=\"rotate(-90 20 " << height/2 << ")\">Log10 Loss</text>\n";

    auto map_x = [&](size_t step) {
        return padding + (double)step / (total_steps - 1) * (width - 2*padding);
    };
    auto map_y = [&](float loss) {
        if (loss <= 0) loss = 1e-6;
        double log_l = std::log10(loss);
        double norm = (log_l - log_min) / log_range;
        return height - padding - (norm * (height - 2*padding));
    };

    // Colors
    std::map<std::string, std::string> colors;
    colors["he"] = "red";
    colors["identity"] = "black";
    colors["scaled_he_0.1"] = "green";
    colors["scaled_he_0.5"] = "blue";

    // Plot Lines
    for (const auto& [scheme, losses] : results) {
        std::string color = colors.count(scheme) ? colors[scheme] : "gray";
        svg << "<path d=\"M";
        for (size_t i = 0; i < losses.size(); ++i) {
            svg << map_x(i) << " " << map_y(losses[i]);
            if (i < losses.size() - 1) svg << " L ";
        }
        svg << "\" fill=\"none\" stroke=\"" << color << "\" stroke-width=\"2\" />\n";
    }

    // Legend
    double leg_x = width - 200;
    double leg_y = 50;
    double lh = 20;

    svg << "<rect x=\"" << leg_x - 10 << "\" y=\"" << leg_y - 10 << "\" width=\"180\" height=\"" << (results.size() * lh + 20) << "\" fill=\"white\" stroke=\"black\" opacity=\"0.8\" />\n";

    int i = 0;
    for (const auto& [scheme, _] : results) {
        std::string color = colors.count(scheme) ? colors[scheme] : "gray";
        double y = leg_y + i * lh;
        svg << "<line x1=\"" << leg_x << "\" y1=\"" << y << "\" x2=\"" << leg_x + 30 << "\" y2=\"" << y << "\" stroke=\"" << color << "\" stroke-width=\"2\" />\n";
        svg << "<text x=\"" << leg_x + 40 << "\" y=\"" << y + 5 << "\">" << scheme << "</text>\n";
        i++;
    }

    svg << "</svg>\n";
    svg.close();
    std::cout << "Graph saved to init_convergence.svg" << std::endl;
}

int main() {
    std::cout << "=== Zenith Meta-Train: Initialization Benchmark ===\n" << std::flush;

    // Disable Fused Kernels for stability
    std::cout << "Disabling Fused Kernels for stability...\n" << std::flush;
    ZenithBlock<float>::use_fused_kernels = false;

    // Config
    const size_t H = 32;
    const size_t W = 32;
    const size_t C = 16;
    const size_t BatchSize = 2; // Keep small for speed
    const size_t Epochs = 5;
    const size_t StepsPerEpoch = 5;

    std::vector<std::string> schemes = {"he", "identity", "scaled_he_0.1", "scaled_he_0.5"};

    // Generator
    std::cout << "Init Generator...\n" << std::flush;
    WaveletGenerator2D<float> gen(H, W);
    std::cout << "Allocating Tensors...\n" << std::flush;
    Tensor<float> batch_input({BatchSize, H, W, 3});
    Tensor<float> batch_grad({BatchSize, H, W, 3});

    std::map<std::string, std::vector<float>> results;

    for (const auto& scheme : schemes) {
        std::cout << "\n--- Testing Initialization: " << scheme << " ---\n" << std::flush;

        // Instantiate fresh model
        ZenithHierarchicalAE<float> model(C);
        model.reinit(scheme);

        SimpleAdam<float> optimizer(1e-3);
        optimizer.add_parameters(model.parameters(), model.gradients());

        std::vector<float> loss_history;
        auto start_time = std::chrono::high_resolution_clock::now();

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

                // Store step loss for detailed plot
                loss_history.push_back(loss);
            }
            float avg_loss = epoch_loss / StepsPerEpoch;
            std::cout << "Epoch " << epoch+1 << " Loss: " << avg_loss << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Time: " << duration << "s" << std::endl;

        results[scheme] = loss_history;
    }

    std::cout << "\n=== Summary ===\n";
    std::string best_scheme = "";
    float best_final_loss = 1e9;

    for (const auto& [scheme, losses] : results) {
        float final_loss = losses.back();
        std::cout << scheme << ": Final Loss = " << final_loss << std::endl;
        if (final_loss < best_final_loss) {
            best_final_loss = final_loss;
            best_scheme = scheme;
        }
    }

    std::cout << "\nFastest to Converge (Lowest Final Loss): " << best_scheme << std::endl;

    // Generate Plot
    generate_convergence_svg(results, Epochs * StepsPerEpoch);

    return 0;
}
