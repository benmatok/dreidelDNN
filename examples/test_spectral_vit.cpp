
#include "../include/dreidel/models/SpectralViT.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

using namespace dreidel;

// Helper to read tensor from file (duplicated from SpectralViT for test util)
template<typename T>
void read_tensor_util(std::ifstream& f, Tensor<T>& t) {
    uint32_t rank;
    f.read(reinterpret_cast<char*>(&rank), 4);

    std::vector<size_t> shape(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        uint32_t d;
        f.read(reinterpret_cast<char*>(&d), 4);
        shape[i] = d;
    }

    // Create tensor with read shape
    t = Tensor<T>(shape);

    size_t num_elements = t.size();
    f.read(reinterpret_cast<char*>(t.data()), num_elements * sizeof(T));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <weights_path> [validation_data_path]" << std::endl;
        return 1;
    }

    std::string weights_path = argv[1];
    std::string val_path = (argc > 2) ? argv[2] : "";

    try {
        std::cout << "--- Spectral ViT Inference Test ---" << std::endl;
        std::cout << "Initializing model..." << std::endl;
        models::SpectralViT<float> model;

        std::cout << "Loading weights from " << weights_path << "..." << std::endl;
        auto start_load = std::chrono::high_resolution_clock::now();
        model.load_weights(weights_path);
        auto end_load = std::chrono::high_resolution_clock::now();
        std::cout << "Weights loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count() << " ms." << std::endl;

        Tensor<float> input;
        Tensor<float> target;
        bool has_validation = false;

        if (!val_path.empty()) {
            std::cout << "Loading validation data from " << val_path << "..." << std::endl;
            std::ifstream f(val_path, std::ios::binary);
            if (f.is_open()) {
                read_tensor_util(f, input);
                read_tensor_util(f, target);
                has_validation = true;
                std::cout << "Validation data loaded." << std::endl;
                std::cout << "Input Shape: "; for(auto d : input.shape()) std::cout << d << " "; std::cout << std::endl;
                std::cout << "Target Shape: "; for(auto d : target.shape()) std::cout << d << " "; std::cout << std::endl;
            } else {
                std::cerr << "Failed to open validation data file." << std::endl;
            }
        }

        if (!has_validation) {
            std::cout << "No validation data provided. Using random input." << std::endl;
            input = Tensor<float>({1, 197, 768});
            input.random(0, 1);
        }

        // Warmup
        std::cout << "Warming up..." << std::endl;
        model.forward(input);

        // Benchmark
        int num_runs = 20;
        std::cout << "Benchmarking (" << num_runs << " runs)..." << std::endl;
        auto start_bench = std::chrono::high_resolution_clock::now();
        for(int i=0; i<num_runs; ++i) {
            Tensor<float> out = model.forward(input);
            // Prevent optimization
            if (out.data()[0] == 123456.0f) std::cout << "!" << std::endl;
        }
        auto end_bench = std::chrono::high_resolution_clock::now();
        double avg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_bench - start_bench).count() / (double)num_runs;
        std::cout << "Average Inference Time: " << avg_ms << " ms" << std::endl;

        // Validation
        if (has_validation) {
            Tensor<float> output = model.forward(input);

            // Compare output with target
            std::cout << "Raw Output Shape: "; for(auto d : output.shape()) std::cout << d << " "; std::cout << std::endl;

            // Post-process output to match target (Pooler CLS token + Slice)
            // Target is (1, 768). Output is (1, 197, 1024).
            // We need to take [0, 0, :] (Batch 0, Seq 0, All Dims) and slice to 768.

            Tensor<float> final_output;
            bool shape_match = false;

            if (output.shape().size() == 3 && target.shape().size() == 2) {
                // Extract CLS token
                // Assuming output is (Batch, Seq, Dim)
                size_t batch = output.shape()[0];
                size_t seq = output.shape()[1];
                size_t dim = output.shape()[2];
                size_t target_dim = target.shape()[1];

                std::cout << "Extracting CLS token and slicing to " << target_dim << "..." << std::endl;

                final_output = Tensor<float>({batch, target_dim});

                for(size_t b=0; b<batch; ++b) {
                    for(size_t d=0; d<target_dim; ++d) {
                        // CLS token is at seq index 0
                        final_output.data()[b*target_dim + d] = output.data()[b*seq*dim + 0*dim + d];
                    }
                }
                shape_match = true;
            } else if (output.shape() == target.shape()) {
                final_output = output;
                shape_match = true;
            }

            if (!shape_match) {
                std::cerr << "Shape mismatch between output and target! Cannot auto-adapt." << std::endl;
            } else {
                float mse = 0;
                float max_diff = 0;
                float norm_target = 0;
                float norm_diff = 0;

                size_t n = final_output.size();
                for(size_t i=0; i<n; ++i) {
                    float diff = final_output.data()[i] - target.data()[i];
                    mse += diff * diff;
                    if (std::abs(diff) > max_diff) max_diff = std::abs(diff);
                    norm_target += target.data()[i] * target.data()[i];
                    norm_diff += diff * diff;
                }
                mse /= n;
                float rel_error = std::sqrt(norm_diff) / std::sqrt(norm_target);

                std::cout << "Validation Results:" << std::endl;
                std::cout << "  MSE: " << mse << std::endl;
                std::cout << "  Max Diff: " << max_diff << std::endl;
                std::cout << "  Relative Error (L2): " << rel_error << std::endl;

                if (rel_error > 0.1) {
                    std::cout << "  [WARNING] High relative error. This is expected if the model was only recasted (projected initialization) without distillation training." << std::endl;
                } else {
                    std::cout << "  [SUCCESS] Low relative error. Recasting is accurate." << std::endl;
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
