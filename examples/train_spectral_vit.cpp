
#include "../include/dreidel/models/SpectralViT.hpp"
#include "../include/dreidel/optim/DiagonalNewton.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>

using namespace dreidel;

// Helper to read tensor from file
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

    t = Tensor<T>(shape);
    size_t num_elements = t.size();
    f.read(reinterpret_cast<char*>(t.data()), num_elements * sizeof(T));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights_path> <distillation_data_path>" << std::endl;
        return 1;
    }

    std::string weights_path = argv[1];
    std::string data_path = argv[2];

    try {
        std::cout << "--- Spectral ViT Distillation Training ---" << std::endl;

        models::SpectralViT<float> model;
        std::cout << "Loading Student Weights from " << weights_path << "..." << std::endl;
        model.load_weights(weights_path);

        Tensor<float> inputs;
        Tensor<float> targets;

        std::cout << "Loading Distillation Data from " << data_path << "..." << std::endl;
        std::ifstream f(data_path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open data file");

        read_tensor_util(f, inputs);
        read_tensor_util(f, targets);

        std::cout << "Data Loaded. Input: ";
        for(auto d : inputs.shape()) std::cout << d << " ";
        std::cout << "Target: ";
        for(auto d : targets.shape()) std::cout << d << " ";
        std::cout << std::endl;

        // Optimizer
        // Low learning rate because FWHT scales energy
        float lr = 1e-5;
        optim::DiagonalNewton<float> optimizer(lr);

        optimizer.add_parameters(model.parameters(), model.gradients(), model.curvatures());

        int epochs = 5;
        // In a real scenario we would batch. Here we treat the whole file as one batch (if memory allows).
        // Or loop over batch dimension manually if implemented.
        // Assuming data is (Batch, Seq, Dim).
        // We will run forward on the whole tensor.

        std::cout << "Starting Distillation Loop (" << epochs << " epochs)..." << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto start = std::chrono::high_resolution_clock::now();

            optimizer.zero_grad();

            // Forward
            Tensor<float> output = model.forward(inputs);

            // Post-process output to match target for loss computation
            // Output: (Batch, Seq, 1024). Target: (Batch, 768).
            // We need to extract CLS token (seq 0) and slice to 768.

            size_t batch = output.shape()[0];
            size_t seq = output.shape()[1];
            size_t dim = output.shape()[2];
            size_t target_dim = targets.shape()[1];

            Tensor<float> final_output({batch, target_dim});

            // Extract and compute loss gradient
            Tensor<float> grad_output(output.shape()); // Gradient w.r.t raw output
            grad_output.fill(0);

            float loss = 0;

            #pragma omp parallel for reduction(+:loss)
            for (long b = 0; b < (long)batch; ++b) {
                for (size_t d = 0; d < target_dim; ++d) {
                    float y_pred = output.data()[b*seq*dim + d]; // CLS token at index 0 (offset 0)
                    float y_true = targets.data()[b*target_dim + d];
                    float diff = y_pred - y_true;
                    loss += diff * diff;

                    // dLoss/dOutput
                    // L = sum(diff^2) -> dL = 2*diff
                    // Store in the correct place in grad_output
                    grad_output.data()[b*seq*dim + d] = 2.0f * diff / (batch * target_dim); // Mean reduction
                }
            }
            loss /= (batch * target_dim);

            // Backward
            model.backward(grad_output);

            // Optimize
            optimizer.step();

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "Epoch " << epoch+1 << "/" << epochs << " - Loss: " << loss << " (" << ms << " ms)" << std::endl;
        }

        std::cout << "Distillation Complete." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
