#ifndef DREIDEL_IO_CHECKPOINT_HPP
#define DREIDEL_IO_CHECKPOINT_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "../core/Tensor.hpp"

namespace dreidel {
namespace io {

class CheckpointManager {
public:
    static bool save(const std::string& filepath, const std::vector<Tensor<float>*>& parameters) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "CheckpointManager: Failed to open " << filepath << " for writing." << std::endl;
            return false;
        }

        // Write number of tensors
        size_t num_params = parameters.size();
        file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

        for (const auto* param : parameters) {
            // Write shape rank
            auto shape = param->shape();
            size_t rank = shape.size();
            file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));

            // Write shape dims
            file.write(reinterpret_cast<const char*>(shape.data()), rank * sizeof(size_t));

            // Write data size (bytes)
            size_t data_size = param->size() * sizeof(float);
            file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));

            // Write data
            file.write(reinterpret_cast<const char*>(param->data()), data_size);
        }

        file.close();
        if (!file) {
             std::cerr << "CheckpointManager: Error occurred while writing to " << filepath << std::endl;
             return false;
        }

        std::cout << "Saved checkpoint to " << filepath << std::endl;
        return true;
    }

    static bool load(const std::string& filepath, const std::vector<Tensor<float>*>& parameters) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return false; // File doesn't exist or can't be opened
        }

        size_t num_params = 0;
        file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

        if (num_params != parameters.size()) {
            std::cerr << "CheckpointManager: Parameter count mismatch. File: " << num_params
                      << ", Model: " << parameters.size() << std::endl;
            return false;
        }

        for (auto* param : parameters) {
            size_t rank = 0;
            file.read(reinterpret_cast<char*>(&rank), sizeof(rank));

            std::vector<size_t> file_shape(rank);
            file.read(reinterpret_cast<char*>(file_shape.data()), rank * sizeof(size_t));

            // Validate shape
            auto model_shape = param->shape();
            if (model_shape != file_shape) {
                std::cerr << "CheckpointManager: Shape mismatch." << std::endl;
                return false;
            }

            size_t data_size = 0;
            file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));

            if (data_size != param->size() * sizeof(float)) {
                std::cerr << "CheckpointManager: Size mismatch." << std::endl;
                return false;
            }

            file.read(reinterpret_cast<char*>(param->data()), data_size);
        }

        std::cout << "Loaded checkpoint from " << filepath << std::endl;
        return true;
    }
};

} // namespace io
} // namespace dreidel

#endif // DREIDEL_IO_CHECKPOINT_HPP
