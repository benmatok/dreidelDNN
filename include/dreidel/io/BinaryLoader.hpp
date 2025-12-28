#ifndef DREIDEL_IO_BINARYLOADER_HPP
#define DREIDEL_IO_BINARYLOADER_HPP

#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "../core/TensorView.hpp"

namespace dreidel {
namespace io {

/**
 * @brief Simple Binary Loader for Zero-Alloc environments.
 *
 * Reads raw bytes from a file into a pre-allocated TensorView.
 * Does NOT verify metadata/shapes in the file (assumes "dumb" binary dump).
 *
 * Format: Just raw bytes.
 */
class BinaryLoader {
public:
    static void load(const std::string& filename, core::TensorView<float>& view) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
             throw std::runtime_error("Failed to open file: " + filename);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t expected_bytes = view.size() * sizeof(float);
        if (static_cast<size_t>(size) != expected_bytes) {
            // For robustness, we could allow partial reads or larger files,
            // but strict check is safer.
            // Check if it matches exactly?
            if (static_cast<size_t>(size) < expected_bytes) {
                 throw std::runtime_error("File too small for tensor: " + filename);
            }
            // Warning if file is larger?
        }

        if (!file.read(reinterpret_cast<char*>(view.data()), expected_bytes)) {
             throw std::runtime_error("Read error: " + filename);
        }
    }

    // Helper to load into a raw pointer
    static void load(const std::string& filename, float* buffer, size_t count) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
             throw std::runtime_error("Failed to open file: " + filename);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t expected_bytes = count * sizeof(float);
         if (static_cast<size_t>(size) < expected_bytes) {
             throw std::runtime_error("File too small");
        }

        file.read(reinterpret_cast<char*>(buffer), expected_bytes);
    }
};

} // namespace io
} // namespace dreidel

#endif // DREIDEL_IO_BINARYLOADER_HPP
