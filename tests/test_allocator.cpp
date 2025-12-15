#include <iostream>
#include <vector>
#include <cassert>
#include "dreidel/core/Tensor.hpp"

using namespace dreidel;

void test_odd_allocation() {
    std::cout << "Testing Odd Allocation Alignment..." << std::endl;
    // Allocate a size that is NOT a multiple of 64 bytes (16 floats).
    // e.g., 10 floats = 40 bytes.
    // If Allocator is buggy, this might crash or fail.
    // With the fix, it should allocate 64 bytes (16 floats) worth of space, but report size as 10.

    try {
        Tensor<float> t({10});
        t.fill(3.14f);

        // Verify access
        for(size_t i=0; i<10; ++i) {
            if (t[i] != 3.14f) throw std::runtime_error("Data mismatch");
        }

        std::cout << "PASS" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        exit(1);
    }
}

int main() {
    test_odd_allocation();
    return 0;
}
