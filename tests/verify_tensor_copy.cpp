#include "../include/dreidel/core/Tensor.hpp"
#include <iostream>
#include <cassert>

using namespace dreidel;

int main() {
    Tensor<float> a({10});
    a.fill(1.0f);

    Tensor<float> b({10});
    b.fill(0.0f);

    std::cout << "Initial A[0]: " << a[0] << std::endl;
    std::cout << "Initial B[0]: " << b[0] << std::endl;

    b = a; // Assignment

    std::cout << "After b = a, B[0]: " << b[0] << std::endl;

    if (b[0] != 1.0f) {
        std::cerr << "Copy assignment FAILED!" << std::endl;
        return 1;
    }

    b.data()[0] = 2.0f; // Modify B
    std::cout << "Modified B[0] to 2.0" << std::endl;
    std::cout << "A[0]: " << a[0] << std::endl;

    if (a[0] == 2.0f) {
        std::cerr << "Deep copy FAILED! (Shared memory)" << std::endl;
        return 1;
    }

    std::cout << "Tensor Copy Verified." << std::endl;
    return 0;
}
