#include <dreidel/dreidel.hpp>
#include <cassert>
#include <iostream>
#include <cmath>

using namespace dreidel;

void test_creation() {
    std::cout << "Testing Tensor Creation..." << std::endl;
    Tensor<float> t({2, 3});
    assert(t.shape()[0] == 2);
    assert(t.shape()[1] == 3);
    assert(t.size() == 6);
    std::cout << "PASS" << std::endl;
}

void test_addition() {
    std::cout << "Testing Tensor Addition..." << std::endl;
    Tensor<float> t1({2, 2}, {1, 2, 3, 4});
    Tensor<float> t2({2, 2}, {10, 20, 30, 40});
    auto t3 = t1 + t2;

    assert(t3[0] == 11);
    assert(t3[3] == 44);
    std::cout << "PASS" << std::endl;
}

void test_matmul() {
    std::cout << "Testing Tensor Matmul (GEMM)..." << std::endl;
    // 2x3 * 3x2 = 2x2
    // [1 2 3]   [7 8]
    // [4 5 6] * [9 1]
    //           [2 3]
    //
    // [1*7+2*9+3*2  1*8+2*1+3*3] = [7+18+6  8+2+9] = [31 19]
    // [4*7+5*9+6*2  4*8+5*1+6*3] = [28+45+12 32+5+18] = [85 55]

    Tensor<float> a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> b({3, 2}, {7, 8, 9, 1, 2, 3});

    auto c = a.matmul(b);

    assert(c.shape()[0] == 2);
    assert(c.shape()[1] == 2);

    assert(c[0] == 31);
    assert(c[1] == 19);
    assert(c[2] == 85);
    assert(c[3] == 55);
    std::cout << "PASS" << std::endl;
}

void test_broadcasting() {
    std::cout << "Testing Tensor Broadcasting..." << std::endl;
    // A: (2, 3)
    // B: (3) -> Broadcast to (2, 3)
    Tensor<float> a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor<float> b({3}, {10, 20, 30});

    // a * b should be:
    // [1*10, 2*20, 3*30]
    // [4*10, 5*20, 6*30]
    auto c = a * b;

    assert(c[0] == 10);
    assert(c[1] == 40);
    assert(c[2] == 90);
    assert(c[3] == 40);
    assert(c[4] == 100);
    assert(c[5] == 180);
    std::cout << "PASS" << std::endl;
}

int main() {
    test_creation();
    test_addition();
    test_matmul();
    test_broadcasting();
    std::cout << "All Tensor tests passed!" << std::endl;
    return 0;
}
