# Getting Started with dreidelDNN

Welcome to **dreidelDNN**, a header-only C++ deep learning framework optimized for scalable CPU training and spectral acceleration.

## Prerequisites

Before you begin, ensure you have the following installed:
- A C++17 compatible compiler (e.g., `g++` 7+, `clang++` 5+, or MSVC).
- OpenMP (optional but recommended for performance).
- Python 3 (for tools and scripts).

## Installation

Since dreidelDNN is a header-only library, "installation" is as simple as cloning the repository and adding the `include/` directory to your include path.

```bash
git clone https://github.com/your-repo/dreidelDNN.git
cd dreidelDNN
```

## Your First Program: Hello World

Let's create a simple program to verify that everything is working correctly. We will create a `Tensor`, print its shape, and run a simple operation.

Create a file named `hello_dreidel.cpp`:

```cpp
#include <dreidel/dreidel.hpp>
#include <iostream>
#include <vector>

using namespace dreidel;

int main() {
    std::cout << "Initializing dreidelDNN..." << std::endl;

    // 1. Create a Tensor of shape [2, 3] filled with value 1.5
    std::vector<size_t> shape = {2, 3};
    Tensor<float> t(shape, 1.5f);

    std::cout << "Created Tensor with shape: ["
              << t.shape()[0] << ", " << t.shape()[1] << "]" << std::endl;

    // 2. Perform a simple operation: Add 2.0 to all elements
    // Note: Tensor operations support broadcasting.
    Tensor<float> result = t + 2.0f;

    std::cout << "First element after addition: " << result[0] << std::endl; // Should be 3.5

    return 0;
}
```

## compiling

To compile this program, make sure to point to the `include` directory. If you are in the root of the repository:

```bash
g++ -std=c++17 -fopenmp -Iinclude hello_dreidel.cpp -o hello_dreidel
```

Run it:

```bash
./hello_dreidel
```

You should see:

```
Initializing dreidelDNN...
Created Tensor with shape: [2, 3]
First element after addition: 3.5
```

## Project Structure

- `include/dreidel/`: Contains all the source headers.
- `examples/`: Example C++ programs.
- `tools/`: Python utilities for model conversion.
- `tests/`: Unit tests.

You are now ready to explore the specific features of dreidelDNN!
