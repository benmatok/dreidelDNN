# Tensor Basics

The `Tensor` class is the core data structure in dreidelDNN. It supports multi-dimensional arrays, efficient memory management, and broadcasting arithmetic.

## Creating Tensors

You can create tensors in several ways.

### From Shape and Data

```cpp
#include <dreidel/dreidel.hpp>

using namespace dreidel;

// Create a 2x2 tensor with specific data
std::vector<size_t> shape = {2, 2};
std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
Tensor<float> t1(shape, data);
```

### Filled with a Value

```cpp
// Create a 3x3 tensor filled with zeros
Tensor<float> t2({3, 3}, 0.0f);
```

### Random Initialization

Tensors can be initialized with random values, often used for weights.

```cpp
Tensor<float> weights({128, 64});
// Initialize with Uniform(-0.1, 0.1)
weights.random(-0.1f, 0.1f);
```

## Accessing Data

You can access data using linear indexing or pointers.

```cpp
// Linear access
float val = t1[0]; // First element

// Pointer access for raw manipulation
float* ptr = t1.data();
```

## Arithmetic Operations

`Tensor` supports standard operators: `+`, `-`, `*`, `/`. These operations are element-wise.

```cpp
Tensor<float> a({2, 2}, 1.0f);
Tensor<float> b({2, 2}, 2.0f);

Tensor<float> c = a + b; // All elements become 3.0
Tensor<float> d = a * b; // All elements become 2.0
```

## Broadcasting

dreidelDNN supports broadcasting similar to NumPy or PyTorch. If dimensions don't match, the smaller tensor is conceptually "stretched" to match the larger one.

### Scalar Broadcasting

```cpp
Tensor<float> x({10, 10}, 1.0f);
Tensor<float> y = x + 5.0f; // Adds 5 to every element
```

### Vector Broadcasting

A common operation in Spectral layers is multiplying a batch of vectors by a scale vector.
`DeepSpectralLinear` relies on `Scale * Input`.

If you have input `(Batch, N)` and scale `(1, N)`, the scale is broadcast across the batch.

```cpp
Tensor<float> input({32, 128}); // Batch 32, dim 128
Tensor<float> scale({1, 128});   // Scale vector

// Generalized broadcasting
Tensor<float> out = input * scale;
```

## Shapes and Reshaping

You can inspect the shape:

```cpp
std::vector<size_t> s = t1.shape();
size_t rank = t1.rank(); // 2
```

Reshaping (view) is not fully dynamic in the current implementation without copying if memory layout changes, but you can create a new tensor with the same data if needed, or use specific layers that handle reshaping.

## Memory Management

Tensors use `AlignedAllocator` to ensure data is aligned to 64-byte boundaries, enabling efficient SIMD (AVX2/AVX-512) operations.
