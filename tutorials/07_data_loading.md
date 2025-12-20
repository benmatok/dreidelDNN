# Data Loading

dreidelDNN does not have a complex `DataLoader` class like PyTorch. Instead, you manage data using `Tensor` objects directly.

## Basic Batching

Data is typically stored in a large Tensor `(TotalSamples, Features)`. You iterate over it in batches.

```cpp
Tensor<float> dataset({60000, 784}); // e.g., MNIST
Tensor<float> labels({60000, 10});

size_t batch_size = 64;
size_t num_samples = dataset.shape()[0];

for (size_t i = 0; i < num_samples; i += batch_size) {
    size_t current_batch_size = std::min(batch_size, num_samples - i);

    // Create a view or copy for the batch
    // Since slicing isn't fully automatic for arbitrary dims, we often just point or copy.
    // A simple helper function to slice rows:

    Tensor<float> batch_input = slice_rows(dataset, i, current_batch_size);
    Tensor<float> batch_label = slice_rows(labels, i, current_batch_size);

    // Training step...
}
```

## Helper: Slice Rows

You can implement a helper to extract batches:

```cpp
Tensor<float> slice_rows(const Tensor<float>& source, size_t start, size_t count) {
    std::vector<size_t> shape = source.shape();
    shape[0] = count;

    Tensor<float> batch(shape);
    size_t row_size = source.strides()[0]; // Assuming row-major contiguous
    // Or simpler: size / dim[0]
    size_t elem_per_row = source.size() / source.shape()[0];

    const float* src_ptr = source.data() + start * elem_per_row;
    float* dst_ptr = batch.data();

    std::copy(src_ptr, src_ptr + count * elem_per_row, dst_ptr);
    return batch;
}
```

## Preprocessing

Data loading and preprocessing (resizing images, normalization) is usually done in C++ using external libraries (like OpenCV or stb_image) or pre-processed in Python and saved as binary files or `.drdl` tensors.

## Reading Binary Data

If you prepared data in Python (e.g., `numpy.tofile()`), you can load it easily:

```cpp
std::vector<float> buffer(file_size / sizeof(float));
file.read(reinterpret_cast<char*>(buffer.data()), file_size);
Tensor<float> t({N, C, H, W}, buffer);
```
