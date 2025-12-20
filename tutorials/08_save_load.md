# Saving and Loading Models

dreidelDNN uses a custom binary format (`.drdl`) or simple serialization interfaces to save model weights.

## The Serializer Interface

Located in `include/dreidel/io/`, the `Serializer` class handles reading and writing.

```cpp
#include <dreidel/io/Serializer.hpp>

// Saving
io::SimpleSerializer serializer;
serializer.save("model_weights.drdl", model_tensor);

// Loading
Tensor<float> loaded_tensor;
serializer.load("model_weights.drdl", loaded_tensor);
```

## Saving a Full Model

Currently, saving a full `Sequential` model involves iterating over its layers and saving their parameters.

```cpp
void save_model(const models::Sequential<float>& model, const std::string& prefix) {
    auto params = model.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        std::string filename = prefix + "_param_" + std::to_string(i) + ".drdl";
        io::SimpleSerializer::save(filename, *params[i]);
    }
}
```

## The DRDL Format

The format is simple:
1. **Magic String**: "DRDL"
2. **Rank**: `uint64_t`
3. **Shape**: `uint64_t` array of size `Rank`
4. **Data**: Raw bytes (float/double)

## Using Python Tools

You can use `tools/recast_pytorch.py` (or write your own script) to inspect or create these files from Python.

```python
import struct
import numpy as np

def save_drdl(filename, tensor):
    with open(filename, 'wb') as f:
        # Magic
        f.write(b'DRDL')
        # ... implementation details ...
```
