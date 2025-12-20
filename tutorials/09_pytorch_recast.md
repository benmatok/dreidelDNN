# Recasting PyTorch Models

One of the most powerful features of dreidelDNN is the ability to take a pre-trained PyTorch model (e.g., a Vision Transformer) and "recast" it into a spectral architecture.

## The Tool: `recast_pytorch.py`

Located in `tools/`, this script automates the conversion process.

### Prerequisites

```bash
pip install torch transformers numpy
```

### Usage

To convert a standard ViT (Dense layers) to a format usable by `SpectralViT` (DeepSpectralLinear layers):

```bash
python tools/recast_pytorch.py \
    --model "google/vit-base-patch16-224" \
    --output "vit_spectral_weights/" \
    --spectral-dim 1024
```

### What it does

1. **Loads the Model**: Downloads the model from HuggingFace.
2. **Extracts Activations**: Runs a forward pass on dummy data to capture input/output pairs for every Linear layer.
3. **Generates Spectral Config**: Creates initial random permutations and scales for `DeepSpectralLinear` layers.
4. **Export**: Saves inputs, outputs, and initial weights to disk.

### Distillation in C++

Once you have the data, you use the C++ training tools (like `train_spectral_vit.cpp`) to optimize the spectral layers to match the dense behavior.

```bash
./train_spectral_vit --data vit_spectral_weights/ --block 0 --epochs 20
```

This "block-wise distillation" trains one block at a time to replicate the original PyTorch block, avoiding the need to train the whole network from scratch.
