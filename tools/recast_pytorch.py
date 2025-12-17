
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import numpy as np
import os
import argparse
import struct
import random

def write_string(f, s):
    encoded = s.encode('utf-8')
    f.write(struct.pack('I', len(encoded)))
    f.write(encoded)

def write_tensor(f, t):
    # t is numpy array
    # Write dims
    f.write(struct.pack('I', len(t.shape)))
    for d in t.shape:
        f.write(struct.pack('I', d))
    # Write data
    f.write(t.astype(np.float32).tobytes())

def write_perm(f, p):
    # p is list/array of integers
    # Write size (which is also dim)
    size = len(p)
    f.write(struct.pack('I', size))
    # Convert to array of uint64
    arr = np.array(p, dtype=np.uint64)
    f.write(arr.tobytes())

def recast_vit(model_name="google/vit-base-patch16-224", min_dim=256, batch_size=32):
    print(f"Loading {model_name}...")
    try:
        model = ViTModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    model.eval()

    weights_list = []

    print("Recasting layers to DeepSpectralLinear (Uninitialized for Distillation)...")

    # Roadmap says: "Replace LinearWHT with DeepSpectralLinear (K=4)."
    DEPTH = 4

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features

            in_target = 1 << (in_features - 1).bit_length()
            if in_target < in_features: in_target *= 2

            out_target = 1 << (out_features - 1).bit_length()
            if out_target < out_features: out_target *= 2

            target_size = max(in_target, out_target)
            if target_size < min_dim: target_size = min_dim

            print(f"Converting {name}: {in_features}->{out_features} to DeepSpectralLinear({target_size}, K={DEPTH})")

            # Generate Parameters for DeepSpectralLinear
            scales = []
            perms = []

            for k in range(DEPTH):
                # Scale: Random initialization as per roadmap/cpp
                # C++ uses: mean 0, stddev 1/sqrt(dim)
                stddev = 1.0 / np.sqrt(target_size)
                scale = np.random.normal(0, stddev, target_size).astype(np.float32)
                scales.append(scale)

                # Permutation: Random
                perm = np.arange(target_size)
                np.random.shuffle(perm)
                perms.append(perm)

            layer_info = {
                "name": name,
                "type": "DeepSpectralLinear",
                "dim": target_size,
                "depth": DEPTH,
                "scales": scales,
                "perms": perms,
                "bias": None
            }

            if module.bias is not None:
                bias_padded = np.zeros(target_size, dtype=np.float32)
                bias_padded[:out_features] = module.bias.detach().numpy()
                layer_info["bias"] = bias_padded

            weights_list.append(layer_info)

    # Save weights to binary file
    output_path = "vit_spectral_weights.bin"
    with open(output_path, "wb") as f:
        # Magic number
        f.write(b"DRDL")
        # Version
        f.write(struct.pack('I', 1))
        # Number of layers
        f.write(struct.pack('I', len(weights_list)))

        for layer in weights_list:
            write_string(f, layer["name"])
            write_string(f, layer["type"])
            f.write(struct.pack('I', layer["dim"]))

            if layer["type"] == "DeepSpectralLinear":
                # Write Depth
                f.write(struct.pack('I', layer["depth"]))

                for k in range(layer["depth"]):
                    # Write Scale
                    write_tensor(f, layer["scales"][k])
                    # Write Perm
                    write_perm(f, layer["perms"][k])

            # Write bias
            if layer["bias"] is not None:
                f.write(struct.pack('?', True))
                write_tensor(f, layer["bias"])
            else:
                f.write(struct.pack('?', False))

    print(f"Saved recasted weights to {output_path}")

    # Generate Distillation Data (Synthetic)
    print("Generating distillation data (synthetic)...")

    synthetic_input = torch.randn(batch_size, 3, 224, 224)

    # We also add some "shapes" - e.g., blocks of constant value to simulate structure
    for i in range(batch_size):
        # Random block
        x = random.randint(0, 150)
        y = random.randint(0, 150)
        w = random.randint(20, 50)
        h = random.randint(20, 50)
        synthetic_input[i, :, y:y+h, x:x+w] = random.random() * 5.0 # High intensity block

    # Capture layer-wise data
    layer_inputs = {}
    layer_outputs = {}

    def get_hook(layer_idx, is_input=False):
        if is_input:
            # Pre-hook signature: hook(module, input)
            def hook(module, input):
                layer_inputs[layer_idx] = input[0].detach()
            return hook
        else:
            # Forward hook signature: hook(module, input, output)
            def hook(module, input, output):
                if isinstance(output, tuple):
                    layer_outputs[layer_idx] = output[0].detach()
                else:
                    layer_outputs[layer_idx] = output.detach()
            return hook

    # Register hooks
    handles = []
    num_encoder_layers = len(model.encoder.layer)
    for i, layer_module in enumerate(model.encoder.layer):
        h1 = layer_module.register_forward_hook(get_hook(i, is_input=False))
        h2 = layer_module.register_forward_pre_hook(get_hook(i, is_input=True))
        handles.append(h1)
        handles.append(h2)

    # Run model
    with torch.no_grad():
        embeddings = model.embeddings(synthetic_input)
        # We also need to capture Pooler input/output
        # Pooler input is Encoder output
        encoder_output = model.encoder(embeddings).last_hidden_state
        pooler_output = model.pooler(encoder_output)

    # Cleanup hooks
    for h in handles: h.remove()

    # Save layer-wise data
    distill_path = "vit_layer_data.bin"
    with open(distill_path, "wb") as f:
        # Write number of blocks (encoder + 1 pooler)
        f.write(struct.pack('I', num_encoder_layers + 1))

        # Encoder Blocks
        for i in range(num_encoder_layers):
            # Input
            write_tensor(f, layer_inputs[i].numpy())
            # Target
            write_tensor(f, layer_outputs[i].numpy())

        # Pooler
        # Input (Encoder Output)
        write_tensor(f, encoder_output.numpy())
        # Target (Pooler Output)
        write_tensor(f, pooler_output.numpy())

    print(f"Saved layer-wise distillation data to {distill_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/vit-base-patch16-224", help="HuggingFace model name")
    parser.add_argument("--min-dim", type=int, default=256, help="Minimum spectral dimension")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for synthetic data")
    args = parser.parse_args()

    recast_vit(args.model, args.min_dim, args.batch_size)
