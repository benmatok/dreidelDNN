
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
    # DeepSpectralLinear reader expects size, then data (uint64)
    size = len(p)
    f.write(struct.pack('I', size))
    # Write data as uint64
    # struct pack 'Q' is unsigned long long (64 bit)
    # We can write one by one or pack all
    # Packing all might be faster but string might be too long?
    # Python struct format string limits?
    # Let's write array directly if possible or loop

    # Convert to array of uint64
    arr = np.array(p, dtype=np.uint64)
    f.write(arr.tobytes())

def recast_vit(model_name="google/vit-base-patch16-224"):
    print(f"Loading {model_name}...")
    model = ViTModel.from_pretrained(model_name)
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
            if target_size < 1024: target_size = 1024

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

    # Generate Validation Data
    print("Generating validation data...")

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        embeddings = model.embeddings(dummy_input) # (1, 197, 768)
        original_output = model.encoder(embeddings).last_hidden_state # (1, 197, 768)
        original_pooled = model.pooler(original_output) # (1, 768)

    # Save validation data
    val_path = "vit_validation_data.bin"
    with open(val_path, "wb") as f:
        # Input (Embeddings)
        write_tensor(f, embeddings.numpy())
        # Output (Pooled)
        write_tensor(f, original_pooled.numpy())

    print(f"Saved validation data to {val_path}")

if __name__ == "__main__":
    recast_vit()
