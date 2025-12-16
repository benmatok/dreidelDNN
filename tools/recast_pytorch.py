
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import scipy.linalg
import numpy as np
import pickle
import os
import argparse
import struct

def solve_diagonal_for_linear(model_layer, input_dim, output_dim, target_dim=1024, batch_size=128):
    """
    Solve for diagonal D such that LinearWHT approximates model_layer.
    """

    # 1. Generate calibration data
    X = torch.randn(batch_size, input_dim)

    # 2. Get target output
    with torch.no_grad():
        Y_target = model_layer(X) # (B, output_dim)

    # 3. Pad to target_dim
    X_padded = torch.zeros(batch_size, target_dim)
    X_padded[:, :input_dim] = X

    Y_padded = torch.zeros(batch_size, target_dim)
    Y_padded[:, :output_dim] = Y_target

    # 4. Solve
    X_np = X_padded.numpy()
    Y_np = Y_padded.numpy()

    # Hadamard matrix
    H = scipy.linalg.hadamard(target_dim).astype(np.float32)

    # IFWHT(y) = H * y / N
    Y_transformed = np.dot(Y_np, H) / target_dim

    # Solve D[j] = sum(X[b,j] * Y'[b,j]) / sum(X[b,j]^2)
    numerator = np.sum(X_np * Y_transformed, axis=0)
    denominator = np.sum(X_np**2, axis=0)

    D = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    return D

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

def recast_vit(model_name="google/vit-base-patch16-224"):
    print(f"Loading {model_name}...")
    model = ViTModel.from_pretrained(model_name)
    model.eval()

    weights_list = []

    print("Recasting layers...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features

            in_target = 1 << (in_features - 1).bit_length()
            if in_target < in_features: in_target *= 2

            out_target = 1 << (out_features - 1).bit_length()
            if out_target < out_features: out_target *= 2

            target_size = max(in_target, out_target)
            if target_size < 1024: target_size = 1024 # Min size for uniformity if desired, but powers of 2 are fine.

            print(f"Recasting {name}: {in_features}->{out_features} to LinearWHT({target_size})")

            D = solve_diagonal_for_linear(module, in_features, out_features, target_size)

            layer_info = {
                "name": name,
                "type": "LinearWHT",
                "dim": target_size,
                "scale": D,
                "bias": None
            }

            if module.bias is not None:
                # Bias needs to be padded to target_dim
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

            # Write scale
            write_tensor(f, layer["scale"])

            # Write bias
            if layer["bias"] is not None:
                f.write(struct.pack('?', True))
                write_tensor(f, layer["bias"])
            else:
                f.write(struct.pack('?', False))

    print(f"Saved recasted weights to {output_path}")

if __name__ == "__main__":
    recast_vit()
