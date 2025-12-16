
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
    Returns D and the relative error.
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

    # 5. Verify approximation on this batch
    # Z = X * D
    Z = X_np * D
    # Y_pred = FWHT(Z)
    Y_pred_full = np.dot(Z, H.T) # FWHT is same as H for Hadamard? Yes symmetric.
    # Y_pred is the first 'output_dim' elements
    Y_pred = Y_pred_full[:, :output_dim]
    Y_true = Y_target.numpy()

    # Error
    diff = Y_pred - Y_true
    mse = np.mean(diff**2)
    rel_error = np.linalg.norm(diff) / np.linalg.norm(Y_true)

    return D, rel_error

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
    total_rel_error = 0
    count = 0

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

            # Solve
            D, rel_err = solve_diagonal_for_linear(module, in_features, out_features, target_size)

            print(f"Recasting {name}: {in_features}->{out_features} to LinearWHT({target_size}) | RelError: {rel_err:.4f}")
            total_rel_error += rel_err
            count += 1

            layer_info = {
                "name": name,
                "type": "LinearWHT",
                "dim": target_size,
                "scale": D,
                "bias": None
            }

            if module.bias is not None:
                bias_padded = np.zeros(target_size, dtype=np.float32)
                bias_padded[:out_features] = module.bias.detach().numpy()
                layer_info["bias"] = bias_padded

            weights_list.append(layer_info)

    print(f"Average Layer Relative Error: {total_rel_error/count:.4f}")

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

    # Generate Validation Data
    print("Generating validation data...")
    # Sample Input: (1, 197, 768) - but wait, the C++ model expects input to be padded to 1024?
    # Actually C++ auto-pads.
    # But for verification, let's provide the original input dimension (1024 in Recasting context? No, original ViT is 768).
    # Wait, the recasting code assumes input to the network is also transformed?
    # The first layer `query`, `key`, `value` are recasted from 768->768 to 1024->1024.
    # So the input to C++ `forward` should be the input to these layers.
    # If the input is embedding output, it is 768.
    # C++ `LinearWHT` will pad 768 -> 1024.
    # So we should save input as 768.

    # We need to run the FULL PyTorch model to get the target output.
    # But wait, our C++ implementation only implements the Encoder blocks and Pooler?
    # It doesn't implement Patch Embeddings.
    # So we should feed the output of Patch Embeddings + Positional Embeddings to the C++ model.
    # i.e., the input to the first encoder block.

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        embeddings = model.embeddings(dummy_input) # (1, 197, 768)
        # Run through encoder
        # We need the output of the encoder.
        # But wait, we replaced layers inside the encoder.
        # We want to verify that Recasted Encoder approximates Original Encoder.

        # Original Output
        original_output = model.encoder(embeddings).last_hidden_state # (1, 197, 768)
        # Pooler
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
