
import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperConfig
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

def recast_whisper(model_name="ivrit-ai/whisper-large-v3", min_dim=256, batch_size=1, seq_len=100, generate_data=True):
    print(f"Loading {model_name}...")
    try:
        # Load config first to check size? No, just try loading model.
        # Use float32 to be safe on CPU, or float16 if supported.
        # To avoid downloading huge files if not needed, we could use a smaller model for testing logic,
        # but the user asked for this specific one.
        model = WhisperModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    model.eval()

    weights_list = []

    print("Recasting layers to DeepSpectralLinear (Uninitialized for Distillation)...")

    # Roadmap says: "Replace LinearWHT with DeepSpectralLinear (K=4)."
    DEPTH = 4

    # We want to recast Linear layers in Encoder and Decoder layers
    # We should avoid recasting the projection layer (if any) or embeddings if they are Linear (unlikely)

    # Iterate all modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if it's part of the encoder or decoder layers to be safe
            # Whisper structure: model.encoder.layers.0.self_attn.k_proj, etc.
            if "layers" not in name:
                print(f"Skipping {name} (not in transformer layers)")
                continue

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
                # Scale: Random initialization
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
    output_path = "whisper_spectral_weights.bin"
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

    if not generate_data:
        return

    # Generate Distillation Data (Synthetic)
    print("Generating distillation data (synthetic)...")

    # Whisper expects input_features: (batch, feature_size, sequence_length)
    # feature_size is 80 or 128 depending on model. Config has it.
    feature_size = model.config.num_mel_bins
    # seq_len for Whisper is usually 3000 (30s) but can be less?
    # Actually for training we might want full blocks.
    # We'll use the passed seq_len.

    synthetic_input = torch.randn(batch_size, feature_size, seq_len)

    # Decoder input ids (start token + random tokens)
    decoder_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len // 2))
    decoder_input_ids[:, 0] = model.config.decoder_start_token_id

    # Capture layer-wise data
    layer_inputs = {}
    layer_outputs = {}

    def get_hook(layer_name, is_input=False):
        if is_input:
            def hook(module, input):
                # input is tuple
                if len(input) > 0:
                     layer_inputs[layer_name] = input[0].detach()
            return hook
        else:
            def hook(module, input, output):
                # output is tuple (hidden_states, ...)
                if isinstance(output, tuple):
                    layer_outputs[layer_name] = output[0].detach()
                else:
                    layer_outputs[layer_name] = output.detach()
            return hook

    handles = []

    # Register hooks for Encoder Layers
    for i, layer_module in enumerate(model.encoder.layers):
        name = f"encoder.layers.{i}"
        h1 = layer_module.register_forward_hook(get_hook(name, is_input=False))
        h2 = layer_module.register_forward_pre_hook(get_hook(name, is_input=True))
        handles.append(h1)
        handles.append(h2)

    # Register hooks for Decoder Layers
    for i, layer_module in enumerate(model.decoder.layers):
        name = f"decoder.layers.{i}"
        h1 = layer_module.register_forward_hook(get_hook(name, is_input=False))
        h2 = layer_module.register_forward_pre_hook(get_hook(name, is_input=True))
        handles.append(h1)
        handles.append(h2)

    # Run model
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_features=synthetic_input, decoder_input_ids=decoder_input_ids)

    # Cleanup hooks
    for h in handles: h.remove()

    # Save layer-wise data
    distill_path = "whisper_layer_data.bin"
    with open(distill_path, "wb") as f:
        # We need to know order or names.
        # We'll write pairs: name_len, name, input, output

        # Total number of captured layers
        num_layers = len(layer_inputs)
        f.write(struct.pack('I', num_layers))

        # Sort keys to be deterministic
        # We want encoder 0..N then decoder 0..N
        sorted_keys = []
        for i in range(len(model.encoder.layers)):
            sorted_keys.append(f"encoder.layers.{i}")
        for i in range(len(model.decoder.layers)):
            sorted_keys.append(f"decoder.layers.{i}")

        for name in sorted_keys:
            if name in layer_inputs and name in layer_outputs:
                write_string(f, name)
                write_tensor(f, layer_inputs[name].numpy())
                write_tensor(f, layer_outputs[name].numpy())
            else:
                print(f"Warning: Missing data for {name}")

    print(f"Saved layer-wise distillation data to {distill_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ivrit-ai/whisper-large-v3", help="HuggingFace model name")
    parser.add_argument("--min-dim", type=int, default=256, help="Minimum spectral dimension")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic data")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence length")
    parser.add_argument("--no-data", action="store_true", help="Skip data generation")
    args = parser.parse_args()

    recast_whisper(args.model, args.min_dim, args.batch_size, args.seq_len, not args.no_data)
