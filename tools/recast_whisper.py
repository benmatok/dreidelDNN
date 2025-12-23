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
    f.write(struct.pack('I', len(t.shape)))
    for d in t.shape:
        f.write(struct.pack('I', d))
    f.write(t.astype(np.float32).tobytes())

def write_perm(f, p):
    # p is list/array of integers
    size = len(p)
    f.write(struct.pack('I', size))
    arr = np.array(p, dtype=np.uint64)
    f.write(arr.tobytes())

def get_spectral_params(target_size, depth=4):
    scales = []
    perms = []
    for k in range(depth):
        # Scale: mean 0, stddev 1/sqrt(dim)
        stddev = 1.0 / np.sqrt(target_size)
        scale = np.random.normal(0, stddev, target_size).astype(np.float32)
        scales.append(scale)

        # Permutation: Random
        perm = np.arange(target_size)
        np.random.shuffle(perm)
        perms.append(perm)
    return scales, perms

def recast_whisper(model_name="openai/whisper-tiny", min_dim=256, batch_size=4, output_weights="whisper_spectral_weights.bin", output_data="whisper_layer_data.bin"):
    print(f"Loading {model_name}...")
    try:
        model = WhisperModel.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return

    model.eval()
    config = model.config
    print(f"Model Config: d_model={config.d_model}, encoder_layers={config.encoder_layers}, decoder_layers={config.decoder_layers}")

    weights_list = []
    DEPTH = 4 # DeepSpectralLinear depth

    # Helper to process a linear layer
    def process_linear(name, module):
        in_features = module.in_features
        out_features = module.out_features

        # Calculate next power of 2
        in_target = 1 << (in_features - 1).bit_length()
        if in_target < in_features: in_target *= 2 # Ensure at least equal, or next power
        if in_target < min_dim: in_target = min_dim

        out_target = 1 << (out_features - 1).bit_length()
        if out_target < out_features: out_target *= 2
        if out_target < min_dim: out_target = min_dim

        # Spectral layers usually handle square transforms or we pad input/output.
        # DeepSpectralLinear in C++ (based on earlier memory) usually works on a fixed dim per block?
        # Or it takes input dim -> output dim.
        # But WHT requires power of 2.
        # Usually we pad to max(in, out) and treat as square spectral transform + slicing?
        # Let's assume max(in, out) power of 2 for the spectral dimension.

        target_size = max(in_target, out_target)

        print(f"  Converting {name}: {in_features}->{out_features} to DeepSpectralLinear({target_size}, K={DEPTH})")

        scales, perms = get_spectral_params(target_size, DEPTH)

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

        return layer_info

    # 1. Input Projection (Simplification: We use the first conv/linear equivalent)
    # Whisper has Conv1d layers. We might model them or just a linear projection for now.
    # The C++ code had `input_proj_`. We will map the FIRST conv layer's effect or just create a dummy projection
    # that maps n_mels -> d_model.
    # Whisper Conv1: (80, d_model, k=3, s=1).
    # We will just export a spectral layer for n_mels -> d_model.
    print("Processing Input Projection...")
    # Mocking a linear module for the projection
    mock_proj = nn.Linear(config.num_mel_bins, config.d_model)
    weights_list.append(process_linear("input_proj", mock_proj))

    # 2. Encoder Layers
    print("Processing Encoder Layers...")
    for i, layer in enumerate(model.encoder.layers):
        # Self Attention Projections
        # usually q, k, v, out
        weights_list.append(process_linear(f"encoder.{i}.attn.q_proj", layer.self_attn.q_proj))
        weights_list.append(process_linear(f"encoder.{i}.attn.k_proj", layer.self_attn.k_proj))
        weights_list.append(process_linear(f"encoder.{i}.attn.v_proj", layer.self_attn.v_proj))
        weights_list.append(process_linear(f"encoder.{i}.attn.out_proj", layer.self_attn.out_proj))

        # MLP
        weights_list.append(process_linear(f"encoder.{i}.mlp.fc1", layer.fc1))
        weights_list.append(process_linear(f"encoder.{i}.mlp.fc2", layer.fc2))

    # 3. Decoder Layers
    print("Processing Decoder Layers...")
    for i, layer in enumerate(model.decoder.layers):
        # Self Attention
        weights_list.append(process_linear(f"decoder.{i}.self_attn.q_proj", layer.self_attn.q_proj))
        weights_list.append(process_linear(f"decoder.{i}.self_attn.k_proj", layer.self_attn.k_proj))
        weights_list.append(process_linear(f"decoder.{i}.self_attn.v_proj", layer.self_attn.v_proj))
        weights_list.append(process_linear(f"decoder.{i}.self_attn.out_proj", layer.self_attn.out_proj))

        # Cross Attention (Encoder-Decoder)
        weights_list.append(process_linear(f"decoder.{i}.cross_attn.q_proj", layer.encoder_attn.q_proj))
        weights_list.append(process_linear(f"decoder.{i}.cross_attn.k_proj", layer.encoder_attn.k_proj))
        weights_list.append(process_linear(f"decoder.{i}.cross_attn.v_proj", layer.encoder_attn.v_proj))
        weights_list.append(process_linear(f"decoder.{i}.cross_attn.out_proj", layer.encoder_attn.out_proj))

        # MLP
        weights_list.append(process_linear(f"decoder.{i}.mlp.fc1", layer.fc1))
        weights_list.append(process_linear(f"decoder.{i}.mlp.fc2", layer.fc2))

    # Save weights
    with open(output_weights, "wb") as f:
        f.write(b"DRDL")
        f.write(struct.pack('I', 1)) # Version
        f.write(struct.pack('I', len(weights_list))) # Num layers

        for layer in weights_list:
            write_string(f, layer["name"])
            write_string(f, layer["type"])
            f.write(struct.pack('I', layer["dim"]))

            # DeepSpectralLinear specifics
            f.write(struct.pack('I', layer["depth"]))
            for k in range(layer["depth"]):
                write_tensor(f, layer["scales"][k])
                write_perm(f, layer["perms"][k])

            # Bias
            if layer["bias"] is not None:
                f.write(struct.pack('?', True))
                write_tensor(f, layer["bias"])
            else:
                f.write(struct.pack('?', False))

    print(f"Saved weights to {output_weights}")

    # --- Distillation Data Generation ---
    print("Generating synthetic distillation data...")

    # Create synthetic input
    # Mel spectrogram: (Batch, 80, 3000) for standard Whisper
    seq_len = 3000
    mel_input = torch.randn(batch_size, config.num_mel_bins, seq_len)

    # Decoder input (tokens)
    # just random tokens
    decoder_input_ids = torch.randint(0, config.vocab_size, (batch_size, 50))

    # Hooks to capture intermediate values
    captured_data = {} # key -> tensor

    # We want to capture inputs/outputs of specific blocks to verify C++ implementation
    # For now, let's just capture the Encoders' inputs/outputs and Decoders' inputs/outputs.

    def get_hook(name):
        def hook(module, input, output):
            # Input is tuple
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input

            # Output might be tuple (hidden_state, cache, attentions)
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output

            captured_data[f"{name}_input"] = inp.detach().cpu().numpy()
            captured_data[f"{name}_output"] = out.detach().cpu().numpy()
        return hook

    # Register hooks
    hooks = []

    # Encoder Layers
    for i, layer in enumerate(model.encoder.layers):
        hooks.append(layer.register_forward_hook(get_hook(f"encoder_layer_{i}")))

    # Decoder Layers
    for i, layer in enumerate(model.decoder.layers):
        hooks.append(layer.register_forward_hook(get_hook(f"decoder_layer_{i}")))

    # Forward pass
    with torch.no_grad():
        # WhisperModel forward expects input_features (mels) and decoder_input_ids
        outputs = model(input_features=mel_input, decoder_input_ids=decoder_input_ids)

    # Clean hooks
    for h in hooks: h.remove()

    # Save data
    with open(output_data, "wb") as f:
        # Write metadata
        f.write(struct.pack('I', config.encoder_layers))
        f.write(struct.pack('I', config.decoder_layers))

        # Write Mel Input
        write_tensor(f, mel_input.numpy())

        # Write Decoder Input (Embeddings? Or IDs? C++ usually wants embeddings or handled internally)
        # The C++ SpectralWhisper forward takes `decoder_input_embeds`.
        # So we should save the embeddings of the decoder_input_ids.
        # We can extract them manually
        decoder_embeds = model.decoder.embed_tokens(decoder_input_ids) * config.d_model**0.5 + model.decoder.embed_positions(decoder_input_ids)
        write_tensor(f, decoder_embeds.detach().numpy())

        # Write Layer Data
        # Encoder
        for i in range(config.encoder_layers):
            write_tensor(f, captured_data[f"encoder_layer_{i}_input"])
            write_tensor(f, captured_data[f"encoder_layer_{i}_output"])

        # Decoder
        for i in range(config.decoder_layers):
            write_tensor(f, captured_data[f"decoder_layer_{i}_input"])
            write_tensor(f, captured_data[f"decoder_layer_{i}_output"])

    print(f"Saved distillation data to {output_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/whisper-tiny", help="HuggingFace model name")
    parser.add_argument("--min-dim", type=int, default=256, help="Minimum spectral dimension")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output-weights", type=str, default="whisper_spectral_weights.bin")
    parser.add_argument("--output-data", type=str, default="whisper_layer_data.bin")
    args = parser.parse_args()

    recast_whisper(args.model, args.min_dim, args.batch_size, args.output_weights, args.output_data)
