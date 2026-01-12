import torch
import struct
import sys
import os

# Ensure tools/ is in path if run from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from taesd import TAESD
except ImportError:
    # If not found, try importing from current directory assuming we are in tools/
    sys.path.append(".")
    from taesd import TAESD

def export_layer(f, layer, name=""):
    if hasattr(layer, 'weight'):
        # Transpose Weights: [Out, In, H, W] -> [In, H, W, Out]
        # This is CRITICAL for the C++ AVX2 implementation provided in the prompt.
        # Original: (0, 1, 2, 3) -> New: (1, 2, 3, 0)
        w = layer.weight.detach().permute(1, 2, 3, 0).cpu().numpy().flatten()
        f.write(struct.pack(f'{len(w)}f', *w))
        # print(f"  Exported weights for {name}: {len(w)} floats")

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            # Write Flattened Bias
            b = layer.bias.detach().cpu().numpy().flatten()
            f.write(struct.pack(f'{len(b)}f', *b))
            # print(f"  Exported bias for {name}: {len(b)} floats")
        else:
            # If bias is None (e.g. bias=False), write zeros
            # We need to know the output channels count.
            # layer.weight shape is [Out, In, H, W] (before permute)
            out_channels = layer.weight.shape[0]
            b = torch.zeros(out_channels).cpu().numpy().flatten()
            f.write(struct.pack(f'{len(b)}f', *b))
            # print(f"  Exported ZERO bias for {name}: {len(b)} floats")

def export_model(model, output_filename):
    print(f"Exporting to {output_filename}...")
    count = 0
    with open(output_filename, "wb") as f:
        # Iterate recursively to find all Conv2d layers in order
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                export_layer(f, module, name)
                count += 1
    print(f"Exported {count} Conv2d layers to {output_filename}.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export TAESD weights to binary format")
    parser.add_argument("--encoder", type=str, default="taesd_encoder.pth", help="Path to encoder pth")
    parser.add_argument("--decoder", type=str, default="taesd_decoder.pth", help="Path to decoder pth")
    parser.add_argument("--out-encoder", type=str, default="taesd_encoder.bin", help="Output encoder bin")
    parser.add_argument("--out-decoder", type=str, default="taesd_decoder.bin", help="Output decoder bin")

    args = parser.parse_args()

    # Load TAESD
    print(f"Loading TAESD from {args.encoder} and {args.decoder}...")

    # We load the full TAESD wrapper but we can also just instantiate Encoder/Decoder separately if we wanted.
    # The TAESD class handles loading state dicts.

    # Check if files exist
    enc_path = args.encoder if os.path.exists(args.encoder) else None
    dec_path = args.decoder if os.path.exists(args.decoder) else None

    if not enc_path and not dec_path:
        print("Warning: Neither encoder nor decoder pth found. Initializing with random weights.")

    taesd = TAESD(encoder_path=enc_path, decoder_path=dec_path)
    taesd.eval()

    if enc_path or (not enc_path and not dec_path):
        export_model(taesd.encoder, args.out_encoder)

    if dec_path or (not enc_path and not dec_path):
        export_model(taesd.decoder, args.out_decoder)

if __name__ == "__main__":
    main()
