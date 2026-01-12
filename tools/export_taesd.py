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

def main():
    encoder_path = None
    decoder_path = None

    # Simple argument parsing
    if len(sys.argv) > 1:
        decoder_path = sys.argv[1]

    print(f"Loading TAESD Decoder (weights: {decoder_path if decoder_path else 'Random Initialization'})...")

    # Initialize model. If paths are None, it initializes with random weights (standard PyTorch behavior)
    # We pass None for paths to skip loading if arguments are missing.
    model = TAESD(encoder_path=None, decoder_path=decoder_path).decoder
    model.eval()

    output_filename = "taesd_decoder.bin"
    print(f"Exporting to {output_filename}...")

    with open(output_filename, "wb") as f:
        # Iterate recursively to find all Conv2d layers in order
        # TAESD Decoder structure is sequential, so named_modules() recursion depth-first yields correct execution order.
        # But named_modules() returns (name, module). It includes container modules.
        # We only want to export leaf Conv2d layers.

        count = 0
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # print(f"Exporting Conv2d: {name} | Shape: {module.weight.shape}")
                export_layer(f, module, name)
                count += 1

        print(f"Exported {count} Conv2d layers.")

if __name__ == "__main__":
    main()
