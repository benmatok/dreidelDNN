import argparse
import struct
import torch
import lpips
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize LPIPS model
loss_fn = lpips.LPIPS(net='vgg').to(device)
loss_fn.eval()

def load_tensor(filepath, shape):
    # Load raw binary float32
    with open(filepath, 'rb') as f:
        data = f.read()

    # Convert to tensor
    # Assuming NHWC from C++
    N, H, W, C = shape

    # Check size matches
    expected_bytes = N * H * W * C * 4
    if len(data) != expected_bytes:
        raise ValueError(f"File size mismatch. Expected {expected_bytes}, got {len(data)}")

    # Unpack
    # Fast loading with torch/numpy if available, avoiding struct for speed if possible
    # We installed numpy in previous steps via pip
    import numpy as np
    arr = np.frombuffer(data, dtype=np.float32)
    t = torch.from_numpy(arr.copy())

    # Reshape NHWC -> NCHW for LPIPS
    t = t.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

    # LPIPS expects input in [-1, 1].
    # C++ data is likely [0, 1] (from picsum/loader).
    # Need to check normalization. ZenithNano typically outputs whatever range it learns.
    # If loader provides [0, 1], we map to [-1, 1].
    t = t * 2.0 - 1.0

    return t.to(device).requires_grad_(True)

def save_tensor_grad(tensor, filepath):
    # Gradient of Loss w.r.t Input (Output of C++ model)
    # Tensor is NCHW. Convert grad back to NHWC for C++.

    # tensor.grad is what we need.
    if tensor.grad is None:
        grad = torch.zeros_like(tensor)
    else:
        grad = tensor.grad

    # Undo normalization?
    # Loss = L(x*2-1). dLoss/dx = dLoss/d(mapped) * 2.
    grad = grad * 2.0

    # Permute NCHW -> NHWC
    grad = grad.permute(0, 2, 3, 1).contiguous()

    # Convert to bytes
    grad_np = grad.cpu().detach().numpy().astype(np.float32)

    with open(filepath, 'wb') as f:
        f.write(grad_np.tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='Prediction binary file')
    parser.add_argument('--target', type=str, required=True, help='Target binary file')
    parser.add_argument('--grad', type=str, required=True, help='Output gradient file')
    parser.add_argument('--shape', type=int, nargs=4, required=True, help='N H W C')
    args = parser.parse_args()

    pred = load_tensor(args.pred, args.shape)

    # Target doesn't need grad
    with torch.no_grad():
        target = load_tensor(args.target, args.shape).detach()

    # Compute Loss
    loss = loss_fn(pred, target)
    loss_val = loss.mean()

    # Backward to get gradients for 'pred'
    loss_val.backward()

    # Save gradients
    save_tensor_grad(pred, args.grad)

    # Print loss for C++ to capture? Or just print to stdout
    print(f"{loss_val.item()}")

if __name__ == "__main__":
    main()
