import argparse
import requests
import io
import struct
import random
import os
import numpy as np
from PIL import Image

def download_image(url, size=(512, 512)):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')
        try:
            resample = Image.Resampling.BICUBIC
        except AttributeError:
            resample = Image.BICUBIC
        img = img.resize(size, resample)
        return img
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def save_batch(images, output_file):
    # Format: N H W C (float32)

    if not images:
        return

    try:
        # Convert list of PIL images to a single numpy array
        # PIL images are (W, H) -> np.array is (H, W, 3)
        # We want (N, H, W, 3)

        # Ensure all images are RGB and same size (already handled in download)
        data = np.stack([np.array(img, dtype=np.float32) for img in images])

        # Normalize to [0, 1]
        data /= 255.0

        # Ensure contiguous and correct type
        data = np.ascontiguousarray(data, dtype=np.float32)

        # Save raw binary
        data.tofile(output_file)

    except Exception as e:
        print(f"Error saving batch: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    images = []
    attempts = 0
    max_attempts = args.batch_size * 2

    while len(images) < args.batch_size and attempts < max_attempts:
        attempts += 1
        if args.validation:
            # Deterministic URLs for validation
            # Use picsum id
            # We need args.batch_size distinct images
            # Use seed + index
            img_id = args.seed + len(images)
            url = f"https://picsum.photos/id/{img_id}/512/512"
        else:
            # Random images
            # adding random query param to avoid cache
            url = f"https://picsum.photos/512/512?random={random.randint(0, 1000000)}"

        print(f"Downloading {url}...")
        img = download_image(url)
        if img:
            images.append(img)

    if len(images) < args.batch_size:
        print("Failed to download enough images")
        # Fill with black or duplicates to avoid crash?
        while len(images) < args.batch_size and len(images) > 0:
            images.append(images[0])

    if len(images) == 0:
        print("No images downloaded.")
        exit(1)

    print(f"Saving {len(images)} images to {args.output}...")
    save_batch(images, args.output)
    print("Done.")

if __name__ == "__main__":
    main()
