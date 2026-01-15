import argparse
import requests
import io
import struct
import random
import os
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
    # Just write raw bytes
    # Normalize to 0-1
    with open(output_file, 'wb') as f:
        for img in images:
            # Get data as float list or bytes
            # PIL to bytes is uint8
            # Convert to float
            pixels = list(img.getdata()) # List of (r,g,b)
            # Flatten and normalize
            # Efficient way:
            # We can use struct to pack, but that's slow for 512x512x3 (~786k floats)
            # Better to use array or just bytes if C++ handles it?
            # C++ expects float.
            # Let's use bytearray and struct.pack_into or just write raw bytes manually?
            # Actually, standard way is to write binary.
            # Python float is double, struct 'f' is float.

            # Optimization: create a bytearray
            # But let's keep it simple first.
            # 512*512*3 = 786432 floats. 3MB.

            # To make it faster, use a simple loop or list comp
            # floats = [val / 255.0 for pixel in pixels for val in pixel]
            # buff = struct.pack(f'{len(floats)}f', *floats)
            # f.write(buff)

            # Even faster:
            # Use int.to_bytes? No, we need floats.
            # If we didn't have numpy... which we don't (I didn't install it).
            # Okay, let's just do the list comp. It might be slow.
            # 786k items.
            pass

            # Re-implementation without numpy:
            # Write raw uint8 and let C++ convert?
            # No, C++ expects float.
            # Let's write the float conversion here.

            data = []
            for pixel in pixels:
                data.append(pixel[0] / 255.0)
                data.append(pixel[1] / 255.0)
                data.append(pixel[2] / 255.0)

            # Pack is slow for large lists.
            # struct.pack('f'*len(data), *data)
            # Try chunking.

            chunk_size = 1024
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                f.write(struct.pack(f'{len(chunk)}f', *chunk))

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
