import os
import time
import requests
import numpy as np
from PIL import Image
import io
import argparse
import random
import shutil

POOL_DIR = "dataset/pool"
MAX_POOL_SIZE = 50
BATCH_LIMIT = 50
TIME_WINDOW = 90.0

def download_image(url, size=(2048, 2048)):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')
        # Resize if significantly larger/smaller, but picsum usually gives requested size
        # We requested 2048x2048
        return img
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def save_image(img, index):
    timestamp = int(time.time() * 1000)
    filename = f"atlas_{timestamp}_{index}.bin"
    filepath = os.path.join(POOL_DIR, filename)
    temp_filepath = filepath + ".part"

    # Convert to float32 HWC
    arr = np.array(img, dtype=np.float32) / 255.0

    # Save as raw binary to temp file
    arr.tofile(temp_filepath)

    # Atomic rename
    os.rename(temp_filepath, filepath)
    return filepath

def maintain_pool():
    if not os.path.exists(POOL_DIR):
        os.makedirs(POOL_DIR)

    download_count = 0
    window_start = time.time()

    print(f"Grazer Daemon started. Pool: {POOL_DIR}")

    while True:
        # Check pool size
        # Only count .bin files, ignore .part
        files = [f for f in os.listdir(POOL_DIR) if f.endswith('.bin')]
        current_size = len(files)

        if current_size < MAX_POOL_SIZE:
            needed = MAX_POOL_SIZE - current_size
            print(f"Pool size: {current_size}. Grazing {needed} more...")

            for i in range(needed):
                # Rate limit check
                now = time.time()
                if now - window_start > TIME_WINDOW:
                    # Reset window
                    window_start = now
                    download_count = 0

                if download_count >= BATCH_LIMIT:
                    sleep_time = TIME_WINDOW - (now - window_start) + 1
                    if sleep_time > 0:
                        print(f"Rate limit reached. Sleeping {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                        window_start = time.time()
                        download_count = 0

                # Download
                # Use random ID to get variety
                url = f"https://picsum.photos/2048/2048?random={random.randint(0, 1000000)}"
                img = download_image(url)
                if img:
                    save_image(img, i)
                    download_count += 1
                    print(f"Grazed {i+1}/{needed} (Total in window: {download_count})")

                # Small sleep to be polite even within limit
                time.sleep(0.5)
        else:
            # Pool full, sleep
            time.sleep(5)

if __name__ == "__main__":
    maintain_pool()
