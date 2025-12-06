import os
import requests
from io import BytesIO
from urllib.parse import urlparse
from PIL import Image
from datasets import load_dataset  # pip install datasets pillow

# 1. Load your dataset from Hugging Face
# Replace this with your actual dataset name and split
dataset = load_dataset("TheFusion21/PokemonCards", split="train")

# 2. Output folder
out_dir = "pokemon_images"
os.makedirs(out_dir, exist_ok=True)

for i, row in enumerate(dataset):
    url = row["image_url"]  # column in the HF dataset
    if url is None or url == "":
        continue

    try:
        # 3. Parse URL to get "hgss4" and "1_hires.png"
        parsed = urlparse(url)
        # path: "/hgss4/1_hires.png" -> "hgss4/1_hires.png"
        path = parsed.path.lstrip("/")
        set_id, filename = path.split("/", 1)  # "hgss4", "1_hires.png"

        base, _ext = os.path.splitext(filename)  # "1_hires", ".png"
        new_name = f"{set_id}_{base}.jpg"        # "hgss4_1_hires.jpg"
        out_path = os.path.join(out_dir, new_name)

        # Skip if already downloaded
        if os.path.exists(out_path):
            continue

        # 4. Download the image
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        # 5. Convert to JPEG using Pillow
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(out_path, format="JPEG", quality=95)

        if i % 100 == 0:
            print(f"Saved {i} images, last: {out_path}")

    except Exception as e:
        print(f"Failed on {url}: {e}")
