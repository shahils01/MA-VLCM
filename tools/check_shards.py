import os
import io
import glob
import argparse
from PIL import Image
try:
    import webdataset as wds
except ImportError:
    print("Please install webdataset: pip install webdataset")
    exit(1)
try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    exit(1)
import traceback

def _custom_decoder(key, data):
    extension = key.split(".")[-1].lower()
    if extension in ["png", "jpg", "jpeg"]:
        if "overhead" in key or "image" in key:
            try:
                with io.BytesIO(data) as stream:
                    img = Image.open(stream)
                    img.load()
                    img = img.convert("RGB")
                    return img
            except Exception as e:
                return f"ERROR_DECODING: {e}"
        return data
    return data

def check_datasets(shards_pattern):
    if os.path.isdir(shards_pattern):
        expanded = sorted(glob.glob(os.path.join(shards_pattern, "**", "*.tar"), recursive=True))
    else:
        expanded = sorted(glob.glob(shards_pattern, recursive=True))
        
    if not expanded:
        print(f"No shards found matching: {shards_pattern}")
        return

    print(f"Found {len(expanded)} shards. Checking...")
    
    corrupted_shards = []
    suspicious_types = []
    
    for shard in tqdm(expanded, desc="Checking Shards"):
        # Wrap decoding and iteration in a try-except to catch tar read errors
        try:
            dataset = wds.WebDataset([shard], shardshuffle=False).decode(_custom_decoder)
            
            for sample in dataset:
                key = sample.get("__key__", "")
                
                # simulate image extraction from train.py
                if "image.png" in sample:
                    image = sample["image.png"]
                elif "overhead.png" in sample:
                    image = sample["overhead.png"]
                else:
                    continue
                
                # Image can be None if train.py _custom_decoder skips it, but here we return ERROR_DECODING
                if image is None:
                    continue
                
                if isinstance(image, str) and image.startswith("ERROR_DECODING:"):
                    corrupted_shards.append((shard, key, "decoding_error", image))
                elif not isinstance(image, Image.Image):
                    # We expect a PIL.Image.Image. If it's something else, it might cause the processor to fail!
                    suspicious_types.append((shard, key, type(image)))
                    
        except Exception as e:
            # If the entire tar file is corrupted/truncated
            corrupted_shards.append((shard, "unknown_key", "shard_error", str(e)))
            
    if corrupted_shards:
        print("\n=== CORRUPTED OR ERROR SHARDS ===")
        for c in corrupted_shards:
            print(f"Shard: {c[0]} | Key: {c[1]} | Type: {c[2]} | Error: {c[3][:100]}")
    else:
        print("\nNo decoding errors found! All shards seem to be readable.")
        
    if suspicious_types:
        print("\n=== SUSPICIOUS IMAGE TYPES (Not PIL Image) ===")
        types_count = {}
        for s in suspicious_types:
            t = str(s[2])
            types_count[t] = types_count.get(t, 0) + 1
            if types_count[t] < 5:
                # only print first few instances of each weird type
                print(f"Shard: {s[0]} | Key: {s[1]} | Type: {s[2]}")
        print("Type counts:", types_count)
    else:
        print("All parsed images successfully returned as PIL Images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check webdataset shards for corrupted images.")
    parser.add_argument("--shards", type=str, default="/scratch/aparame/Research/VLCM_Data_Collection/data_scratch", help="Path to shard directory or glob pattern")
    args = parser.parse_args()
    
    check_datasets(args.shards)
