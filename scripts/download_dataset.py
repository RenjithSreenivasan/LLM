"""
Download and extract a Wikipedia subset dataset (~540MB uncompressed).
Uses the standard WikiText-103-raw benchmark dataset.
"""

import os
from datasets import load_dataset
from tqdm import tqdm

def download_wikipedia_subset(output_dir: str = "data") -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "wikipedia_raw.txt")
    
    print("=" * 60)
    print("Downloading / Loading WikiText-103 (Wikipedia subset ~540MB)...")
    print("=" * 60)
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    print(f"Dataset loaded:")
    print(f" - Train examples:      {len(dataset['train']):,}")
    print(f" - Validation examples: {len(dataset['validation']):,}")
    print(f" - Test examples:       {len(dataset['test']):,}")
    
    if not os.path.exists(output_file) or os.path.getsize(output_file) < 100_000_000:
        print(f"\nWriting raw Wikipedia text to {output_file}...")
        total_bytes = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for split_name in ["train", "validation", "test"]:
                split_ds = dataset[split_name]
                for item in tqdm(split_ds, desc=f"Exporting {split_name} split"):
                    line = item["text"]
                    if line.strip():  # Skip completely empty lines
                        f.write(line.strip() + "\n")
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n[SUCCESS] Raw Wikipedia corpus saved to: {output_file}")
        print(f"[SUCCESS] Total corpus size: {file_size_mb:.2f} MB")
    else:
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\nCorpus already exists at: {output_file} ({file_size_mb:.2f} MB)")
        
    return output_file

if __name__ == "__main__":
    download_wikipedia_subset()
