"""
Complete Data Preprocessing Pipeline for Language Model Training.
1. Loads Wikipedia subset dataset (~540MB WikiText-103).
2. Performs GPT-2 style Byte-Pair Encoding (BPE) tokenization into uint16 token IDs.
3. Splits into 80% training and 20% validation sets.
4. Saves high-performance binary memory-mapped files.
"""

import os
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from gpt2_tokenizer import GPT2TokenizerWrapper
from config import DATA_DIR

def run_preprocessing_pipeline(max_tokens: int = 1_000_000, output_prefix: str = "wiki"):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("=" * 65)
    print(" DATA PREPROCESSING PIPELINE: WIKIPEDIA SUBSET -> GPT-2 TOKENS")
    print("=" * 65)
    
    # 1. Initialize GPT-2 Tokenizer
    print("\n[Step 1/4] Loading GPT-2 Tokenizer...")
    tokenizer = GPT2TokenizerWrapper()
    print(f" -> Vocabulary Size: {tokenizer.vocab_size:,} tokens")
    print(f" -> End-of-Text Token: {tokenizer.tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # 2. Load dataset
    print("\n[Step 2/4] Loading Wikipedia subset (WikiText-103)...")
    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    except Exception as e:
        print(f"Loading via 'Salesforce/wikitext' failed ({e}), trying fallback 'wikitext'...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    total_rows = len(dataset)
    print(f" -> Available raw rows in training corpus: {total_rows:,}")

    # 3. Tokenize text into uint16 array
    print(f"\n[Step 3/4] Tokenizing text using GPT-2 BPE (Target tokens: {max_tokens:,})...")
    token_accumulator = []
    total_chars = 0
    lines_processed = 0

    pbar = tqdm(total=max_tokens, unit="tokens", desc="Tokenizing")
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
            
        total_chars += len(text)
        lines_processed += 1
        
        # Tokenize with EOS separator
        tokens = tokenizer.encode(text, add_eos=True)
        token_accumulator.extend(tokens)
        pbar.update(len(tokens))
        
        if len(token_accumulator) >= max_tokens:
            token_accumulator = token_accumulator[:max_tokens]
            break
            
    pbar.close()

    total_tokens = len(token_accumulator)
    print(f"\nTokenization Summary:")
    print(f" -> Processed lines:        {lines_processed:,}")
    print(f" -> Total characters:       {total_chars:,}")
    print(f" -> Total tokens generated: {total_tokens:,}")
    print(f" -> Average chars/token:    {total_chars / max(1, total_tokens):.2f}")

    # 4. 80/20 Train/Validation Split & Binary Storage
    print("\n[Step 4/4] Splitting into 80/20 Train/Validation Sets...")
    split_idx = int(0.80 * total_tokens)
    
    tokens_np = np.array(token_accumulator, dtype=np.uint16)
    train_tokens = tokens_np[:split_idx]
    val_tokens = tokens_np[split_idx:]
    
    train_file = os.path.join(DATA_DIR, f"{output_prefix}_train_gpt2.bin")
    val_file = os.path.join(DATA_DIR, f"{output_prefix}_val_gpt2.bin")
    
    train_tokens.tofile(train_file)
    val_tokens.tofile(val_file)
    
    train_size_mb = os.path.getsize(train_file) / (1024 * 1024)
    val_size_mb = os.path.getsize(val_file) / (1024 * 1024)
    
    print(f" -> Training set (80%):   {len(train_tokens):,} tokens ({train_size_mb:.2f} MB) -> {train_file}")
    print(f" -> Validation set (20%): {len(val_tokens):,} tokens ({val_size_mb:.2f} MB) -> {val_file}")
    print("=" * 65)
    print(" PREPROCESSING COMPLETE!")
    print("=" * 65)
    
    return train_file, val_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Wikipedia with GPT-2 Tokenization")
    parser.add_argument("--max_tokens", type=int, default=1_000_000, help="Number of tokens to extract and tokenize")
    parser.add_argument("--output_prefix", type=str, default="wiki", help="Prefix for binary files")
    args = parser.parse_args()
    
    run_preprocessing_pipeline(max_tokens=args.max_tokens, output_prefix=args.output_prefix)
