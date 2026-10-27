"""
Prepare training and validation data splits and convert text to token tensors.
"""

import os
import torch
from tokenizers import Tokenizer
from config import DATA_DIR, MODELS_DIR

def prepare_data():
    corpus_file = os.path.join(DATA_DIR, "input.txt")
    tokenizer_file = os.path.join(MODELS_DIR, "tokenizer.json")
    
    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Corpus file not found at: {corpus_file}")
    if not os.path.exists(tokenizer_file):
        raise FileNotFoundError(f"Tokenizer not found at: {tokenizer_file}. Run scripts/train_tokenizer.py first.")

    print(f"Loading tokenizer from {tokenizer_file}...")
    tokenizer = Tokenizer.from_file(tokenizer_file)
    
    print(f"Reading corpus from {corpus_file}...")
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    print("Encoding text into token IDs...")
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids
    print(f"Total tokens in dataset: {len(token_ids)}")
    
    # 90% train, 10% validation split
    n = int(0.9 * len(token_ids))
    train_tokens = torch.tensor(token_ids[:n], dtype=torch.long)
    val_tokens = torch.tensor(token_ids[n:], dtype=torch.long)
    
    train_path = os.path.join(DATA_DIR, "train.pt")
    val_path = os.path.join(DATA_DIR, "val.pt")
    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)
    
    print(f"Saved {len(train_tokens)} train tokens to: {train_path}")
    print(f"Saved {len(val_tokens)} val tokens to: {val_path}")

if __name__ == "__main__":
    prepare_data()
