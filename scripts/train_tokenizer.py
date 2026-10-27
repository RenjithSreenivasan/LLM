"""
Train a custom Byte-Pair Encoding (BPE) tokenizer using Hugging Face tokenizers.
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from config import DATA_DIR, MODELS_DIR, VOCAB_SIZE

def train_tokenizer(corpus_path: str, save_path: str, vocab_size: int = VOCAB_SIZE):
    print(f"Training BPE tokenizer on {corpus_path} with vocab size {vocab_size}...")
    
    # Initialize BPE model
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    # Configure trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True
    )
    
    # Train from files
    tokenizer.train(files=[corpus_path], trainer=trainer)
    
    # Save tokenizer
    tokenizer.save(save_path)
    print(f"Tokenizer saved successfully to: {save_path}")
    return tokenizer

if __name__ == "__main__":
    sample_text_path = os.path.join(DATA_DIR, "input.txt")
    
    # Create sample data if input.txt doesn't exist yet
    if not os.path.exists(sample_text_path):
        sample_text = (
            "Once upon a time, in a bright green meadow, there was a little robot named Pip. "
            "Pip loved learning language models. Every morning, Pip would read tiny stories "
            "and compute token probabilities. One day, Pip met a wise owl named Archimedes. "
            "'To build a great language model,' Archimedes said, 'you need clean data, "
            "a well-architected Transformer, and patience.' "
            "Pip worked with devotion, training attention heads step by step. "
            "Soon, Pip's model could generate sweet stories that made all the woodland animals smile.\n"
        ) * 100
        with open(sample_text_path, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"Created sample training corpus at {sample_text_path}")
        
    out_tokenizer_path = os.path.join(MODELS_DIR, "tokenizer.json")
    train_tokenizer(sample_text_path, out_tokenizer_path)
