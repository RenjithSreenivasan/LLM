"""
Preprocess conversational chatbot dataset for ultra-fast local training.
Converts dialogue pairs into GPT-2 token streams.
"""

import os
import sys
import numpy as np

# Ensure scripts dir in sys.path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from config import DATA_DIR
from gpt2_tokenizer import GPT2TokenizerWrapper

def prepare_chatbot_data(
    input_file: str = os.path.join(DATA_DIR, "chatbot_conversations.txt"),
    output_train: str = os.path.join(DATA_DIR, "chat_train.bin"),
    output_val: str = os.path.join(DATA_DIR, "chat_val.bin"),
    repeats: int = 40  # Repeat small conversation corpus to give enough training signal
):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Chat dataset not found at: {input_file}")

    print("=" * 65)
    print(" PREPARING CHATBOT CONVERSATIONAL DATASET")
    print("=" * 65)

    tokenizer = GPT2TokenizerWrapper()
    print(f"Loaded GPT-2 Tokenizer (Vocab: {tokenizer.vocab_size:,})")

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Split into dialogues
    dialogues = [d.strip() for d in content.split("<|endoftext|>") if d.strip()]
    print(f"Loaded {len(dialogues)} unique conversation pairs.")

    # Tokenize each dialogue
    all_tokens = []
    for _ in range(repeats):
        # Shuffle dialogues across repeats for variety
        import random
        random.shuffle(dialogues)
        for d in dialogues:
            toks = tokenizer.encode(d, add_eos=True)
            all_tokens.extend(toks)

    total_tokens = len(all_tokens)
    split_point = int(0.85 * total_tokens)

    tokens_np = np.array(all_tokens, dtype=np.uint16)
    train_tokens = tokens_np[:split_point]
    val_tokens = tokens_np[split_point:]

    train_tokens.tofile(output_train)
    val_tokens.tofile(output_val)

    print(f"\nPreprocessing Complete:")
    print(f" -> Total Chat Tokens: {total_tokens:,}")
    print(f" -> Training Tokens:   {len(train_tokens):,} -> {output_train}")
    print(f" -> Validation Tokens: {len(val_tokens):,} -> {output_val}")
    print("=" * 65)

if __name__ == "__main__":
    prepare_chatbot_data()
