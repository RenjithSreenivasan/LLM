"""
GPT-2 Style Tokenizer Module.
Implements Byte-Level Byte-Pair Encoding (BPE) tokenization for language modeling.
"""

from typing import List, Union
import numpy as np
import torch
from transformers import AutoTokenizer

class GPT2TokenizerWrapper:
    """
    Wrapper for GPT-2 Byte-Pair Encoding (BPE) tokenizer.
    Vocabulary Size: 50,257 tokens.
    Special Token: <|endoftext|> (ID: 50256)
    """
    def __init__(self, model_name: str = "gpt2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # In GPT-2, the end-of-text token is typically used as the padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """Convert string text into list of integer token IDs."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: Union[List[int], np.ndarray, torch.Tensor]) -> str:
        """Convert integer token IDs back to a readable string."""
        if isinstance(token_ids, (torch.Tensor, np.ndarray)):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def encode_batch(self, texts: List[str], max_length: int = 128, padding: bool = True):
        """Batch encode with truncation and optional padding."""
        return self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length" if padding else False,
            return_tensors="pt"
        )

    def inspect_tokens(self, text: str):
        """Prints a visual breakdown of how GPT-2 tokenizes text into subword units."""
        token_ids = self.encode(text, add_eos=False)
        tokens = [self.tokenizer.decode([t]) for t in token_ids]
        
        print("\n--- GPT-2 Token Breakdown ---")
        print(f"Original Text: '{text}'")
        print(f"Token Count:   {len(token_ids)}")
        print(f"{'Index':<6} | {'Token ID':<10} | {'Subword String'}")
        print("-" * 40)
        for i, (tid, subword) in enumerate(zip(token_ids, tokens)):
            # Replace whitespace or newlines with readable visual indicators
            display_subword = repr(subword)
            print(f"{i:<6} | {tid:<10} | {display_subword}")
        print("-" * 40)

if __name__ == "__main__":
    tok = GPT2TokenizerWrapper()
    print(f"Loaded GPT-2 Tokenizer with vocab size: {tok.vocab_size}")
    tok.inspect_tokens("Artificial Intelligence with PyTorch and Transformers!")
