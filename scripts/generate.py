"""
Flexible Text Generation / Inference script.
Supports both GPT-2 standard checkpoints (models/best_model.pt) and custom models.
"""

import os
import sys
import argparse
import torch

# Ensure utf-8 console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import DEVICE, MODELS_DIR
from gpt2_model import GPT2LMHeadModel, GPTConfig
from gpt2_tokenizer import GPT2TokenizerWrapper

def generate_text(
    prompt: str = "The history of",
    checkpoint_path: str = os.path.join(MODELS_DIR, "best_model.pt"),
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 40
):
    if not os.path.exists(checkpoint_path):
        # Fallback to latest_checkpoint.pt if best_model.pt is not found
        fallback = os.path.join(MODELS_DIR, "latest_checkpoint.pt")
        if os.path.exists(fallback):
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    cfg = checkpoint["config"]
    if isinstance(cfg, GPTConfig):
        config = cfg
    elif isinstance(cfg, dict):
        config = GPTConfig(**cfg)
    else:
        config = GPTConfig(
            vocab_size=cfg.vocab_size,
            block_size=cfg.block_size,
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_layer=cfg.n_layer,
            bias=getattr(cfg, "bias", True),
            tie_weights=getattr(cfg, "tie_weights", True)
        )

    model = GPT2LMHeadModel(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = GPT2TokenizerWrapper()
    prompt_ids = tokenizer.encode(prompt, add_eos=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    print("\n" + "=" * 65)
    print(f" GENERATING FROM PROMPT: '{prompt}'")
    print("=" * 65)
    out_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    generated_text = tokenizer.decode(out_idx[0])

    print("\n--- Model Output ---")
    print(generated_text)
    print("-" * 65 + "\n")
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Model Text Generation")
    parser.add_argument("--prompt", type=str, default="The history of", help="Input prompt")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(MODELS_DIR, "best_model.pt"), help="Path to checkpoint")
    parser.add_argument("--tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling cutoff")
    args = parser.parse_args()

    generate_text(
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_k=args.top_k
    )
