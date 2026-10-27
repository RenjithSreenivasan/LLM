"""
Advanced Language Model Inference Engine.
Features:
- Checkpoint loading with automatic architecture reconstruction.
- Temperature scaling (including greedy decoding when temperature=0).
- Top-K sampling and Top-P (Nucleus) sampling.
- Repetition penalty support.
- Real-time token-by-token streaming generator.
- CLI and GUI importable interface.
"""

import os
import sys
import time
import argparse
from typing import Generator, Tuple, Optional, List

import torch
import torch.nn.functional as F

# Ensure utf-8 console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import DEVICE, MODELS_DIR
from gpt2_model import GPTConfig, GPT2LMHeadModel
from gpt2_tokenizer import GPT2TokenizerWrapper

class ModelInferenceEngine:
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.tokenizer = GPT2TokenizerWrapper()
        self.model: Optional[GPT2LMHeadModel] = None
        self.config: Optional[GPTConfig] = None
        self.current_checkpoint: Optional[str] = None

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        cfg = checkpoint["config"]
        if isinstance(cfg, GPTConfig):
            self.config = cfg
        elif isinstance(cfg, dict):
            self.config = GPTConfig(**cfg)
        else:
            self.config = GPTConfig(
                vocab_size=getattr(cfg, "vocab_size", 50257),
                block_size=getattr(cfg, "block_size", 1024),
                n_layer=getattr(cfg, "n_layer", 12),
                n_head=getattr(cfg, "n_head", 12),
                n_embd=getattr(cfg, "n_embd", 768),
                bias=getattr(cfg, "bias", True),
                tie_weights=getattr(cfg, "tie_weights", True)
            )

        self.model = GPT2LMHeadModel(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.current_checkpoint = checkpoint_path
        print(f"Model loaded successfully! ({self.model.get_num_params(non_embedding=False):,} parameters)")

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_on_eos: bool = True
    ) -> Generator[Tuple[str, int, float, float], None, None]:
        """
        Yields token-by-token:
            (token_text, token_id, elapsed_seconds, tokens_per_second)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_checkpoint() first.")

        prompt_ids = self.tokenizer.encode(prompt, add_eos=False)
        if not prompt_ids:
            prompt_ids = [self.tokenizer.eos_token_id]

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_token_ids: List[int] = []

        start_time = time.time()
        block_size = self.config.block_size

        for step in range(max_new_tokens):
            # Crop context if sequence length exceeds block size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

            # Forward pass
            logits, _ = self.model(idx_cond)
            next_token_logits = logits[0, -1, :].clone()

            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated_token_ids) > 0:
                for prev_token in set(generated_token_ids):
                    if next_token_logits[prev_token] < 0:
                        next_token_logits[prev_token] *= repetition_penalty
                    else:
                        next_token_logits[prev_token] /= repetition_penalty

            # Temperature Scaling & Greedy Sampling
            if temperature <= 0.01:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature

                # Top-K Truncation
                if top_k > 0:
                    top_k_val = min(top_k, next_token_logits.size(-1))
                    val, _ = torch.topk(next_token_logits, top_k_val)
                    next_token_logits[next_token_logits < val[-1]] = -float("Inf")

                # Top-P (Nucleus) Truncation
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Filter out tokens with cumulative probability exceeding threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift mask right to include the first token above threshold
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float("Inf")

                # Multinomial Sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            token_val = next_token_id.item()
            generated_token_ids.append(token_val)

            # Append token to running context
            idx = torch.cat((idx, next_token_id.unsqueeze(0)), dim=1)

            # Decode individual token
            token_text = self.tokenizer.decode([token_val])
            elapsed = time.time() - start_time
            tok_per_sec = (step + 1) / max(1e-5, elapsed)

            yield (token_text, token_val, elapsed, tok_per_sec)

            # Check for EOS
            if stop_on_eos and token_val == self.tokenizer.eos_token_id:
                break

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> str:
        """Non-streaming generation returning complete text."""
        tokens = []
        for text_chunk, _, _, _ in self.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        ):
            tokens.append(text_chunk)
        return "".join(tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Model Inference CLI")
    parser.add_argument("--prompt", type=str, default="Wikipedia is", help="Prompt text")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(MODELS_DIR, "best_model.pt"), help="Checkpoint file")
    parser.add_argument("--tokens", type=int, default=50, help="Max new tokens")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (0.0 = greedy)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K cutoff (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus threshold (1.0 = disabled)")
    parser.add_argument("--rep_penalty", type=float, default=1.1, help="Repetition penalty")
    args = parser.parse_args()

    engine = ModelInferenceEngine(checkpoint_path=args.checkpoint)
    print(f"\nPrompt: {args.prompt}")
    print("Generating (streaming): ", end="", flush=True)

    for chunk, tid, elap, speed in engine.stream_generate(
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty
    ):
        print(chunk, end="", flush=True)

    print(f"\n\n[Done] Speed: {speed:.1f} tokens/sec ({elap:.2f}s)")
