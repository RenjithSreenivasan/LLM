"""
GPT-Style Transformer Language Model from Scratch (100M - 200M Parameters).
Features:
- 12 Decoder Layers, 768 Hidden Dimensions, 12 Attention Heads (GPT-2 Small architecture).
- Multi-Head Causal Self-Attention with Scaled Dot-Product Attention (FlashAttention compatible).
- Pre-Layer Normalization and MLP Feed-Forward with GELU activation.
- Gradient Checkpointing support for memory-efficient training.
- Mixed Precision (FP16/BF16) execution.
- AdamW Optimizer configuration with decoupled weight decay.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

@dataclass
class GPTConfig:
    vocab_size: int = 50257     # GPT-2 standard vocabulary size
    block_size: int = 1024      # Maximum sequence length (context window)
    n_layer: int = 12           # Number of transformer layers
    n_head: int = 12            # Number of attention heads (768 / 12 = 64 per head)
    n_embd: int = 768           # Embedding / Hidden dimension
    dropout: float = 0.1        # Dropout probability
    bias: bool = True           # Use bias in Linears and LayerNorms (like GPT-2)
    tie_weights: bool = True    # Tie token embeddings with LM Head
    gradient_checkpointing: bool = False  # Enable gradient checkpointing to save VRAM

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT and OpenAI GPT.
    Reference: Gaussian Error Linear Units (Hendrycks and Gimpel, 2016)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Key, Query, Value projections combined into a single matrix for high throughput
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer (fallback if scaled_dot_product_attention is not available)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension (n_embd)

        # Calculate Query, Key, Values for all heads in batch and move head forward to the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # Fast Scaled Dot-Product Attention (uses FlashAttention on NVIDIA GPUs automatically)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual fallback
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Feed-forward expansion to 4 * n_embd (3072 dims)
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm architecture with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (tying token embeddings to lm_head as in GPT-2)
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)
        # Apply special scaled initialization to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device) # (t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks with optional Gradient Checkpointing
        for block in self.transformer.h:
            if self.config.gradient_checkpointing and self.training:
                # Gradient checkpointing saves intermediate activations during forward pass
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # If targets are provided, calculate cross-entropy loss for next-token prediction
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            # Inference-time optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # (b, 1, vocab_size)
            loss = None

        return logits, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count, position embeddings get subtracted,
        and token embeddings are subtracted if they are not tied.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            if not self.config.tie_weights:
                n_params -= self.transformer.wte.weight.numel()
        return n_params

    def configure_optimizers(
        self,
        weight_decay: float = 0.1,
        learning_rate: float = 6e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda"
    ) -> torch.optim.AdamW:
        """
        Setup AdamW optimizer with decoupled weight decay:
        - 2D parameters (weights in Linear and Embedding) get weight decay.
        - 1D parameters (biases and LayerNorm scale/bias) DO NOT get weight decay.
        """
        # Separate parameters into decay and no_decay groups
        decay_params = []
        nodecay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"AdamW Configuration:")
        print(f" -> Decayed parameter tensors:     {len(decay_params)} ({num_decay:,} parameters)")
        print(f" -> Non-decayed parameter tensors: {len(nodecay_params)} ({num_nodecay:,} parameters)")

        # Use fused AdamW if supported and on CUDA
        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f" -> Using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50
    ) -> torch.Tensor:
        """
        Autoregressive next-token text generation.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if sequence length exceeds block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Scale logits by temperature
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
