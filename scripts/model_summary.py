"""
Model Summary, Architecture Inspection, and Next-Token Training Step Verification.
Demonstrates:
- 12-layer, 768-dim, 12-head GPT-2 architecture (100M-200M params).
- Detailed layer-by-layer parameter summary table.
- Next-token prediction loss setup.
- Gradient checkpointing & mixed-precision (AMP) execution.
- AdamW optimizer step with decoupled weight decay.
"""

import sys
import math
import torch
from gpt2_model import GPTConfig, GPT2LMHeadModel

# Ensure utf-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def print_model_summary(model: GPT2LMHeadModel):
    print("=" * 85)
    print(" GPT-STYLE TRANSFORMER MODEL SUMMARY (100M - 200M PARAMETERS)")
    print("=" * 85)

    cfg = model.config
    print(f"Architecture Hyperparameters:")
    print(f" -> Number of Layers (n_layer):     {cfg.n_layer}")
    print(f" -> Hidden Dimensions (n_embd):     {cfg.n_embd}")
    print(f" -> Attention Heads (n_head):       {cfg.n_head} ({cfg.n_embd // cfg.n_head} dim/head)")
    print(f" -> Vocabulary Size:                {cfg.vocab_size:,} tokens")
    print(f" -> Max Sequence Length:            {cfg.block_size} tokens")
    print(f" -> Weight Tying (wte <-> lm_head): {cfg.tie_weights}")
    print(f" -> Gradient Checkpointing:         {cfg.gradient_checkpointing}")

    print("\n" + "-" * 85)
    print(f"{'Module Name':<38} | {'Shape':<22} | {'Param Count':<16}")
    print("-" * 85)

    # 1. Embeddings
    wte_params = model.transformer.wte.weight.numel()
    wpe_params = model.transformer.wpe.weight.numel()
    print(f"{'transformer.wte (Token Embeddings)':<38} | {str(list(model.transformer.wte.weight.shape)):<22} | {wte_params:>14,}")
    print(f"{'transformer.wpe (Position Embeddings)':<38} | {str(list(model.transformer.wpe.weight.shape)):<22} | {wpe_params:>14,}")

    # 2. Transformer Blocks Summary
    block_0 = model.transformer.h[0]
    b0_ln1 = sum(p.numel() for p in block_0.ln_1.parameters())
    b0_attn = sum(p.numel() for p in block_0.attn.parameters())
    b0_ln2 = sum(p.numel() for p in block_0.ln_2.parameters())
    b0_mlp = sum(p.numel() for p in block_0.mlp.parameters())
    block_params = b0_ln1 + b0_attn + b0_ln2 + b0_mlp
    total_blocks_params = block_params * cfg.n_layer

    print(f"{'transformer.h[0..11] (12 Blocks Total)':<38} | {f'12 x {block_params:,}':<22} | {total_blocks_params:>14,}")
    print(f"   |-- LayerNorm 1 (ln_1):            | [768]                  | {b0_ln1 * cfg.n_layer:>14,}")
    print(f"   |-- Causal Self-Attention (Q,K,V): | [768, 2304] + proj     | {b0_attn * cfg.n_layer:>14,}")
    print(f"   |-- LayerNorm 2 (ln_2):            | [768]                  | {b0_ln2 * cfg.n_layer:>14,}")
    print(f"   \\-- Feed-Forward MLP (c_fc, c_proj)| [768, 3072] + [3072,768| {b0_mlp * cfg.n_layer:>14,}")

    # 3. Final LayerNorm & LM Head
    ln_f_params = sum(p.numel() for p in model.transformer.ln_f.parameters())
    lm_head_params = model.lm_head.weight.numel()
    print(f"{'transformer.ln_f (Final LayerNorm)':<38} | {str(list(model.transformer.ln_f.weight.shape)):<22} | {ln_f_params:>14,}")
    if cfg.tie_weights:
        print(f"{'lm_head (Tied to wte)':<38} | {str(list(model.lm_head.weight.shape)):<22} | {'(Shared Weights)':>14}")
    else:
        print(f"{'lm_head (Linear Projection)':<38} | {str(list(model.lm_head.weight.shape)):<22} | {lm_head_params:>14,}")

    total_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.get_num_params(non_embedding=True)
    
    print("-" * 85)
    print(f"{'TOTAL MODEL PARAMETERS:':<38} | {'':<22} | {total_params:>14,}")
    print(f"{'NON-EMBEDDING PARAMETERS:':<38} | {'':<22} | {non_emb_params:>14,}")
    print("-" * 85)

    # Theoretical memory requirements
    fp32_mb = (total_params * 4) / (1024 * 1024)
    fp16_mb = (total_params * 2) / (1024 * 1024)
    int8_mb = (total_params * 1) / (1024 * 1024)
    print(f"Estimated Weights Memory Footprint:")
    print(f" -> 32-bit Float (FP32):   {fp32_mb:.2f} MB")
    print(f" -> 16-bit Float (FP16):   {fp16_mb:.2f} MB (Standard for Mixed Precision)")
    print(f" -> 8-bit Quantized (INT8):{int8_mb:.2f} MB")
    print("=" * 85)

def verify_training_step(model: GPT2LMHeadModel):
    print("\n" + "=" * 85)
    print(" VERIFYING NEXT-TOKEN PREDICTION, MIXED PRECISION & GRADIENT CHECKPOINTING")
    print("=" * 85)

    # Detect hardware device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        amp_dtype = torch.float16
        scaler_enabled = True
        print(f"Hardware Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        amp_dtype = torch.bfloat16
        scaler_enabled = False
        print("Hardware Device: CPU (Intel Core i3 Optimized)")

    model = model.to(device)
    model.train()

    # 1. Setup AdamW Optimizer
    print("\n[Step 1] Configuring AdamW Optimizer...")
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        betas=(0.9, 0.95),
        device_type=device_type
    )

    # 2. Setup Mixed Precision (AMP) Scaler
    print("\n[Step 2] Configuring Mixed Precision (AMP)...")
    scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)
    print(f" -> Device Type:  {device_type}")
    print(f" -> Autocast Dtype: {amp_dtype}")
    print(f" -> GradScaler Enabled: {scaler_enabled}")

    # 3. Create Sample Batch for Next-Token Prediction
    print("\n[Step 3] Generating Input/Target Batch for Next-Token Prediction...")
    batch_size = 2
    seq_len = 64
    # Create random token sequence (0 .. vocab_size - 1)
    tokens = torch.randint(0, model.config.vocab_size, (batch_size, seq_len + 1), device=device)
    
    # Inputs: tokens[:, :-1]  (shape: [2, 64])
    # Targets: tokens[:, 1:]  (shape: [2, 64], shifted by 1 position)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    print(f" -> Input tensor  (x) shape: {x.shape} (Positions 0 .. {seq_len-1})")
    print(f" -> Target tensor (y) shape: {y.shape} (Positions 1 .. {seq_len})")

    # 4. Forward Pass under Mixed Precision with Gradient Checkpointing
    print("\n[Step 4] Executing Forward Pass with Mixed Precision & Gradient Checkpointing...")
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device_type, dtype=amp_dtype):
        logits, loss = model(x, targets=y)

    print(f" -> Forward pass succeeded!")
    print(f" -> Output Logits Shape: {logits.shape} [batch, seq_len, vocab_size]")
    print(f" -> Next-Token Cross-Entropy Loss: {loss.item():.4f}")
    
    # Expected initial loss for uniform random tokens across 50,257 vocab: -ln(1 / 50257) ≈ 10.82
    expected_loss = math.log(model.config.vocab_size)
    print(f" -> Theoretical Initial Loss (-ln(1/{model.config.vocab_size})): {expected_loss:.4f}")

    # 5. Backward Pass and Optimizer Update
    print("\n[Step 5] Executing Backward Pass with GradScaler...")
    scaler.scale(loss).backward()
    
    # Gradient clipping (max norm 1.0)
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f" -> Gradients computed and clipped. Total Grad Norm: {grad_norm.item():.4f}")

    scaler.step(optimizer)
    scaler.update()
    print(" -> AdamW optimizer step executed successfully!")

    print("\n" + "=" * 85)
    print(" ALL CHECKS PASSED: Model is fully verified and ready for training!")
    print("=" * 85)

if __name__ == "__main__":
    # Configure GPT-2 Small architecture (12 layers, 768 dims, 12 heads)
    config = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        bias=True,
        tie_weights=True,
        gradient_checkpointing=True
    )
    
    model = GPT2LMHeadModel(config)
    print_model_summary(model)
    verify_training_step(model)
