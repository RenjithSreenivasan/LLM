"""
Ultra-Fast Local Chatbot Training Script.
Optimized specifically for low-spec PCs (Intel Core i3) to train in seconds without overheating.
Trains on conversational dialogue dataset (User / Assistant format).
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure scripts dir in sys.path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from config import DATA_DIR, MODELS_DIR
from gpt2_model import GPTConfig, GPT2LMHeadModel
from dataset import CausalLMDataset

# Ensure utf-8 console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def train_chatbot(
    epochs: int = 5,
    max_steps: int = 400,
    batch_size: int = 8,
    block_size: int = 64,
    learning_rate: float = 8e-4,
    output_checkpoint: str = os.path.join(MODELS_DIR, "chatbot_model.pt")
):
    print("=" * 75)
    print(" [FAST LOCAL CHATBOT TRAINING - INTEL i3 OPTIMIZED]")
    print("=" * 75)

    # 1. Device configuration (CPU optimized with thread limits to prevent overheating)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        amp_dtype = torch.float16
        print("Device: CUDA GPU (Accelerated)")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        amp_dtype = torch.bfloat16
        # Use 4 threads for fast execution
        torch.set_num_threads(4)
        print("Device: CPU (Intel Core i3 Optimized - Low Heat, 4 Threads)")

    # 2. Load preprocessed chatbot tokens
    train_bin = os.path.join(DATA_DIR, "chat_train.bin")
    val_bin = os.path.join(DATA_DIR, "chat_val.bin")

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print("Chatbot data not found. Running prepare_chatbot_data.py first...")
        from prepare_chatbot_data import prepare_chatbot_data
        prepare_chatbot_data()

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    train_dataset = CausalLMDataset(train_data, block_size=block_size)
    val_dataset = CausalLMDataset(val_data, block_size=block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Dataset Loaded:")
    print(f" -> Training Tokens:   {len(train_data):,} ({len(train_dataset):,} samples)")
    print(f" -> Validation Tokens: {len(val_data):,} ({len(val_dataset):,} samples)")
    print(f" -> Batch Size:        {batch_size} | Context: {block_size} tokens")

    # 3. Micro Chatbot Architecture (Compact, 30x faster than 124M model)
    config = GPTConfig(
        vocab_size=50257,
        block_size=block_size,
        n_layer=4,          # 4 layers (fast forward/backward)
        n_head=4,           # 4 attention heads
        n_embd=128,         # 128 hidden dim (ultra-lightweight)
        dropout=0.1,
        bias=True,
        tie_weights=True,
        gradient_checkpointing=False  # Not needed for small models, saves recomputation time!
    )

    model = GPT2LMHeadModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Initialized:")
    print(f" -> Parameters:        {total_params:,} (~{total_params / 1e6:.1f}M)")
    print(f" -> Architecture:      4 Layers, 128 Dims, 4 Heads")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 4. Fast Training Loop
    print("\n" + "=" * 75)
    print(f" TRAINING FOR {max_steps} STEPS...")
    print("=" * 75)

    train_losses = []
    val_losses = []
    steps = []

    model.train()
    start_time = time.time()
    step = 0
    best_loss = float("inf")

    pbar = tqdm(total=max_steps, desc="Training Chatbot", unit="step")

    while step < max_steps:
        for x, y in train_loader:
            if step >= max_steps:
                break

            x = x.to(device)
            y = y.to(device)

            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                _, loss = model(x, targets=y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            train_losses.append(loss.item())
            steps.append(step)

            # Fast validation evaluation every 50 steps
            if step % 50 == 0 or step == max_steps:
                model.eval()
                v_losses = []
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        with torch.autocast(device_type=device_type, dtype=amp_dtype):
                            _, v_loss = model(vx, targets=vy)
                        v_losses.append(v_loss.item())
                        if len(v_losses) >= 10:
                            break
                val_loss_val = float(np.mean(v_losses)) if v_losses else loss.item()
                val_losses.append((step, val_loss_val))
                model.train()

                pbar.set_postfix({"train_loss": f"{loss.item():.3f}", "val_loss": f"{val_loss_val:.3f}"})

                if val_loss_val < best_loss:
                    best_loss = val_loss_val

            pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_time
    steps_per_sec = step / max(1e-5, elapsed)

    # 5. Save the trained Chatbot Model Checkpoint
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
        "val_loss": best_loss,
        "train_loss": train_losses[-1] if train_losses else 0.0
    }, output_checkpoint)

    # Also save as latest_checkpoint.pt so the Desktop GUI picks it up immediately
    latest_path = os.path.join(MODELS_DIR, "latest_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": step,
        "val_loss": best_loss,
        "train_loss": train_losses[-1] if train_losses else 0.0
    }, latest_path)

    # 6. Save loss curve plot
    plot_path = os.path.join(MODELS_DIR, "chatbot_loss_curve.png")
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, train_losses, label="Train Loss", color="#2563EB", alpha=0.6)
    if val_losses:
        vx, vy = zip(*val_losses)
        plt.plot(vx, vy, label="Validation Loss", color="#DC2626", lw=2, marker="o")
    plt.title("Local Chatbot Training Loss (Intel i3 CPU)", fontsize=12, fontweight="bold")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print("\n" + "=" * 75)
    print(" TRAINING COMPLETE!")
    print(f" -> Duration:          {elapsed:.2f} seconds ({steps_per_sec:.1f} steps/second)")
    print(f" -> Final Train Loss:  {train_losses[-1]:.4f}")
    print(f" -> Best Val Loss:     {best_loss:.4f}")
    print(f" -> Saved Model to:    {output_checkpoint}")
    print(f" -> Saved Plot to:     {plot_path}")
    print("=" * 75)

    return model

if __name__ == "__main__":
    train_chatbot()
