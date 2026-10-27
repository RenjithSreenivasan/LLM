"""
Training script for training a GPT Language Model from scratch.
Configured for Intel Core i3 with tqdm progress tracking and matplotlib loss curves.
"""

import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import (
    DEVICE,
    MODELS_DIR,
    DATA_DIR,
    VOCAB_SIZE,
    BLOCK_SIZE,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    DROPOUT,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_ITERS,
    EVAL_INTERVAL,
    EVAL_ITERS
)
from model import GPTLanguageModel

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out

def train():
    train_path = os.path.join(DATA_DIR, "train.pt")
    val_path = os.path.join(DATA_DIR, "val.pt")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Processed datasets not found. Please run scripts/prepare_data.py first.")

    train_data = torch.load(train_path, weights_only=True)
    val_data = torch.load(val_path, weights_only=True)
    
    print(f"Device: {DEVICE}")
    print(f"Active PyTorch Threads: {torch.get_num_threads()}")
    print(f"Train dataset tokens: {len(train_data)}")
    print(f"Validation dataset tokens: {len(val_data)}")

    # Initialize model
    model = GPTLanguageModel(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT
    ).to(DEVICE)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {param_count:,} parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    eval_steps = []

    pbar = tqdm(range(MAX_ITERS), desc="Training LM", unit="step")
    for step in pbar:
        # Periodic evaluation
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            eval_steps.append(step)
            pbar.set_postfix({"train_loss": f"{losses['train']:.4f}", "val_loss": f"{losses['val']:.4f}"})

        # Sample batch and train step
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save model checkpoint
    checkpoint_path = os.path.join(MODELS_DIR, "model_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
        }
    }, checkpoint_path)
    print(f"\nModel checkpoint saved to: {checkpoint_path}")

    # Plot loss curves using matplotlib
    plot_path = os.path.join(MODELS_DIR, "loss_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(eval_steps, train_losses, label="Train Loss", color="royalblue", lw=2)
    plt.plot(eval_steps, val_losses, label="Validation Loss", color="crimson", lw=2)
    plt.title("LLM Training Loss Curve (Intel i3 CPU)")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to: {plot_path}")

if __name__ == "__main__":
    train()
