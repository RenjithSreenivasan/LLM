"""
Production-grade Training Loop for Transformer Language Models.
Features:
- Batched forward, loss calculation, backward pass, and AdamW weight updates.
- GPU Efficiency Optimizations: Mixed Precision (AMP), Gradient Accumulation, TF32, Non-blocking transfers.
- Regular and Best-Model Checkpointing every few hundred steps.
- Validation Loss Tracking and Early Stopping / Loss Stabilization Detection.
- Real-time Visualizations: Dual-panel Matplotlib loss curves, TensorBoard logging, and tqdm progress bars.
- 2-3 Epoch training loop with Cosine Annealing Learning Rate Schedule.
"""

import os
import sys
import time
import math
import json
import argparse
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure utf-8 console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from gpt2_model import GPTConfig, GPT2LMHeadModel
from dataset import CausalLMDataset, create_dataloaders
from config import DATA_DIR, MODELS_DIR

# -----------------------------------------------------------------------------
# Learning Rate Schedule (Cosine Annealing with Warmup)
# -----------------------------------------------------------------------------
def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    # 1) Linear warmup for warmup_steps
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    # 2) If step > max_steps, return min_lr
    if step > max_steps:
        return min_lr
    # 3) In between, use cosine decay down to min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# -----------------------------------------------------------------------------
# Validation Loss Evaluator
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_validation_loss(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    device_type: str,
    amp_dtype: torch.dtype,
    max_eval_batches: int = 25
) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_eval_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0

# -----------------------------------------------------------------------------
# Matplotlib Loss Curve Plotter
# -----------------------------------------------------------------------------
def save_loss_plot(history: Dict[str, list], output_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Plot 1: Train & Validation Losses
    if history["train_steps"]:
        ax1.plot(history["train_steps"], history["train_losses"], label="Train Loss (smoothed)", color="#2563EB", lw=2, alpha=0.85)
    if history["val_steps"]:
        ax1.plot(history["val_steps"], history["val_losses"], label="Validation Loss", color="#DC2626", lw=2.5, marker="o", markersize=4)

    ax1.set_title("Transformer Language Model Training Progress", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # Plot 2: Learning Rate Schedule
    if history["lr_steps"]:
        ax2.plot(history["lr_steps"], history["learning_rates"], label="Learning Rate", color="#059669", lw=2)
    ax2.set_xlabel("Optimization Step", fontsize=11)
    ax2.set_ylabel("Learning Rate", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Main Training Loop Function
# -----------------------------------------------------------------------------
def train_model(
    model_preset: str = "full",
    epochs: int = 3,
    max_steps: int = None,
    resume_from: str = None,
    batch_size: int = 4,
    block_size: int = 64,
    grad_accum_steps: int = 4,
    learning_rate: float = 5e-4,
    min_lr: float = 5e-5,
    warmup_steps: int = 50,
    checkpoint_interval: int = 100,
    eval_interval: int = 25,
    eval_batches: int = 10,
    early_stop_patience: int = 5,
    early_stop_min_delta: float = 0.005,
    checkpoint_dir: str = MODELS_DIR,
    log_dir: str = os.path.join(MODELS_DIR, "tensorboard_logs")
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 80)
    print(" STARTING TRANSFORMER TRAINING LOOP")
    print("=" * 80)

    # 1. Device Setup & GPU Efficiency Optimizations
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        amp_dtype = torch.float16
        scaler_enabled = True
        pin_memory = True
        # Enable TF32 for Tensor Cores on Ampere/Ada/Hopper GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Device: CUDA GPU -> {torch.cuda.get_device_name(0)}")
        print(" -> TF32 Enabled: True | Mixed Precision: FP16 | Pin Memory: True")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
        amp_dtype = torch.bfloat16
        scaler_enabled = False
        pin_memory = False
        torch.set_num_threads(4)
        print("Device: CPU (Intel Core i3 Optimized - 4 Threads)")
        print(" -> Mixed Precision: BF16 | Pin Memory: False")

    # 2. Dataset Loading (Memory-mapped uint16)
    train_bin = os.path.join(DATA_DIR, "wiki_train_gpt2.bin")
    val_bin = os.path.join(DATA_DIR, "wiki_val_gpt2.bin")

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        raise FileNotFoundError(f"Binary datasets not found in {DATA_DIR}. Run scripts/preprocess_pipeline.py first.")

    train_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin, dtype=np.uint16, mode="r")

    train_dataset = CausalLMDataset(train_data, block_size=block_size)
    val_dataset = CausalLMDataset(val_data, block_size=block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=0
    )

    steps_per_epoch = len(train_loader) // grad_accum_steps
    if max_steps is not None:
        total_steps = max_steps
        warmup_steps = min(warmup_steps, max(1, total_steps // 10))
    else:
        total_steps = epochs * steps_per_epoch

    print(f"\nTraining Dataset:")
    print(f" -> Train tokens:       {len(train_data):,} ({len(train_dataset):,} samples)")
    print(f" -> Validation tokens:  {len(val_data):,} ({len(val_dataset):,} samples)")
    print(f" -> Micro-batch size:   {batch_size}")
    print(f" -> Grad Accum steps:   {grad_accum_steps} (Effective batch size = {batch_size * grad_accum_steps})")
    print(f" -> Sequence length:    {block_size} tokens")
    print(f" -> Epochs:             {epochs}")
    print(f" -> Steps per epoch:    {steps_per_epoch}")
    print(f" -> Total opt steps:    {total_steps}")

    # 3. Model Architecture Selection
    if model_preset == "full":
        # 124M Parameter Model (12 layers, 768 dims, 12 heads)
        config = GPTConfig(
            vocab_size=50257,
            block_size=max(1024, block_size),
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.1,
            bias=True,
            tie_weights=True,
            gradient_checkpointing=True
        )
    else:
        # Scaled Model (4 layers, 256 dims, 4 heads ~3.5M params, for quick CPU testing)
        config = GPTConfig(
            vocab_size=50257,
            block_size=max(256, block_size),
            n_layer=4,
            n_head=4,
            n_embd=256,
            dropout=0.1,
            bias=True,
            tie_weights=True,
            gradient_checkpointing=True
        )

    model = GPT2LMHeadModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Initialized ({model_preset.upper()} preset):")
    print(f" -> Total Parameters:   {total_params:,}")
    print(f" -> Gradient Checkpointing: {config.gradient_checkpointing}")

    # 4. Optimizer & Scaler Setup
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type
    )
    scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)

    step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from checkpoint: {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Note: Reinitializing optimizer state: {e}")
        step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f" -> Resumed successfully from step {step} (previous val loss: {best_val_loss:.4f})")

    # 5. TensorBoard Setup
    tb_writer = SummaryWriter(log_dir=log_dir) if TENSORBOARD_AVAILABLE else None
    if tb_writer:
        print(f" -> TensorBoard logging initialized at: {log_dir}")

    # 6. Training State Tracking
    history = {
        "train_steps": [],
        "train_losses": [],
        "val_steps": [],
        "val_losses": [],
        "lr_steps": [],
        "learning_rates": []
    }
    start_time = time.time()
    plot_path = os.path.join(checkpoint_dir, "training_loss_curve.png")
    history_json_path = os.path.join(checkpoint_dir, "training_history.json")

    print("\n" + "=" * 80)
    print(" COMMENCING EPOCHS")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_tokens = 0
        running_loss = 0.0
        accum_loss = 0.0

        pbar = tqdm(
            total=steps_per_epoch,
            desc=f"Epoch {epoch}/{epochs}",
            unit="step",
            leave=True
        )

        for micro_step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            epoch_tokens += x.numel()

            # Forward pass under mixed precision
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                _, loss = model(x, targets=y)
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps

            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            accum_loss += loss.item() * grad_accum_steps

            # Perform optimizer step once grad_accum_steps are reached
            if (micro_step + 1) % grad_accum_steps == 0:
                step += 1

                # Update learning rate according to cosine schedule
                current_lr = get_lr(step, warmup_steps, total_steps, learning_rate, min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step & scaler update
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                running_loss = accum_loss / grad_accum_steps
                accum_loss = 0.0

                # Record metrics
                history["train_steps"].append(step)
                history["train_losses"].append(running_loss)
                history["lr_steps"].append(step)
                history["learning_rates"].append(current_lr)

                if tb_writer:
                    tb_writer.add_scalar("Loss/Train", running_loss, step)
                    tb_writer.add_scalar("LearningRate", current_lr, step)
                    tb_writer.add_scalar("GradNorm", grad_norm.item(), step)

                # Periodic Validation Evaluation
                current_val_loss = None
                if step % eval_interval == 0 or step == total_steps:
                    val_loss = evaluate_validation_loss(
                        model, val_loader, device, device_type, amp_dtype, max_eval_batches=eval_batches
                    )
                    current_val_loss = val_loss
                    history["val_steps"].append(step)
                    history["val_losses"].append(val_loss)

                    if tb_writer:
                        tb_writer.add_scalar("Loss/Validation", val_loss, step)

                    # Update real-time loss curve plot
                    save_loss_plot(history, plot_path)

                    # Check for improvement & save best checkpoint
                    if val_loss < (best_val_loss - early_stop_min_delta):
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
                        torch.save({
                            "epoch": epoch,
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config,
                            "val_loss": val_loss,
                            "train_loss": running_loss
                        }, best_ckpt_path)
                    else:
                        patience_counter += 1

                # Periodic Regular Checkpoint
                if step % checkpoint_interval == 0 or step == total_steps:
                    step_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                    latest_ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
                    ckpt_payload = {
                        "epoch": epoch,
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "train_loss": running_loss,
                        "val_loss": current_val_loss or (history["val_losses"][-1] if history["val_losses"] else 0.0)
                    }
                    torch.save(ckpt_payload, step_ckpt_path)
                    torch.save(ckpt_payload, latest_ckpt_path)

                # Update progress bar
                elapsed_sec = max(1e-5, time.time() - start_time)
                tokens_per_sec = (step * batch_size * grad_accum_steps * block_size) / elapsed_sec
                pbar.set_postfix({
                    "train_loss": f"{running_loss:.3f}",
                    "val_loss": f"{history['val_losses'][-1]:.3f}" if history["val_losses"] else "N/A",
                    "lr": f"{current_lr:.1e}",
                    "tok/s": f"{tokens_per_sec:,.0f}"
                })
                pbar.update(1)

                # Check if total steps reached
                if step >= total_steps:
                    print(f"\n[INFO] Target total steps ({total_steps}) reached.")
                    break

                # Loss Stabilization / Early Stopping Check
                if patience_counter >= early_stop_patience:
                    print(f"\n[INFO] Loss stabilized: No validation improvement for {early_stop_patience} evaluations.")
                    print(f" -> Best Validation Loss achieved: {best_val_loss:.4f}")
                    break

            if micro_step + 1 >= steps_per_epoch * grad_accum_steps or step >= total_steps:
                break

        pbar.close()

        # Save metrics history to JSON
        with open(history_json_path, "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= early_stop_patience or step >= total_steps:
            if step >= total_steps:
                print(f"Completed target steps ({total_steps}) at epoch {epoch}.")
            else:
                print(f"Early stopping epoch loop at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    save_loss_plot(history, plot_path)
    if tb_writer:
        tb_writer.close()

    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE!")
    print(f" -> Total Duration:      {total_time:.2f} seconds ({total_time / 60:.2f} mins)")
    print(f" -> Total Opt Steps:     {step}")
    print(f" -> Best Val Loss:       {best_val_loss:.4f}")
    print(f" -> Loss Plot Saved to:  {plot_path}")
    print(f" -> Checkpoints in:      {checkpoint_dir}")
    print("=" * 80)
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Model Training Loop")
    parser.add_argument("--model_preset", type=str, default="small", choices=["small", "full"],
                        help="'full' for 124M GPT-2; 'small' for lightweight fast CPU run")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional maximum optimization steps to train")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro-batch size")
    parser.add_argument("--block_size", type=int, default=64, help="Sequence length (tokens)")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Steps between checkpoints")
    parser.add_argument("--eval_interval", type=int, default=25, help="Steps between validation evaluations")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience evaluations")
    args = parser.parse_args()

    train_model(
        model_preset=args.model_preset,
        epochs=args.epochs,
        max_steps=args.max_steps,
        resume_from=args.resume,
        batch_size=args.batch_size,
        block_size=args.block_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        early_stop_patience=args.patience
    )
