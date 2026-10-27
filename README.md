# Language Model from Scratch (PyTorch & Transformer)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RenjithSreenivasan/LLM/blob/main/train_on_colab.ipynb)

A complete project for building, training, and running a Transformer Language Model from scratch using PyTorch, Hugging Face `tokenizers`, `datasets`, and `transformers`. Optimized for both local CPU execution and **Free Cloud GPU Training (Google Colab / Kaggle)**.

---

## 1. Project Structure

```text
LLM/
├── .venv/                   # Python 3.13 Virtual Environment
├── data/                    # Datasets and processed token tensors
│   ├── input.txt            # Training corpus text (character/small sample)
│   ├── wiki_train_gpt2.bin  # 80% Training tokens (GPT-2 BPE, uint16)
│   ├── wiki_val_gpt2.bin    # 20% Validation tokens (GPT-2 BPE, uint16)
│   ├── train.pt             # Tokenized training tensor
│   └── val.pt               # Tokenized validation tensor
├── models/                  # Checkpoints, tokenizers, and plots
│   ├── tokenizer.json       # Custom BPE tokenizer
│   ├── model_checkpoint.pt  # PyTorch model weights and config
│   └── loss_curve.png       # Matplotlib training loss plot
├── scripts/                 # Source code
│   ├── download_dataset.py  # Download WikiText-103 Wikipedia subset (~540MB)
│   ├── gpt2_tokenizer.py    # GPT-2 BPE Tokenizer wrapper (50,257 vocab)
│   ├── dataset.py           # PyTorch CausalLMDataset & DataLoader with 80/20 split
│   ├── preprocess_pipeline.py # Full Wikipedia to GPT-2 binary token pipeline
│   ├── verify_pipeline.py   # Batching, decoding, & target shift verification
│   ├── gpt2_model.py        # 124M-parameter GPT-2 Architecture (12L, 768D, 12H)
│   ├── model_summary.py     # Parameter breakdown, AMP & gradient checkpointing check
│   ├── train_loop.py        # Production-grade Training Loop with Checkpointing & Plots
│   ├── inference.py         # Streaming inference engine with Top-K/Top-P/Temp controls
│   ├── gui.py               # Modern Python Desktop GUI for real-time testing
│   ├── config.py            # Hardware & hyperparameter configurations
│   ├── model.py             # Lightweight CPU educational Transformer
│   ├── train_tokenizer.py   # BPE Tokenizer training
│   ├── prepare_data.py      # Dataset encoding and tensor splitting
│   ├── train.py             # Model training loop with tqdm & loss plotting
│   ├── generate.py          # Inference / text generation
│   └── verify_environment.py# Environment and hardware diagnostics
├── train_on_colab.ipynb     # 1-Click Free Cloud GPU Training Notebook
├── launch_gui.bat           # 1-click Windows Batch Desktop GUI launcher
├── launch_gui.ps1           # 1-click PowerShell Desktop GUI launcher
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 2. Hardware & CUDA Analysis

- **Processor**: 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (2 Cores, 4 Logical Threads).
- **GPU**: Intel(R) UHD Graphics (Integrated).
- **CUDA Detection (`torch.cuda.is_available()`)**: `False`.
  > **Why?** CUDA is an NVIDIA-proprietary architecture requiring physical NVIDIA GPU hardware. PyTorch automatically uses your Intel i3 CPU with Intel MKL multi-threading acceleration.
- **CPU Optimization Configured**:
  In [`scripts/config.py`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/scripts/config.py):
  ```python
  torch.set_num_threads(4)
  torch.set_num_interop_threads(2)
  ```

---

## 3. Data Preprocessing Pipeline (GPT-2 Tokenization & WikiText-103)

### Features:
1. **Wikipedia Subset Dataset**: Uses **WikiText-103** (~540MB uncompressed raw text, 1.8M articles/paragraphs).
2. **GPT-2 BPE Tokenization**: Standard Byte-Level Byte-Pair Encoding with 50,257 vocabulary size and `<|endoftext|>` separator.
3. **Memory-Mapped Binary Output**: Serialized as `np.uint16` (2 bytes per token instead of 8 bytes in Python int, saving 75% RAM on your i3 laptop).
4. **80/20 Split & PyTorch DataLoaders**: Splits continuous token streams into 80% train and 20% validation, yielding `(x, y)` batches shifted by 1 token for autoregressive causal language modeling.

### Run Preprocessing:
```powershell
# Tokenize and create 80/20 train/val splits
.\.venv\Scripts\python scripts/preprocess_pipeline.py --max_tokens 1000000
```

### Verify DataLoader & Batch Inspection:
```powershell
.\.venv\Scripts\python scripts/verify_pipeline.py
```

---

## 4. GPT-Style Transformer Model Architecture (124M Parameters)

Implemented in [`scripts/gpt2_model.py`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/scripts/gpt2_model.py) and inspected via [`scripts/model_summary.py`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/scripts/model_summary.py):

- **Parameters**: **124,439,808** (124.4M with weight tying, or 163.0M without).
- **Layers**: 12 Transformer Blocks.
- **Hidden Dimensions**: 768.
- **Attention Heads**: 12 (64 dimensions per head).
- **Vocabulary**: 50,257 tokens (GPT-2 standard).
- **Max Sequence Length**: 1,024 tokens.
- **Gradient Checkpointing**: Reduces activation memory during training by recomputing block activations on backward pass.
- **Mixed Precision**: Autocast (FP16 on GPU / BF16 on CPU) + GradScaler for fast, numerically stable training.
- **AdamW Optimizer**: Decoupled weight decay (0.1 on 2D weights; 0.0 on 1D biases and layer norms).

### Inspect Model Summary & Verify Training Step:
```powershell
.\.venv\Scripts\python scripts/model_summary.py
```

---

## 5. Complete Training Loop (Batching, GPU Efficiency, Validation, & Checkpointing)

Implemented in [`scripts/train_loop.py`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/scripts/train_loop.py):

- **Batch Optimization & Updates**: Micro-batching with gradient accumulation, cosine annealing learning rate with warmup, and gradient norm clipping ($1.0$).
- **GPU Efficiency Features**:
  - TensorFloat-32 (TF32) matrix multiplications enabled on NVIDIA GPUs.
  - Mixed Precision (`torch.autocast`) with `torch.amp.GradScaler`.
  - Non-blocking host-to-device memory transfers (`pin_memory=True`, `.to(device, non_blocking=True)`).
  - Gradient Checkpointing minimizing activation memory.
- **Validation & Early Stopping**: Evaluates validation loss across held-out splits every $N$ steps. Automatically triggers early stopping when validation loss stabilizes.
- **Checkpointing**:
  - Regular checkpoints saved every few hundred steps (`checkpoint_step_X.pt`).
  - Best model checkpoint saved whenever validation loss hits a new minimum (`best_model.pt`).
  - Latest checkpoint saved for resuming (`latest_checkpoint.pt`).
- **Visualizations**:
  - Live progress bar (`tqdm`) tracking throughput (tokens/sec) and loss.
  - Dual-panel Matplotlib plot saved to [`models/training_loss_curve.png`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/models/training_loss_curve.png) displaying smoothed train loss, validation loss, and learning rate schedule.
  - TensorBoard event logging in [`models/tensorboard_logs/`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/models/tensorboard_logs).

### Run Training:
```powershell
# Train model for 3 epochs (or specify max_steps):
.\.venv\Scripts\python scripts/train_loop.py --model_preset small --epochs 3 --eval_interval 25 --checkpoint_interval 100

# View real-time TensorBoard dashboard:
.\.venv\Scripts\tensorboard --logdir models/tensorboard_logs
```

### Run Generation from Best Checkpoint:
```powershell
.\.venv\Scripts\python scripts/generate.py --prompt "Wikipedia is an online" --tokens 50
```

---

## 6. Python Desktop GUI & Real-Time Streaming Playground

Implemented in [`scripts/gui.py`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/scripts/gui.py) and [`scripts/inference.py`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/scripts/inference.py):

- **Modern Clean Slate Theme**: Styled using native Tkinter with dark palette and responsive layout.
- **Model Checkpoint Selector**: Automatically discovers all `.pt` checkpoints in `models/` (`best_model.pt`, `latest_checkpoint.pt`, etc.) with file browsing and instant live reloading.
- **Adjustable Generation Controls**:
  - **Temperature Slider** ($0.0 \dots 2.0$): From greedy deterministic decoding to highly creative generation.
  - **Top-K Slider** ($0 \dots 100$): Limits candidate token pool to top $K$ probabilities.
  - **Top-P (Nucleus) Slider** ($0.1 \dots 1.0$): Dynamic cumulative probability thresholding.
  - **Max Tokens Slider** ($10 \dots 300$): Controls generation length.
  - **Repetition Penalty Slider** ($1.0 \dots 2.0$): Penalizes repetitive word patterns.
- **Real-Time Token-by-Token Streaming Display**:
  - Tokens stream into the output viewer asynchronously as they are sampled.
  - Non-blocking background worker thread with `[⏹ Stop Generation]` cancellation control.
  - Real-time telemetry: Live token counter, generation speed (tokens/sec), and elapsed time.
  - One-click copy output to clipboard and preset prompt shortcuts.

### How to Launch the Desktop GUI:
```powershell
# Method 1: Using PowerShell
.\launch_gui.ps1

# Method 2: Using Windows Command Prompt or Double-Click
.\launch_gui.bat

# Method 3: Using Python
.\.venv\Scripts\python scripts/gui.py
```

### CLI Streaming Inference:
```powershell
.\.venv\Scripts\python scripts/inference.py --prompt "The history of" --tokens 50 --temp 0.8 --top_k 40 --top_p 0.9
```

---

## 7. How to Activate the Environment

Open PowerShell in this project folder (`c:\Users\renji\OneDrive\Desktop\Project\LLM`):

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 8. Installed Libraries

The following libraries are installed and verified:
- **PyTorch** (`torch` 2.7.1+cpu)
- **Hugging Face Transformers** (`transformers` 4.54.0)
- **Hugging Face Datasets** (`datasets` 3.6.0)
- **Hugging Face Tokenizers** (`tokenizers` 0.21.2)
- **TensorBoard** (`tensorboard` 2.21.0)
- **NumPy** (`numpy` 2.2.6)
- **Matplotlib** (`matplotlib` 3.11.2)
- **tqdm** (`tqdm` 4.67.1)

---

## 9. Workflow Guide

### Step 1: Run Diagnostics & Hardware Verification
```powershell
.\.venv\Scripts\python scripts/verify_environment.py
```

### Step 2: Train the Tokenizer
Trains a Byte-Pair Encoding (BPE) tokenizer on [`data/input.txt`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/data/input.txt) and outputs [`models/tokenizer.json`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/models/tokenizer.json):
```powershell
.\.venv\Scripts\python scripts/train_tokenizer.py
```

### Step 3: Prepare the Dataset
Encodes text into token ID tensors and saves [`data/train.pt`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/data/train.pt) and [`data/val.pt`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/data/val.pt):
```powershell
.\.venv\Scripts\python scripts/prepare_data.py
```

### Step 4: Train the Language Model
Trains the Transformer with live `tqdm` progress, periodic loss logging, and saves the model to [`models/model_checkpoint.pt`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/models/model_checkpoint.pt) and plot to [`models/loss_curve.png`](file:///c:/Users/renji/OneDrive/Desktop/Project/LLM/models/loss_curve.png):
```powershell
.\.venv\Scripts\python scripts/train.py
```

### Step 5: Generate Text
Run autoregressive text generation with your trained model:
```powershell
.\.venv\Scripts\python scripts/generate.py "Once upon a time"
```
