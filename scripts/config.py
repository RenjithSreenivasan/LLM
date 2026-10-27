"""
Central configuration for training a Language Model from scratch.
Configured with optimal defaults for Intel Core i3 (2 cores / 4 threads).
"""

import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

# Device Configuration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    # Optimize CPU threading for Intel Core i3 (4 logical threads)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

# Model Hyperparameters (Sized for smooth training on CPU)
VOCAB_SIZE = 1000      # Compact vocabulary for fast training
BLOCK_SIZE = 128       # Context window length (tokens)
N_EMBD = 128           # Embedding dimension
N_HEAD = 4             # Attention heads (128 / 4 = 32 dim per head)
N_LAYER = 4            # Number of Transformer decoder layers
DROPOUT = 0.1          # Dropout rate

# Training Hyperparameters
BATCH_SIZE = 16        # Micro-batch size
LEARNING_RATE = 5e-4   # AdamW learning rate
MAX_ITERS = 500        # Number of training steps
EVAL_INTERVAL = 50     # How often to evaluate and log loss
EVAL_ITERS = 20        # Number of batches for validation loss
