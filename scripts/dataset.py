"""
Dataset and DataLoader implementation for Causal Language Modeling.
Supports memory-mapped token files for memory-efficient training on CPU/GPU.
"""

import os
from typing import Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CausalLMDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction (Causal LM).
    Yields pairs (x, y) where:
      x = tokens[idx : idx + block_size]
      y = tokens[idx + 1 : idx + block_size + 1]  (shifted by 1 token)
    """
    def __init__(self, token_data: Union[np.ndarray, np.memmap, torch.Tensor], block_size: int = 128):
        self.block_size = block_size
        
        if isinstance(token_data, torch.Tensor):
            self.data = token_data
        elif isinstance(token_data, (np.ndarray, np.memmap)):
            # Keep as numpy or convert on the fly to avoid duplicating RAM
            self.data = token_data
        else:
            raise TypeError(f"Unsupported data type for token_data: {type(token_data)}")

        # Number of samples we can extract
        self.total_tokens = len(self.data)
        self.num_samples = max(0, (self.total_tokens - 1) // self.block_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        
        # Extract chunk of length block_size + 1
        chunk = self.data[start_idx : end_idx + 1]
        
        if isinstance(chunk, (np.ndarray, np.memmap)):
            chunk = torch.from_numpy(chunk.astype(np.int64))
        else:
            chunk = chunk.to(torch.long)
            
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def split_and_create_datasets(
    tokens: Union[np.ndarray, np.memmap, torch.Tensor],
    block_size: int = 128,
    train_ratio: float = 0.80
) -> Tuple[CausalLMDataset, CausalLMDataset]:
    """
    Split token stream into 80% train and 20% validation datasets.
    """
    total_tokens = len(tokens)
    split_point = int(train_ratio * total_tokens)
    
    # Ensure split aligns with block_size boundary for clean chunks
    split_point = (split_point // block_size) * block_size
    
    train_data = tokens[:split_point]
    val_data = tokens[split_point:]
    
    train_dataset = CausalLMDataset(train_data, block_size=block_size)
    val_dataset = CausalLMDataset(val_data, block_size=block_size)
    
    return train_dataset, val_dataset

def create_dataloaders(
    train_dataset: CausalLMDataset,
    val_dataset: CausalLMDataset,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders with batching.
    Train DataLoader has shuffle=True.
    Val DataLoader has shuffle=False.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
