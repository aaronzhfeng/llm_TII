#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Training Script

Fine-tunes a pre-trained base model on instruction-following data.
Uses ChatML format with loss masking (only compute loss on assistant responses).

Key features:
- Loads from pre-trained checkpoint
- Supports both JSONL and binary SFT datasets
- Loss masking on instruction tokens
- Lower learning rate for fine-tuning
- Proper handling of variable-length sequences

Usage:
    # Single GPU
    python train_sft.py configs/sft_qwen3_1.8b.py

    # Multi-GPU with DDP
    torchrun --standalone --nproc_per_node=8 train_sft.py configs/sft_qwen3_1.8b.py
"""

import os
import sys
import time
import math
import json
import pickle
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "enhanced_training_system"))

# Import model architecture
try:
    from model_builder import ConfigurableGPT
    from model_config import ModelArchitectureConfig
    MODULAR_ARCH_AVAILABLE = True
except ImportError:
    MODULAR_ARCH_AVAILABLE = False
    print("WARNING: Modular architecture not available")

try:
    from training_logger import TrainingLogger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

# I/O
out_dir = 'out-sft'
checkpoint_path = None  # Path to pre-trained checkpoint (REQUIRED)
eval_interval = 500
log_interval = 1
eval_iters = 50
eval_only = False
always_save_checkpoint = True
keep_all_checkpoints = True

# Logging
save_log_to_json = True
log_save_interval = 100
gradient_log_interval = 50
wandb_log = False
wandb_project = 'sft'
wandb_run_name = 'sft-run'

# Data
sft_data_dir = None  # Path to prepared SFT data (REQUIRED)
data_format = 'jsonl'  # 'jsonl' or 'binary'

# Training hyperparameters (SFT-specific)
batch_size = 4
gradient_accumulation_steps = 8
block_size = 2048

# Optimizer (lower LR for fine-tuning)
learning_rate = 2e-5  # Much lower than pre-training
min_lr = 2e-6
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
max_iters = 3000
warmup_iters = 100
lr_decay_iters = 3000
decay_lr = True

# System
device = 'cuda'
dtype = 'bfloat16'
compile = True
use_zero1 = False
use_fsdp = False

# Regularization
dropout = 0.05  # Slight dropout for fine-tuning

# Padding
pad_token_id = 151643  # <|endoftext|> for Qwen3

# Ignore index for loss
ignore_index = -100


# =============================================================================
# SFT DATASET
# =============================================================================

class SFTDataset(Dataset):
    """
    Dataset for SFT training.
    Supports both JSONL and pre-tokenized binary formats.
    """
    
    def __init__(
        self,
        data_path: str,
        block_size: int,
        pad_token_id: int = 151643,
        data_format: str = 'jsonl'
    ):
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.data_format = data_format
        
        if data_format == 'jsonl':
            self._load_jsonl(data_path)
        elif data_format == 'binary':
            self._load_binary(data_path)
        else:
            raise ValueError(f"Unknown data format: {data_format}")
    
    def _load_jsonl(self, data_path: str):
        """Load JSONL format dataset."""
        self.examples = []
        
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    self.examples.append({
                        'input_ids': example['input_ids'],
                        'labels': example['labels']
                    })
        
        print(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def _load_binary(self, data_path: str):
        """Load pre-tokenized binary format."""
        # Load metadata
        data_dir = Path(data_path).parent
        meta_path = data_dir / "meta.json"
        
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            self.meta = {}
        
        # Memory-map the binary files
        self.input_ids = np.memmap(data_path, dtype=np.int32, mode='r')
        
        labels_path = data_path.replace('.bin', '_labels.bin')
        if os.path.exists(labels_path):
            self.labels = np.memmap(labels_path, dtype=np.int32, mode='r')
        else:
            self.labels = None
        
        # For binary format, we sample random windows
        self.num_samples = len(self.input_ids) - self.block_size
        
        print(f"Loaded binary dataset: {len(self.input_ids):,} tokens")
    
    def __len__(self):
        if self.data_format == 'jsonl':
            return len(self.examples)
        else:
            return self.num_samples
    
    def __getitem__(self, idx):
        if self.data_format == 'jsonl':
            return self._get_jsonl_item(idx)
        else:
            return self._get_binary_item(idx)
    
    def _get_jsonl_item(self, idx):
        """Get item from JSONL dataset."""
        example = self.examples[idx]
        input_ids = example['input_ids']
        labels = example['labels']
        
        # Truncate if needed
        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
            labels = labels[:self.block_size]
        
        # Pad if needed
        pad_len = self.block_size - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [self.ignore_index] * pad_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _get_binary_item(self, idx):
        """Get item from binary dataset."""
        input_ids = self.input_ids[idx:idx+self.block_size].astype(np.int64)
        
        if self.labels is not None:
            labels = self.labels[idx:idx+self.block_size].astype(np.int64)
        else:
            # If no separate labels, use input_ids (standard LM)
            labels = input_ids.copy()
        
        return {
            'input_ids': torch.from_numpy(input_ids),
            'labels': torch.from_numpy(labels)
        }
    
    @property
    def ignore_index(self):
        return -100


def sft_collate_fn(batch: List[Dict], pad_token_id: int = 151643) -> Dict[str, torch.Tensor]:
    """
    Collate function for SFT batches.
    Handles variable-length sequences with proper padding.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids != pad_token_id).long()
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, 
           learning_rate: float, min_lr: float) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load pre-trained checkpoint for fine-tuning."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    model_args = checkpoint.get('model_args', checkpoint.get('config', {}))
    
    # Create model
    if MODULAR_ARCH_AVAILABLE:
        arch_config = ModelArchitectureConfig.from_dict(model_args)
        model = ConfigurableGPT(arch_config)
    else:
        raise RuntimeError("Modular architecture required for SFT")
    
    # Load state dict
    state_dict = checkpoint['model']
    
    # Fix key prefixes if needed
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, model_args


def compute_sft_loss(
    model: nn.Module, 
    input_ids: torch.Tensor, 
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute SFT loss with proper masking.
    
    Loss is only computed on tokens where labels != ignore_index.
    This allows masking out instruction tokens.
    """
    # Pass a dummy target to get full sequence logits (not just last position)
    # The model computes logits for all positions only when targets are provided
    output = model(input_ids, targets=input_ids)
    
    # Handle model returning tuple (logits, loss) or just logits
    logits = output[0] if isinstance(output, tuple) else output
    
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Compute loss with ignore_index (our custom masking)
    loss = F.cross_entropy(
        shift_logits, 
        shift_labels, 
        ignore_index=ignore_index,
        reduction='mean'
    )
    
    return loss


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, ctx, eval_iters, device):
    """Estimate loss on train and val sets."""
    out = {}
    model.eval()
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for i, batch in enumerate(loader):
            if i >= eval_iters:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            with ctx:
                loss = compute_sft_loss(model, input_ids, labels)
            
            losses.append(loss.item())
        
        out[split] = np.mean(losses) if losses else float('inf')
    
    model.train()
    return out


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    # -----------------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------------
    
    # Load config from command line
    config_keys = [k for k, v in globals().items() 
                   if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
    
    # Execute config file if provided
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if os.path.exists(config_file):
            print(f"Loading config from: {config_file}")
            with open(config_file) as f:
                exec(f.read(), globals())
    
    # Parse command line overrides
    for arg in sys.argv[2:]:
        if '=' in arg:
            key, val = arg.lstrip('-').split('=', 1)
            if key in globals():
                try:
                    globals()[key] = eval(val)
                except:
                    globals()[key] = val
    
    # Validate required arguments
    if checkpoint_path is None:
        print("ERROR: checkpoint_path is required. Specify the pre-trained model checkpoint.")
        sys.exit(1)
    
    if sft_data_dir is None:
        print("ERROR: sft_data_dir is required. Prepare SFT data with prepare_sft.py first.")
        sys.exit(1)
    
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set seeds
    torch.manual_seed(1337 + seed_offset)
    torch.cuda.manual_seed(1337 + seed_offset)
    np.random.seed(1337 + seed_offset)
    
    # Device setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Create output directory
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    
    # -----------------------------------------------------------------------------
    # Data Loading
    # -----------------------------------------------------------------------------
    
    if master_process:
        print("\n" + "="*60)
        print("Loading SFT Dataset")
        print("="*60)
    
    # Determine data paths
    if data_format == 'jsonl':
        train_path = os.path.join(sft_data_dir, 'train.jsonl')
        val_path = os.path.join(sft_data_dir, 'val.jsonl')
    else:
        train_path = os.path.join(sft_data_dir, 'train.bin')
        val_path = os.path.join(sft_data_dir, 'val.bin')
    
    train_dataset = SFTDataset(train_path, block_size, pad_token_id, data_format)
    val_dataset = SFTDataset(val_path, block_size, pad_token_id, data_format)
    
    # Create dataloaders
    if ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: sft_collate_fn(b, pad_token_id),
    )
    
    if master_process:
        print(f"Train examples: {len(train_dataset):,}")
        print(f"Val examples: {len(val_dataset):,}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps * ddp_world_size}")
    
    # -----------------------------------------------------------------------------
    # Model Loading
    # -----------------------------------------------------------------------------
    
    if master_process:
        print("\n" + "="*60)
        print("Loading Pre-trained Model")
        print("="*60)
    
    model, model_args = load_checkpoint(checkpoint_path, device)
    
    # Apply dropout for fine-tuning
    if dropout > 0:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout
    
    model.to(device)
    
    # Compile model
    if compile:
        if master_process:
            print("Compiling model...")
        model = torch.compile(model)
    
    # Wrap in DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    
    # -----------------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------------
    
    # Separate weight decay for different param types
    decay_params = []
    nodecay_params = []
    
    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        fused=True if device_type == 'cuda' else False
    )
    
    if master_process:
        num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"\nOptimizer: AdamW")
        print(f"Trainable parameters: {num_params:,}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # -----------------------------------------------------------------------------
    # Training Logger
    # -----------------------------------------------------------------------------
    
    if LOGGER_AVAILABLE and master_process and save_log_to_json:
        logger = TrainingLogger(out_dir, model_args)
    else:
        logger = None
    
    # WandB
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=model_args)
    
    # -----------------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------------
    
    if master_process:
        print("\n" + "="*60)
        print("Starting SFT Training")
        print("="*60)
        print(f"Max iterations: {max_iters}")
        print(f"Warmup iterations: {warmup_iters}")
        print(f"Eval interval: {eval_interval}")
    
    iter_num = 0
    best_val_loss = float('inf')
    train_iter = iter(train_loader)
    
    t0 = time.time()
    
    while iter_num < max_iters:
        # Set learning rate
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model, train_loader, val_loader, ctx, eval_iters, device)
            
            print(f"\n{'─'*60}")
            print(f"Iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"{'─'*60}")
            
            if wandb_log:
                wandb.log({
                    'iter': iter_num,
                    'train/loss': losses['train'],
                    'val/loss': losses['val'],
                    'lr': lr,
                })
            
            if logger:
                logger.log_eval(iter_num, losses['train'], losses['val'])
            
            # Save checkpoint
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = min(best_val_loss, losses['val'])
                
                if keep_all_checkpoints:
                    ckpt_name = f'ckpt_{iter_num:06d}.pt'
                else:
                    ckpt_name = 'ckpt.pt'
                
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': {k: globals()[k] for k in config_keys if k in globals()},
                }
                
                torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
                print(f"Saved checkpoint: {ckpt_name}")
        
        if eval_only:
            break
        
        # Training step with gradient accumulation
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        loss_accum = 0.0
        for micro_step in range(gradient_accumulation_steps):
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                if ddp:
                    train_sampler.set_epoch(iter_num)
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
            with ctx:
                loss = compute_sft_loss(model, input_ids, labels)
                loss = loss / gradient_accumulation_steps
            
            loss_accum += loss.item()
            
            # Backward pass
            scaler.scale(loss).backward()
        
        # Gradient clipping
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0 and master_process:
            tokens_per_sec = batch_size * block_size * gradient_accumulation_steps / dt
            print(f"iter {iter_num:5d} | loss {loss_accum:.4f} | lr {lr:.2e} | {dt*1000:.1f}ms | {tokens_per_sec:.0f} tok/s")
            
            if wandb_log:
                wandb.log({
                    'iter': iter_num,
                    'train/loss_step': loss_accum,
                    'train/lr': lr,
                    'train/tokens_per_sec': tokens_per_sec,
                })
            
            # Log to JSON at same frequency as terminal (log_interval)
            if logger and iter_num % log_interval == 0:
                logger.log_iter(iter_num, loss_accum, dt * 1000, lr)
                # Auto-save periodically
                if iter_num % log_save_interval == 0:
                    logger.save()
        
        iter_num += 1
    
    # -----------------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------------
    
    if master_process:
        print("\n" + "="*60)
        print("SFT Training Complete!")
        print("="*60)
        print(f"Final val loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {out_dir}")
        
        if logger:
            logger.finalize()
            logger.save()  # Actually save the JSON file!
    
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

