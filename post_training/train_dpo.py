#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) Training Script

Aligns a SFT model with human preferences using DPO.
Learns from preference pairs (chosen vs rejected responses) without a reward model.

Reference:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    Rafailov et al., 2023 - https://arxiv.org/abs/2305.18290

Key features:
- No reward model needed (unlike RLHF)
- Directly optimizes policy from preference pairs
- Reference model (frozen SFT model) for KL constraint
- Lower memory than RLHF

Usage:
    # Single GPU
    python train_dpo.py configs/dpo_qwen3_1.8b.py

    # Multi-GPU with DDP
    torchrun --standalone --nproc_per_node=8 train_dpo.py configs/dpo_qwen3_1.8b.py
"""

import os
import sys
import time
import math
import json
import copy
from pathlib import Path
from contextlib import nullcontext
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
out_dir = 'out-dpo'
sft_checkpoint_path = None  # Path to SFT checkpoint (REQUIRED)
eval_interval = 200
log_interval = 1
eval_iters = 50
eval_only = False
always_save_checkpoint = True
keep_all_checkpoints = True

# Logging
save_log_to_json = True
log_save_interval = 50
wandb_log = False
wandb_project = 'dpo'
wandb_run_name = 'dpo-run'

# Data
dpo_data_dir = None  # Path to prepared DPO data (REQUIRED)
data_format = 'jsonl'

# Training hyperparameters
batch_size = 2  # Smaller due to computing both chosen and rejected
gradient_accumulation_steps = 8
block_size = 2048

# DPO-specific parameters
beta = 0.1  # KL penalty coefficient (higher = more conservative)
label_smoothing = 0.0  # Optional label smoothing
reference_free = False  # If True, skip reference model (not recommended)

# Optimizer
learning_rate = 5e-7  # Very low for DPO (even lower than SFT)
min_lr = 1e-7
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
max_iters = 1000
warmup_iters = 50
lr_decay_iters = 1000
decay_lr = True

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False  # Compilation with two models can be memory-intensive

# Padding
pad_token_id = 151643
ignore_index = -100


# =============================================================================
# DPO DATASET
# =============================================================================

class DPODataset(Dataset):
    """
    Dataset for DPO training.
    Each example contains: prompt, chosen response, rejected response.
    
    Expected JSONL format:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }
    
    Or with full conversation:
    {
        "prompt_ids": [...],
        "chosen_ids": [...],
        "rejected_ids": [...],
        "chosen_labels": [...],
        "rejected_labels": [...]
    }
    """
    
    def __init__(
        self,
        data_path: str,
        block_size: int,
        pad_token_id: int = 151643,
    ):
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.examples = []
        
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        
        print(f"Loaded {len(self.examples)} DPO examples from {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Get pre-tokenized IDs
        chosen_ids = example['chosen_ids']
        rejected_ids = example['rejected_ids']
        chosen_labels = example.get('chosen_labels', chosen_ids)
        rejected_labels = example.get('rejected_labels', rejected_ids)
        
        # Truncate and pad
        def process_sequence(ids, labels):
            if len(ids) > self.block_size:
                ids = ids[:self.block_size]
                labels = labels[:self.block_size]
            
            pad_len = self.block_size - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                labels = labels + [self.ignore_index] * pad_len
            
            return ids, labels
        
        chosen_ids, chosen_labels = process_sequence(chosen_ids, chosen_labels)
        rejected_ids, rejected_labels = process_sequence(rejected_ids, rejected_labels)
        
        return {
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long),
        }
    
    @property
    def ignore_index(self):
        return -100


def dpo_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DPO batches."""
    return {
        'chosen_ids': torch.stack([item['chosen_ids'] for item in batch]),
        'chosen_labels': torch.stack([item['chosen_labels'] for item in batch]),
        'rejected_ids': torch.stack([item['rejected_ids'] for item in batch]),
        'rejected_labels': torch.stack([item['rejected_labels'] for item in batch]),
    }


# =============================================================================
# DPO LOSS COMPUTATION
# =============================================================================

def get_batch_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = False
) -> torch.Tensor:
    """
    Compute log probabilities for a batch of sequences.
    
    Args:
        model: Language model
        input_ids: Input token IDs [batch_size, seq_len]
        labels: Target labels [batch_size, seq_len] (-100 for masked tokens)
        average_log_prob: If True, average over sequence length
    
    Returns:
        Log probabilities [batch_size]
    """
    logits = model(input_ids)
    
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute per-token log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for actual tokens
    # [batch_size, seq_len-1]
    per_token_logps = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask out padding and instruction tokens
    loss_mask = (shift_labels != -100).float()
    
    # Sum log probs over sequence
    seq_logps = (per_token_logps * loss_mask).sum(dim=-1)
    
    if average_log_prob:
        # Average over non-masked tokens
        seq_logps = seq_logps / loss_mask.sum(dim=-1).clamp(min=1)
    
    return seq_logps


def compute_dpo_loss(
    policy_model: nn.Module,
    reference_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute DPO loss.
    
    DPO Loss:
        L_DPO = -log σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
    
    Where:
        - π is the policy model
        - π_ref is the reference (frozen) model
        - y_w is the chosen (winning) response
        - y_l is the rejected (losing) response
        - β is the KL penalty coefficient
    
    Args:
        policy_model: Model being trained
        reference_model: Frozen reference model (SFT checkpoint)
        batch: Dictionary with chosen_ids, chosen_labels, rejected_ids, rejected_labels
        beta: KL penalty coefficient
        label_smoothing: Optional label smoothing
        reference_free: If True, skip reference model computation
    
    Returns:
        loss: Scalar loss tensor
        metrics: Dictionary with additional metrics
    """
    chosen_ids = batch['chosen_ids']
    chosen_labels = batch['chosen_labels']
    rejected_ids = batch['rejected_ids']
    rejected_labels = batch['rejected_labels']
    
    # Compute policy log probs
    policy_chosen_logps = get_batch_logps(policy_model, chosen_ids, chosen_labels)
    policy_rejected_logps = get_batch_logps(policy_model, rejected_ids, rejected_labels)
    
    # Compute reference log probs (with no gradient)
    if not reference_free:
        with torch.no_grad():
            ref_chosen_logps = get_batch_logps(reference_model, chosen_ids, chosen_labels)
            ref_rejected_logps = get_batch_logps(reference_model, rejected_ids, rejected_labels)
    else:
        ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
        ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
    
    # Compute log ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    # DPO logits (what we pass to sigmoid/loss)
    logits = beta * (pi_logratios - ref_logratios)
    
    # DPO loss with optional label smoothing
    if label_smoothing > 0:
        # Soft labels
        losses = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        # Standard DPO loss
        losses = -F.logsigmoid(logits)
    
    loss = losses.mean()
    
    # Compute metrics
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
    
    metrics = {
        'loss': loss.item(),
        'reward_margin': reward_margin.item(),
        'reward_accuracy': reward_accuracy.item(),
        'chosen_reward': chosen_rewards.mean().item(),
        'rejected_reward': rejected_rewards.mean().item(),
        'logits_mean': logits.mean().item(),
        'logits_std': logits.std().item(),
    }
    
    return loss, metrics


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
    """Load checkpoint for fine-tuning."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_args = checkpoint.get('model_args', checkpoint.get('config', {}))
    
    if MODULAR_ARCH_AVAILABLE:
        arch_config = ModelArchitectureConfig.from_dict(model_args)
        model = ConfigurableGPT(arch_config)
    else:
        raise RuntimeError("Modular architecture required for DPO")
    
    state_dict = checkpoint['model']
    
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, model_args


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    # Load config
    config_keys = [k for k, v in globals().items()
                   if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if os.path.exists(config_file):
            print(f"Loading config from: {config_file}")
            with open(config_file) as f:
                exec(f.read(), globals())
    
    for arg in sys.argv[2:]:
        if '=' in arg:
            key, val = arg.lstrip('-').split('=', 1)
            if key in globals():
                try:
                    globals()[key] = eval(val)
                except:
                    globals()[key] = val
    
    # Validate
    if sft_checkpoint_path is None:
        print("ERROR: sft_checkpoint_path is required.")
        sys.exit(1)
    
    if dpo_data_dir is None:
        print("ERROR: dpo_data_dir is required.")
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
    
    torch.manual_seed(1337 + seed_offset)
    torch.cuda.manual_seed(1337 + seed_offset)
    np.random.seed(1337 + seed_offset)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    
    if master_process:
        print("\n" + "="*60)
        print("Loading DPO Dataset")
        print("="*60)
    
    train_path = os.path.join(dpo_data_dir, 'train.jsonl')
    val_path = os.path.join(dpo_data_dir, 'val.jsonl')
    
    train_dataset = DPODataset(train_path, block_size, pad_token_id)
    val_dataset = DPODataset(val_path, block_size, pad_token_id)
    
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
        collate_fn=dpo_collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=dpo_collate_fn,
    )
    
    if master_process:
        print(f"Train examples: {len(train_dataset):,}")
        print(f"Val examples: {len(val_dataset):,}")
    
    # -------------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------------
    
    if master_process:
        print("\n" + "="*60)
        print("Loading Models")
        print("="*60)
    
    # Load policy model (will be trained)
    policy_model, model_args = load_checkpoint(sft_checkpoint_path, device)
    policy_model.to(device)
    
    # Load reference model (frozen copy of SFT model)
    if not reference_free:
        if master_process:
            print("Loading reference model (frozen)...")
        reference_model, _ = load_checkpoint(sft_checkpoint_path, device)
        reference_model.to(device)
        reference_model.eval()
        
        # Freeze reference model
        for param in reference_model.parameters():
            param.requires_grad = False
    else:
        reference_model = None
    
    # Compile policy model (reference doesn't need compilation)
    if compile:
        if master_process:
            print("Compiling policy model...")
        policy_model = torch.compile(policy_model)
    
    # DDP
    if ddp:
        policy_model = DDP(policy_model, device_ids=[ddp_local_rank])
    
    raw_policy = policy_model.module if ddp else policy_model
    
    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    
    decay_params = []
    nodecay_params = []
    
    for name, param in raw_policy.named_parameters():
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
        print(f"\nDPO Parameters:")
        print(f"  beta (KL penalty): {beta}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Reference free: {reference_free}")
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # WandB
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=model_args)
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    
    if master_process:
        print("\n" + "="*60)
        print("Starting DPO Training")
        print("="*60)
    
    iter_num = 0
    best_val_loss = float('inf')
    train_iter = iter(train_loader)
    
    t0 = time.time()
    
    while iter_num < max_iters:
        # Learning rate
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation
        if iter_num % eval_interval == 0 and master_process:
            policy_model.eval()
            
            val_metrics = {'loss': [], 'reward_accuracy': [], 'reward_margin': []}
            
            for i, batch in enumerate(val_loader):
                if i >= eval_iters:
                    break
                
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with ctx:
                    with torch.no_grad():
                        _, metrics = compute_dpo_loss(
                            raw_policy, reference_model, batch,
                            beta=beta, reference_free=reference_free
                        )
                
                for k in val_metrics:
                    val_metrics[k].append(metrics[k])
            
            val_loss = np.mean(val_metrics['loss'])
            val_acc = np.mean(val_metrics['reward_accuracy'])
            val_margin = np.mean(val_metrics['reward_margin'])
            
            print(f"\n{'─'*60}")
            print(f"Iter {iter_num}: val_loss {val_loss:.4f} | acc {val_acc:.3f} | margin {val_margin:.3f}")
            print(f"{'─'*60}")
            
            if wandb_log:
                wandb.log({
                    'iter': iter_num,
                    'val/loss': val_loss,
                    'val/reward_accuracy': val_acc,
                    'val/reward_margin': val_margin,
                })
            
            # Save checkpoint
            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = min(best_val_loss, val_loss)
                
                if keep_all_checkpoints:
                    ckpt_name = f'ckpt_{iter_num:06d}.pt'
                else:
                    ckpt_name = 'ckpt.pt'
                
                checkpoint = {
                    'model': raw_policy.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                
                torch.save(checkpoint, os.path.join(out_dir, ckpt_name))
                print(f"Saved checkpoint: {ckpt_name}")
            
            policy_model.train()
        
        if eval_only:
            break
        
        # Training step
        optimizer.zero_grad(set_to_none=True)
        
        loss_accum = 0.0
        metrics_accum = {}
        
        for micro_step in range(gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                if ddp:
                    train_sampler.set_epoch(iter_num)
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if ddp:
                policy_model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
            with ctx:
                loss, metrics = compute_dpo_loss(
                    raw_policy, reference_model, batch,
                    beta=beta, label_smoothing=label_smoothing,
                    reference_free=reference_free
                )
                loss = loss / gradient_accumulation_steps
            
            loss_accum += loss.item()
            
            for k, v in metrics.items():
                if k not in metrics_accum:
                    metrics_accum[k] = []
                metrics_accum[k].append(v)
            
            scaler.scale(loss).backward()
        
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0 and master_process:
            avg_acc = np.mean(metrics_accum.get('reward_accuracy', [0]))
            avg_margin = np.mean(metrics_accum.get('reward_margin', [0]))
            
            print(f"iter {iter_num:5d} | loss {loss_accum:.4f} | acc {avg_acc:.3f} | margin {avg_margin:.3f} | lr {lr:.2e} | {dt*1000:.1f}ms")
            
            if wandb_log:
                wandb.log({
                    'iter': iter_num,
                    'train/loss': loss_accum,
                    'train/reward_accuracy': avg_acc,
                    'train/reward_margin': avg_margin,
                    'train/lr': lr,
                })
        
        iter_num += 1
    
    # Cleanup
    if master_process:
        print("\n" + "="*60)
        print("DPO Training Complete!")
        print("="*60)
        print(f"Checkpoints saved to: {out_dir}")
    
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()

