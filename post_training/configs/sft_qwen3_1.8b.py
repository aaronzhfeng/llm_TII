# =============================================================================
# SFT Configuration for Qwen3-1.8B
# =============================================================================
#
# This configuration fine-tunes the pre-trained Qwen3-1.8B base model
# on instruction-following data using Supervised Fine-Tuning (SFT).
#
# Usage:
#   Single GPU:
#     python train_sft.py configs/sft_qwen3_1.8b.py
#
#   Multi-GPU (8x B200):
#     torchrun --standalone --nproc_per_node=8 train_sft.py configs/sft_qwen3_1.8b.py
#
# =============================================================================

import os

# Prevent memory fragmentation for long fine-tuning runs
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# =============================================================================
# CHECKPOINT & DATA (REQUIRED - Update these paths!)
# =============================================================================

# Path to pre-trained base model checkpoint
checkpoint_path = '/raid/zhf004/llm_TII/enhanced_training_system/out-qwen3-1.8b-b200-50h/ckpt_160000.pt'

# Path to prepared SFT dataset (run prepare_sft.py first)
sft_data_dir = '/raid/zhf004/llm_TII/post_training/data/sft_alpaca'

# Data format: 'jsonl' (recommended) or 'binary'
data_format = 'jsonl'

# =============================================================================
# OUTPUT
# =============================================================================

out_dir = 'out-qwen3-1.8b-sft'

# =============================================================================
# TRAINING HYPERPARAMETERS (SFT-Optimized)
# =============================================================================

# Batch configuration
# Effective batch size = batch_size × gradient_accumulation_steps × num_gpus
# For 8 GPUs: 4 × 4 × 8 = 128 examples per step
batch_size = 4
gradient_accumulation_steps = 4

# Sequence length (match pre-training or shorter for SFT)
block_size = 2048

# =============================================================================
# OPTIMIZER (Lower LR for Fine-tuning)
# =============================================================================

# Learning rate: Much lower than pre-training to avoid catastrophic forgetting
# Pre-training was 3e-4, SFT typically uses 1e-5 to 5e-5
learning_rate = 2e-5
min_lr = 2e-6

# Weight decay: Lower than pre-training
weight_decay = 0.01

# Adam betas
beta1 = 0.9
beta2 = 0.95

# Gradient clipping
grad_clip = 1.0

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

# Total iterations
# For 50K examples with batch size 128: ~390 iters/epoch
# 3 epochs ≈ 1200 iterations
max_iters = 3000

# Short warmup for fine-tuning
warmup_iters = 100

# LR decay over full training
lr_decay_iters = 3000
decay_lr = True

# =============================================================================
# REGULARIZATION
# =============================================================================

# Slight dropout for fine-tuning (was 0.0 in pre-training)
dropout = 0.05

# =============================================================================
# EVALUATION & CHECKPOINTING
# =============================================================================

# Evaluate more frequently during SFT
eval_interval = 200
eval_iters = 50

# Save checkpoints
always_save_checkpoint = True
keep_all_checkpoints = True

# =============================================================================
# LOGGING
# =============================================================================

log_interval = 10
save_log_to_json = True
log_save_interval = 50
gradient_log_interval = 100

# WandB (optional)
wandb_log = True
wandb_project = 'qwen3-sft'
wandb_run_name = 'qwen3-1.8b-sft-alpaca'

# =============================================================================
# SYSTEM
# =============================================================================

device = 'cuda'
dtype = 'bfloat16'

# torch.compile for faster training (recommended for long runs)
compile = True

# Distributed training options
use_zero1 = False  # Not needed for SFT (smaller batches)
use_fsdp = False

# =============================================================================
# TOKENIZER SETTINGS
# =============================================================================

# Qwen3 special tokens
pad_token_id = 151643  # <|endoftext|>

# Loss masking (standard for SFT)
ignore_index = -100

# =============================================================================
# QUICK CONFIGURATIONS (Uncomment one to override defaults)
# =============================================================================

# --- Quick Test (5 minutes) ---
# max_iters = 100
# eval_interval = 50
# compile = False

# --- Fast Run (~30 minutes on 8x B200) ---
# max_iters = 500
# eval_interval = 100

# --- Full Training (~1-2 hours on 8x B200) ---
# max_iters = 3000
# eval_interval = 200

# --- Extended Training (~3-4 hours) ---
# max_iters = 6000
# eval_interval = 500

