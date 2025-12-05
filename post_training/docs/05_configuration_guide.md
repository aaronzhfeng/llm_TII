# Configuration Guide

## Overview

Post-training uses Python config files that set global variables. This allows easy experimentation with different hyperparameters.

## SFT Configuration

**File**: `configs/sft_qwen3_1.8b.py`

### Required Settings

```python
# Path to pre-trained checkpoint (MUST UPDATE)
checkpoint_path = '/raid/zhf004/llm_TII/enhanced_training_system/out-qwen3-1.8b-b200-50h/ckpt_160000.pt'

# Path to prepared SFT data (MUST UPDATE if different)
sft_data_dir = '/raid/zhf004/llm_TII/post_training/data/sft_alpaca'

# Output directory
out_dir = 'out-qwen3-1.8b-sft'
```

### Training Hyperparameters

```python
# Batch configuration
batch_size = 4                    # Per-GPU batch size
gradient_accumulation_steps = 4   # Effective batch = 4 × 4 × 8 = 128

# Sequence length
block_size = 2048                 # Max sequence length

# Optimizer
learning_rate = 2e-5              # Much lower than pre-training (3e-4)
min_lr = 2e-6                     # LR floor for cosine decay
weight_decay = 0.01               # L2 regularization
beta1 = 0.9                       # Adam beta1
beta2 = 0.95                      # Adam beta2
grad_clip = 1.0                   # Gradient clipping threshold

# Schedule
max_iters = 3000                  # Total training iterations
warmup_iters = 100                # Linear warmup
lr_decay_iters = 3000             # Cosine decay length
decay_lr = True                   # Enable LR decay

# Regularization
dropout = 0.05                    # Applied during fine-tuning
```

### Evaluation & Checkpointing

```python
eval_interval = 200               # Evaluate every N iterations
eval_iters = 50                   # Batches per evaluation
always_save_checkpoint = True     # Save at every eval
keep_all_checkpoints = True       # Keep all checkpoints
```

### Logging

```python
log_interval = 10                 # Log every N iterations
save_log_to_json = True           # Save JSON logs
log_save_interval = 50            # JSON save frequency
gradient_log_interval = 100       # Gradient stats frequency

# WandB
wandb_log = True
wandb_project = 'qwen3-sft'
wandb_run_name = 'qwen3-1.8b-sft-alpaca'
```

### System Settings

```python
device = 'cuda'
dtype = 'bfloat16'                # bf16 for B200
compile = True                    # torch.compile optimization
use_zero1 = False                 # ZeRO-1 (not needed for SFT)
use_fsdp = False                  # FSDP (not needed for 1.8B)
```

## DPO Configuration

**File**: `configs/dpo_qwen3_1.8b.py`

### Required Settings

```python
# Path to SFT checkpoint (from Stage 1)
sft_checkpoint_path = '/raid/zhf004/llm_TII/post_training/out-qwen3-1.8b-sft/ckpt.pt'

# Path to DPO data
dpo_data_dir = '/raid/zhf004/llm_TII/post_training/data/dpo_ultrafeedback'

# Output
out_dir = 'out-qwen3-1.8b-dpo'
```

### DPO-Specific Parameters

```python
# KL penalty coefficient
beta = 0.1                        # Range: 0.05 - 0.5

# Learning rate (very low for DPO)
learning_rate = 5e-7
min_lr = 5e-8

# Batch (smaller due to memory - 2 models loaded)
batch_size = 2
gradient_accumulation_steps = 8   # Effective = 128

# Iterations (fewer than SFT)
max_iters = 1000
warmup_iters = 50
```

## Command Line Overrides

Override any config parameter from command line:

```bash
# Single parameter
python train_sft.py configs/sft_qwen3_1.8b.py --learning_rate=1e-5

# Multiple parameters
python train_sft.py configs/sft_qwen3_1.8b.py \
    --learning_rate=1e-5 \
    --max_iters=500 \
    --batch_size=2 \
    --compile=False
```

## Quick Configurations

### Quick Test (~5 minutes)

```bash
python train_sft.py configs/sft_qwen3_1.8b.py \
    --max_iters=100 \
    --eval_interval=50 \
    --compile=False
```

### Fast Run (~30 minutes)

```bash
torchrun --standalone --nproc_per_node=8 train_sft.py configs/sft_qwen3_1.8b.py \
    --max_iters=500 \
    --eval_interval=100
```

### Full Training (~1-2 hours)

```bash
torchrun --standalone --nproc_per_node=8 train_sft.py configs/sft_qwen3_1.8b.py
# Uses default max_iters=3000
```

### Extended Training (~3-4 hours)

```bash
torchrun --standalone --nproc_per_node=8 train_sft.py configs/sft_qwen3_1.8b.py \
    --max_iters=6000 \
    --eval_interval=500
```

## Hyperparameter Tuning

### Learning Rate Sweep

```bash
for lr in 1e-5 2e-5 5e-5; do
    python train_sft.py configs/sft_qwen3_1.8b.py \
        --learning_rate=$lr \
        --max_iters=500 \
        --out_dir=out-sft-lr-$lr
done
```

### Beta Sweep (DPO)

```bash
for beta in 0.05 0.1 0.2 0.5; do
    python train_dpo.py configs/dpo_qwen3_1.8b.py \
        --beta=$beta \
        --max_iters=300 \
        --out_dir=out-dpo-beta-$beta
done
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train_sft.py configs/sft_qwen3_1.8b.py --batch_size=2

# Increase gradient accumulation to maintain effective batch
python train_sft.py configs/sft_qwen3_1.8b.py \
    --batch_size=2 \
    --gradient_accumulation_steps=8
```

### Slow Training

```bash
# Enable torch.compile (increases startup time but faster per-iter)
python train_sft.py configs/sft_qwen3_1.8b.py --compile=True
```

### Unstable Training

```bash
# Reduce learning rate
python train_sft.py configs/sft_qwen3_1.8b.py --learning_rate=1e-5

# Increase warmup
python train_sft.py configs/sft_qwen3_1.8b.py --warmup_iters=200

# Reduce gradient clip
python train_sft.py configs/sft_qwen3_1.8b.py --grad_clip=0.5
```

### Resume Training

```bash
# Resume from specific checkpoint
python train_sft.py configs/sft_qwen3_1.8b.py \
    --checkpoint_path=out-qwen3-1.8b-sft/ckpt_001000.pt
```

## Recommended Settings by Dataset Size

| Dataset Size | batch_size | grad_accum | max_iters | eval_interval |
|--------------|------------|------------|-----------|---------------|
| 10K | 4 | 4 | 1000 | 100 |
| 50K | 4 | 4 | 3000 | 200 |
| 100K | 4 | 4 | 5000 | 500 |
| 500K | 4 | 8 | 10000 | 1000 |

