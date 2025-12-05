# =============================================================================
# DPO Configuration for Qwen3-1.8B
# =============================================================================
#
# Direct Preference Optimization configuration for aligning the SFT model
# with human preferences.
#
# Prerequisites:
#   1. Complete SFT training first (train_sft.py)
#   2. Prepare DPO dataset (prepare_dpo.py)
#
# Usage:
#   Single GPU:
#     python train_dpo.py configs/dpo_qwen3_1.8b.py
#
#   Multi-GPU:
#     torchrun --standalone --nproc_per_node=8 train_dpo.py configs/dpo_qwen3_1.8b.py
#
# =============================================================================

import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# =============================================================================
# CHECKPOINT & DATA (REQUIRED - Update these paths!)
# =============================================================================

# Path to SFT checkpoint (output of train_sft.py)
sft_checkpoint_path = '/raid/zhf004/llm_TII/post_training/out-qwen3-1.8b-sft/ckpt.pt'

# Path to prepared DPO dataset
dpo_data_dir = '/raid/zhf004/llm_TII/post_training/data/dpo_ultrafeedback'

# Data format
data_format = 'jsonl'

# =============================================================================
# OUTPUT
# =============================================================================

out_dir = 'out-qwen3-1.8b-dpo'

# =============================================================================
# DPO HYPERPARAMETERS
# =============================================================================

# Beta: KL divergence penalty coefficient
# Higher beta = more conservative (stay closer to reference model)
# Lower beta = more aggressive preference optimization
# Typical range: 0.05 - 0.5
beta = 0.1

# Label smoothing (optional, can help stability)
label_smoothing = 0.0

# Reference-free mode (not recommended, but can save memory)
reference_free = False

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Smaller batch size (need to compute both chosen and rejected)
# Effective batch = batch_size × gradient_accumulation × num_gpus
batch_size = 2
gradient_accumulation_steps = 8

# Sequence length
block_size = 2048

# =============================================================================
# OPTIMIZER (Very Low LR for DPO)
# =============================================================================

# Learning rate: Even lower than SFT
# DPO typically uses 1e-7 to 5e-6
learning_rate = 5e-7
min_lr = 1e-7

weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

# DPO typically needs fewer iterations than SFT
max_iters = 1000
warmup_iters = 50
lr_decay_iters = 1000
decay_lr = True

# =============================================================================
# EVALUATION & CHECKPOINTING
# =============================================================================

eval_interval = 100
eval_iters = 50
always_save_checkpoint = True
keep_all_checkpoints = True

# =============================================================================
# LOGGING
# =============================================================================

log_interval = 5
save_log_to_json = True
log_save_interval = 50

wandb_log = False
wandb_project = 'qwen3-dpo'
wandb_run_name = 'qwen3-1.8b-dpo-ultrafeedback'

# =============================================================================
# SYSTEM
# =============================================================================

device = 'cuda'
dtype = 'bfloat16'

# Disable compile for DPO (two models = memory intensive)
compile = False

# =============================================================================
# TOKENIZER
# =============================================================================

pad_token_id = 151643
ignore_index = -100

