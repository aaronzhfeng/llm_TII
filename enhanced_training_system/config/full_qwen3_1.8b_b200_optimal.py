# Qwen3-1.8B B200 Optimized Configuration
# Optimized for 8× B200 GPUs: Pure DDP, DataLoader, FlashAttention-2

# =============================================================================
# ARCHITECTURE
# =============================================================================

arch_preset = 'custom'

# === Architecture Components ===
normalization = 'rmsnorm'
activation = 'silu'
attention_backend = 'flash_attn_2'
position_encoding = 'rope'
norm_position = 'pre'
ffn_type = 'swiglu'
bias = False
weight_tying = False

# === Dimensions ===
n_layer = 24
n_head = 16
n_embd = 2048
num_key_value_heads = 8
block_size = 2048
vocab_size = 151643
dropout = 0.0
d_ff = 6144
intermediate_size = 6144
rope_theta = 1_000_000
norm_eps = 1e-6

# =============================================================================
# TRAINING - B200 Optimized (Phase 1: Conservative Start)
# =============================================================================

# === Data ===
dataset = 'slimpajama_6b_qwen3'
batch_size = 64                      # Micro-batch per GPU
gradient_accumulation_steps = 4       # Per GPU (global = 4 × 8 = 32)

# === Optimizer ===
learning_rate = 3e-4
max_iters = 25000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# === Learning Rate Schedule ===
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 25000
min_lr = 3e-5

# =============================================================================
# SYSTEM - B200 Specific Optimizations
# =============================================================================

# === Hardware ===
device = 'cuda'
dtype = 'bfloat16'
compile = True

# === Optimizations ===
use_dataloader = True
dataloader_num_workers = 16
dataloader_prefetch_factor = 2
use_cuda_graphs = False

# === Parallelism ===
use_fsdp = False
use_zero1 = False

# =============================================================================
# I/O & LOGGING
# =============================================================================

# === Output ===
out_dir = 'out-qwen3-1.8b-b200'
eval_interval = 1000
log_interval = 10
eval_iters = 50
eval_only = False
eval_at_start = False
always_save_checkpoint = False
init_from = 'scratch'

# === Logging ===
save_log_to_json = True
log_save_interval = 10
gradient_log_interval = 10

# === Weights & Biases ===
wandb_log = False
wandb_project = 'qwen3-1.8b-b200'
wandb_run_name = 'b200-optimal'

