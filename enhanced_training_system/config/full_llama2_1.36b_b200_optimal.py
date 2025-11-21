# LLaMA2 1.36B B200 Optimized Configuration
# Optimized for 8× B200 GPUs: Pure DDP, DataLoader, FlashAttention-2

# =============================================================================
# ARCHITECTURE
# =============================================================================

arch_preset = 'llama'
attention_backend = 'flash_attn_2'

# === Dimensions ===
n_layer = 18
n_head = 18
n_embd = 2304
block_size = 2048
dropout = 0.0
bias = False
d_ff = 6144
intermediate_size = 6144

# =============================================================================
# TRAINING
# =============================================================================

# === Data ===
dataset = 'slimpajama_6b_llama'
batch_size = 64
gradient_accumulation_steps = 4

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
# SYSTEM
# =============================================================================

# === Hardware ===
device = 'cuda'
dtype = 'bfloat16'
compile = True

# === Optimizations ===
use_cuda_graphs = False
use_dataloader = True
dataloader_num_workers = 16
dataloader_prefetch_factor = 2

# === Parallelism ===
use_zero1 = False
use_fsdp = False

# =============================================================================
# I/O & LOGGING
# =============================================================================

out_dir = 'out-llama-1.36b-b200'
eval_interval = 1000
log_interval = 10
eval_iters = 50
eval_only = False
eval_at_start = False
always_save_checkpoint = False
init_from = 'scratch'

save_log_to_json = True
log_save_interval = 10
gradient_log_interval = 10

wandb_log = False
wandb_project = 'llama-1.36b-b200'
wandb_run_name = 'b200-optimal'

