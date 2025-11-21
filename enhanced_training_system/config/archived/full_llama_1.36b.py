"""
LLaMA 1.36B Architecture - Production Configuration
===================================================

Based on scaling law optimization from: info/llama_1.36e21_32kV.json

Model Design:
- Parameters: 1.294B (1,294,159,104 exactly)
- Config: 18L-18H-2304D-6144ff (from llama_1.36e21_32kV.json)
- FFN ratio: 8/3 × d_model (classic LLaMA)
- Optimal training: 84.72B tokens (from scaling law analysis)
- Target loss: 2.37 (theoretical minimum)

Architecture (LLaMA-style):
- RoPE (Rotary Position Embeddings)
- RMSNorm (faster than LayerNorm)
- SwiGLU activation (d_ff = 8/3 × d_model = 6144)
- Pre-norm (better training stability)
- No weight tying
- No bias
- Head dimension: 128 (FlashAttention optimal)

Alternative Kernel-Optimized Designs:
- See config/full_llama_1.36b_c1.py: 24L-16H-2048D-5632ff = 1.364B (deeper, narrower)
- See config/full_llama_1.36b_c2.py: 19L-18H-2304D-6144ff = 1.358B (one extra layer)
- See config/full_llama_1.36b_c3.py: 18L-18H-2304D-6656ff = 1.358B (wider FFN)

Usage:
    # Quick test on 6B tokens
    python train.py config/full_llama_1.36b.py

    # Multi-GPU training (4x A100)
    torchrun --standalone --nproc_per_node=4 train.py config/full_llama_1.36b.py

    # Production training (8x B200)
    torchrun --standalone --nproc_per_node=8 train.py \\
        config/full_llama_1.36b.py \\
        --dataset=slimpajama_627b \\
        --use_fsdp=True
"""

# =============================================================================
# ARCHITECTURE - What the model IS
# =============================================================================

# === Architecture Preset ===
arch_preset = 'llama'  # Use LLaMA components: RoPE + RMSNorm + SwiGLU + Pre-norm

# === Attention Backend Override ===
attention_backend = 'flash_attn_2'  # Use FlashAttention-2 (proven and stable)
                                     # Falls back to sdpa → manual if not available

# === Model Dimensions (From JSON: llama_1.36e21_32kV.json) ===
n_layer = 18                # Number of transformer layers
n_head = 18                 # Number of attention heads
n_embd = 2304               # Hidden dimension / embedding dimension
block_size = 2048           # Maximum sequence length / context window
dropout = 0.0               # Dropout rate (LLaMA uses 0.0)
bias = False                # No bias in linear layers (LLaMA standard)

# === FFN Dimension (SwiGLU) ===
# From JSON: intermediate_size = 6144
# Classic LLaMA ratio: 8/3 × d_model = 6144
d_ff = 6144                 # FFN dimension (24×256, from scaling law optimization)
intermediate_size = 6144    # Alias for compatibility

# === Derived Parameters ===
# head_dim = 128                    # Per-head dimension (2304 / 18) - FlashAttention optimal
# vocab_size = 32000                # Will be set from tokenizer metadata

# === Architecture Components (specified by arch_preset='llama') ===
# normalization = 'rmsnorm'         # RMSNorm (faster than LayerNorm)
# norm_eps = 1e-06                  # From JSON: rms_norm_eps
# position_encoding = 'rope'        # Rotary Position Embeddings
# rope_theta = 10000.0              # RoPE base frequency
# ffn_type = 'swiglu'               # SwiGLU activation
# norm_position = 'pre'             # Pre-norm architecture
# weight_tying = False              # No weight tying (better for large models)
# attention_backend = 'flash_attn_2' # FlashAttention-2 if available

# =============================================================================
# TRAINING - How to train it
# =============================================================================

# === Data ===
dataset = 'slimpajama_6b_llama'    # Dataset name (data/slimpajama_6b_llama/)
                                    # Uses LLaMA-2 tokenizer (32K vocab)
                                    # Change to 'slimpajama_627b_llama' for production
gradient_accumulation_steps = 16   # Gradient accumulation steps PER GPU
                                    # Global steps = 64 * num_gpus
                                    # For 8 GPUs: 64 per GPU × 8 = 512 global steps
                                    # For 2 GPUs: 64 per GPU × 2 = 128 global steps
batch_size = 8                     # Micro-batch size per GPU
                                    # Note: For 1.29B model, batch_size=6-8 works on 2× A6000

# Effective batch size = batch_size × gradient_accumulation_steps × num_gpus
# Example (4x A100): 8 × 16 × 4 = 512 samples/iter = 1,048,576 tokens/iter
# Example (8x B200): 16 × 8 × 8 = 1024 samples/iter = 2,097,152 tokens/iter

# === Optimizer (AdamW) ===
learning_rate = 3e-4               # Peak learning rate (scaled for 1.36B model)
max_iters = 25000                  # Total training iterations
                                    # Example: 25k iters × 262k tokens/iter = 6.5B tokens
                                    # Chinchilla optimal (D≈20N): 27B tokens = ~103k iters
weight_decay = 1e-1                # L2 regularization
beta1 = 0.9                        # Adam beta1
beta2 = 0.95                       # Adam beta2 (LLaMA uses 0.95)
grad_clip = 1.0                    # Gradient clipping threshold

# === Learning Rate Schedule ===
decay_lr = True                    # Enable cosine decay
warmup_iters = 2000                # Linear warmup iterations (~2% of training)
lr_decay_iters = 25000             # Should match max_iters
min_lr = 3e-5                      # Minimum LR (10% of peak)

# =============================================================================
# SYSTEM - Where/how to run
# =============================================================================

# === Hardware ===
device = 'cuda'                    # Device type
dtype = 'bfloat16'                 # Training precision (better than float16)
compile = True                     # Use torch.compile() for speedup

# === Advanced Optimizations (B200 specific) ===
use_cuda_graphs = False            # CUDA Graphs (8-15% speedup, requires static shapes)
                                    # Set to True for B200 testing (max performance)
use_dataloader = False             # PyTorch DataLoader with workers (reduces CPU bottleneck)
                                    # Set to True for B200 (prevents CPU bottleneck on fast GPUs)
dataloader_num_workers = 4         # Number of data loading workers (if use_dataloader=True)
dataloader_prefetch_factor = 2     # Number of batches to prefetch per worker

# === Parallelism ===
# For 4x A100 80GB (comfortable):
use_zero1 = True                   # ZeRO-1 optimizer sharding (enabled for testing)
use_fsdp = False                   # Fully Sharded Data Parallel

# For 3x RTX A4500 20GB (tight memory):
# use_zero1 = True                 # CRITICAL: saves ~10GB per GPU
# use_fsdp = False
# batch_size = 2                   # Reduce batch size
# gradient_accumulation_steps = 40 # Increase accumulation

# For 8x B200 128GB (optimal):
# use_zero1 = False
# use_fsdp = True                  # Better scaling for 8+ GPUs
# batch_size = 16
# gradient_accumulation_steps = 8

# =============================================================================
# I/O & LOGGING
# =============================================================================

# === Output ===
out_dir = 'out-llama-1.36b'        # Output directory for checkpoints
eval_interval = 1000               # Evaluate every N iterations
log_interval = 10                  # Log every N iterations
eval_iters = 50                    # Number of iterations for evaluation (reduced for speed)
eval_only = False                  # If True, run evaluation and exit
eval_at_start = False              # If True, run evaluation before first training iteration
always_save_checkpoint = True      # Save checkpoint after each eval
init_from = 'scratch'              # 'scratch', 'resume', or 'gpt2*'

# === Logging ===
save_log_to_json = True            # Save training logs to JSON
log_save_interval = 10             # Save log every N iterations (increased frequency)
gradient_log_interval = 10         # Log gradient stats every N iterations

# === Weights & Biases (optional) ===
wandb_log = False                  # Enable W&B logging
wandb_project = 'llama-1.36b'      # W&B project name
wandb_run_name = 'run-1'           # W&B run name

# =============================================================================
# METADATA (From scaling law analysis: llama_1.36e21_32kV.json)
# =============================================================================

# Model: 18L-18H-2304D-6144ff = 1.294B params
# Architecture: LLaMA-style (RoPE + RMSNorm + SwiGLU + Pre-norm)
# Head dimension: 128 (FlashAttention optimal)
#
# Scaling Law Results:
# - Theoretical loss: 2.372087
# - Optimal configuration: 1.294B params × 84.72B tokens
# - Validation (62M tokens): loss = 4.712
# - Per-token FLOPs (PaLM): 6N + 12LHQT = 9.18 GF/token
#
# Training Recommendations:
# 1. For optimal (~85B tokens): ~324k iterations @ 262k tokens/iter
# 2. For quick testing (6B tokens): ~23k iterations
# 3. For validation: slimpajama_6b (6B tokens) → expect loss ~4.0-4.5
#
# Expected Performance (2× A6000, ZeRO-1):
# - Tokens/sec: ~16,000-18,000
# - MFU: 42-50% (with FlashAttention-2 and PaLM formula)
# - Memory/GPU: ~38-42 GB (batch_size=6 works, 8 is tight)
# - Time to 85B tokens: ~1,500 hours (~62 days)
#
# Expected Performance (4× A100, DDP):
# - Tokens/sec: ~55,000-65,000
# - MFU: 42-52%
# - Memory/GPU: ~25-30 GB
# - Time to 85B tokens: ~370-450 hours (~15-19 days)
#
# Expected Performance (8× B200, FSDP):
# - Tokens/sec: ~150,000-180,000
# - MFU: 48-58%
# - Memory/GPU: ~30-40 GB
# - Time to 85B tokens: ~130-160 hours (~5-7 days)

# =============================================================================
# NOTES & WARNINGS
# =============================================================================

# 1. Tokenizer: This config assumes LLaMA-2 tokenizer (32K vocab)
#    Download before SSH: transformers.LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#
# 2. Dataset: Prepare datasets before training
#    - cd data/slimpajama_6b && python prepare.py
#    - cd data/slimpajama_627b && python prepare.py
#
# 3. Batch size tuning:
#    - Adjust batch_size and gradient_accumulation_steps based on GPU memory
#    - Effective batch should be 512-1024 samples for stable training
#    - Monitor memory usage and reduce batch_size if OOM
#
# 4. Learning rate:
#    - 3e-4 is good starting point for 1.36B model
#    - Increase to 4e-4 if training is too slow to converge
#    - Decrease to 2e-4 if loss is unstable
#
# 5. Early stopping:
#    - Monitor validation loss every eval_interval
#    - Stop when validation loss plateaus (typically after 20-30k iters)
#    - Don't overtrain beyond optimal token count (~85B tokens)

