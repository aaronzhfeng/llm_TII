# Training Guide: Quick Start by Model

Complete commands for training GPT-2, LLaMA 2, LLaMA 3, and Qwen3 models.  
**Each section is self-contained** - pick your model and follow the commands.

---

## 📋 Table of Contents

- [GPT-2 1.36B](#gpt-2-136b)
- [LLaMA 2 1.36B](#llama-2-136b)
- [LLaMA 3 Optimal (1.5B)](#llama-3-optimal-15b) ⭐ **Grid Search Optimized**
- [LLaMA 3 Chinchilla (2.2B)](#llama-3-chinchilla-22b) ⭐ **Grid Search Optimized**
- [Qwen3 Optimal (1.8B)](#qwen3-optimal-18b) ⭐ **NEW - Best Opensource Dense Model**
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)

---

## GPT-2 1.36B

### 1. Prepare Dataset

```bash
cd /root/llm_TII/enhanced_training_system

# Prepare SlimPajama-6B with GPT-2 tokenizer
cd data/slimpajama_6b_gpt2
python prepare.py
# Expected: train.bin (~6GB), val.bin (~30MB), meta.pkl

cd ../..
```

### 2. Smoke Test (10 iterations, 1 minute)

```bash
# Single GPU test
python train.py config/full_gpt2_1.36b.py \
  --max_iters=10 \
  --compile=False

# Expected: ✓ Model builds, ✓ Data loads, ✓ No errors
```

### 3. Quick Test (100 iterations, ~10 minutes)

```bash
# Single GPU
python train.py config/full_gpt2_1.36b.py \
  --max_iters=100 \
  --eval_interval=50

# Multi-GPU (4 GPUs)
torchrun --standalone --nproc_per_node=4 train.py \
  config/full_gpt2_1.36b.py \
  --max_iters=100 \
  --eval_interval=50
```

### 4. Full Training (2000 iterations, ~6-8 hours on 2× A6000)

```bash
# 2× A6000 (Current testing setup)
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_gpt2_1.36b.py \
  --max_iters=2000 \
  --use_zero1=True \
  --batch_size=4 \
  --gradient_accumulation_steps=32 \
  --always_save_checkpoint=False
```

### 5. Monitor Training

```bash
# Watch progress
tail -f out-gpt2-1.36b/run_*.json

# Check GPU usage
nvidia-smi -l 5

# View latest metrics
python -c "
import json, glob
log = sorted(glob.glob('out-gpt2-1.36b/run_*.json'))[-1]
data = json.load(open(log))
latest = data['training_iterations'][-1]
print(f\"Iter {latest['iter']}: Loss {latest['loss']:.4f}, MFU {latest.get('mfu', 'N/A')}\")
"
```

### 6. Evaluation

```bash
# Evaluate checkpoint
python train.py config/full_gpt2_1.36b.py \
  --init_from=resume \
  --eval_only=True
```

**Expected Results (6B tokens with 2× A6000):**
- Loss after 2000 iters: ~8.0-9.0 (early training)
- MFU: 35-40% (with ZeRO-1)
- Tokens/sec: ~12,000-15,000
- Memory: ~25-28 GB/GPU with ZeRO-1

---

## LLaMA 2 1.36B

### 1. Download Tokenizer (First time only)

```bash
cd /root/llm_TII/enhanced_training_system

# Install transformers if needed
pip install transformers sentencepiece protobuf

# Download LLaMA-2 tokenizer
python << 'EOF'
from transformers import LlamaTokenizer

print("Downloading LLaMA-2 tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.save_pretrained("./llama2_tokenizer")
print(f"✓ Saved to ./llama2_tokenizer/ (vocab={tokenizer.vocab_size})")
EOF

# Verify files
ls -lh llama2_tokenizer/
# Should show: tokenizer.model, tokenizer_config.json
```

**Note:** You may need to:
- Accept LLaMA-2 license on HuggingFace: https://huggingface.co/meta-llama/Llama-2-7b-hf
- Login: `huggingface-cli login`

### 2. Prepare Dataset

```bash
# Prepare SlimPajama-6B with LLaMA-2 tokenizer
cd data/slimpajama_6b_llama
python prepare.py
# Expected: train.bin (~6GB), val.bin (~30MB), meta.pkl

cd ../..
```

### 3. Smoke Test (10 iterations, 1 minute)

```bash
python train.py config/full_llama_1.36b.py \
  --max_iters=10 \
  --compile=False

# Expected: ✓ Architecture: 18L-18H-2304D-6144ff
#           ✓ Vocab: 32000 (LLaMA-2)
#           ✓ No errors
```

### 4. Quick Test (100 iterations, ~15 minutes)

```bash
# Single GPU
python train.py config/full_llama_1.36b.py \
  --max_iters=100 \
  --eval_interval=50

# Multi-GPU (4 GPUs)
torchrun --standalone --nproc_per_node=4 train.py \
  config/full_llama_1.36b.py \
  --max_iters=100 \
  --eval_interval=50
```

### 5. Full Training (2000 iterations, ~6-8 hours on 2× A6000)

```bash
# 2× A6000 (Current testing setup)
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama_1.36b.py \
  --max_iters=2000 \
  --batch_size=6 \
  --gradient_accumulation_steps=32 \
  --use_zero1=True
```

### 6. Monitor Training

```bash
# Watch progress
tail -f out-llama-1.36b/run_*.json

# Check GPU usage
nvidia-smi -l 5

# Generate plots (after some iterations)
cd plots
python plot_single_run.py ../saves/run_*.json
cd ..
```

### 7. Evaluation

```bash
python train.py config/full_llama_1.36b.py \
  --init_from=resume \
  --eval_only=True
```

**Expected Results (6B tokens with 2× A6000):**
- Loss after 2000 iters: ~7.5-8.5 (early training)
- MFU: 35-40% (with ZeRO-1)
- Tokens/sec: ~10,000-13,000
- Memory: ~28-32 GB/GPU with ZeRO-1

**Note:** For optimal convergence (85B tokens), would need ~32,000 iterations on production hardware

---

## LLaMA 3 Optimal (1.5B)

⭐ **Optimized via backward N-D grid search for 1.36e21 FLOPs budget**

**Architecture:** 18L-16H-2048D-7168ff with GQA (8 KV heads, 2:1 ratio)  
**Expected Loss:** 2.335 (best achievable with this budget)  
**Optimal Tokens:** 101.909B  
**Training Time:** ~15 days (8× A100) or ~4 days (8× B200)

### 1. Download Tokenizer (Same as LLaMA 3.1)

```bash
cd /root/llm_TII/enhanced_training_system

# Download LLaMA-3 tokenizer (128K vocab)
python << 'EOF'
from transformers import AutoTokenizer

print("Downloading LLaMA-3.1 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer.save_pretrained("./llama3_tokenizer")
print(f"✓ Saved to ./llama3_tokenizer/ (vocab={tokenizer.vocab_size})")
EOF
```

### 2. Prepare Dataset

```bash
# For testing: Prepare SlimPajama-6B with LLaMA-3 tokenizer (20-40 min)
cd data/slimpajama_6b_llama3
python prepare.py

cd ../..

# For optimal training (102B tokens): Use slimpajama_627b_llama3 instead
# (Requires 1.2TB storage and 10-20 hours preparation)
```

### 3. Smoke Test (10 iterations, 1 minute)

```bash
python train.py config/full_llama3_1.5b_optimal.py \
  --max_iters=10 \
  --compile=False

# Expected: ✓ Architecture: 18L-16H-2048D with GQA (8 KV)
#           ✓ Parameters: ~1.545B
#           ✓ No errors
```

### 4. Quick Test (100 iterations, ~15 minutes)

```bash
# 2× A6000 with ZeRO-1
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama3_1.5b_optimal.py \
  --max_iters=100 \
  --eval_interval=50 \
  --use_zero1=True
```

### 5. Full Training (2000 iterations, ~6-8 hours on 2× A6000)

```bash
# 2× A6000 with ZeRO-1 (Current testing setup)
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama3_1.5b_optimal.py \
  --max_iters=2000 \
  --use_zero1=True
```

### 6. Monitor Training

```bash
# Watch progress
tail -f out-llama3-1.5b-optimal/run_*.json

# Check GPU usage
nvidia-smi -l 5
```

### 7. Evaluation

```bash
python train.py config/full_llama3_1.5b_optimal.py \
  --init_from=resume \
  --eval_only=True
```

**Expected Results (6B tokens with 2× A6000):**
- Loss after 2000 iters: ~7.0-8.0 (early training)
- MFU: 40-45% (with ZeRO-1)
- Tokens/sec: ~10,000-12,000
- Memory: ~30-35 GB/GPU with ZeRO-1

**Why This Config:**
- ✅ **Best loss** for 1.36e21 FLOPs budget
- ✅ Fits comfortably on 2× A6000
- ✅ Smaller model (easier deployment)
- ✅ Ideal for research/loss optimization

**Note:** For full convergence (102B tokens), would need ~50,000 iterations on production hardware

**Compare to:** [LLaMA 3 Chinchilla (2.2B)](#llama-3-chinchilla-22b) for larger model

---

## LLaMA 3 Chinchilla (2.2B)

⭐ **Optimized via backward N-D grid search with Chinchilla D≈20N constraint**

**Architecture:** 30L-16H-2048D-7168ff with GQA (8 KV heads, 2:1 ratio)  
**Expected Loss:** 2.351 (only 0.7% higher than optimal)  
**Optimal Tokens:** 61.545B (40% fewer!)  
**Training Time:** ~10 days (8× A100) or ~3 days (8× B200)

### 1. Download Tokenizer (Same as above)

```bash
cd /root/llm_TII/enhanced_training_system

# Download LLaMA-3 tokenizer (128K vocab)
python << 'EOF'
from transformers import AutoTokenizer

print("Downloading LLaMA-3.1 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer.save_pretrained("./llama3_tokenizer")
print(f"✓ Saved to ./llama3_tokenizer/ (vocab={tokenizer.vocab_size})")
EOF
```

### 2. Prepare Dataset

```bash
# For testing: Prepare SlimPajama-6B with LLaMA-3 tokenizer (20-40 min)
cd data/slimpajama_6b_llama3
python prepare.py

cd ../..

# For optimal training (62B tokens): Use slimpajama_627b_llama3 instead
# (Requires 1.2TB storage and 10-20 hours preparation)
```

### 3. Smoke Test (10 iterations, 1 minute)

```bash
# IMPORTANT: Requires multi-GPU! This model is too large for single GPU
# For single GPU testing, use full_llama3_1.5b_optimal.py instead

# 2× A6000 smoke test with ZeRO-1 and reduced batch:
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama3_2.2b_chinchilla.py \
  --use_zero1=True \
  --batch_size=2 \
  --gradient_accumulation_steps=64 \
  --max_iters=10 \
  --compile=False

# Expected: ✓ Architecture: 30L-16H-2048D with GQA (8 KV)
#           ✓ Parameters: ~2.224B (larger model!)
#           ✓ Memory: ~45 GB peak per GPU
#           ✓ No errors
```

### 4. Quick Test (100 iterations, ~40 minutes)

```bash
# 2× A6000 with ZeRO-1 (enables larger batch)
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama3_2.2b_chinchilla.py \
  --use_zero1=True \
  --batch_size=4 \
  --gradient_accumulation_steps=32 \
  --max_iters=100 \
  --eval_interval=50
```

### 5. Full Training (2000 iterations, ~14-16 hours on 2× A6000)

```bash
# 2× A6000 with ZeRO-1 (Current testing setup)
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama3_2.2b_chinchilla.py \
  --max_iters=2000 \
  --use_zero1=True \
  --use_fsdp=False \
  --batch_size=4 \
  --gradient_accumulation_steps=32
```

### 6. Monitor Training

```bash
# Watch progress
tail -f out-llama3-2.2b-chinchilla/run_*.json

# Check GPU usage
nvidia-smi -l 5
```

### 7. Evaluation

```bash
python train.py config/full_llama3_2.2b_chinchilla.py \
  --init_from=resume \
  --eval_only=True
```

**Expected Results (6B tokens with 2× A6000):**
- Loss after 2000 iters: ~7.5-8.5 (early training)
- MFU: 35-40% (with ZeRO-1 and batch_size=4)
- Tokens/sec: ~8,000-9,000
- Memory: ~40-42 GB/GPU with ZeRO-1

**Why This Config:**
- ✅ **44% larger model** (better downstream tasks)
- ✅ **33% faster training** (fewer tokens needed for convergence)
- ✅ Follows Chinchilla best practices
- ✅ Fits on 2× A6000 with ZeRO-1

**Trade-off:** Slightly higher loss than 1.5B optimal, but much larger, more capable model

**Note:** For full convergence (62B tokens), would need ~29,000 iterations on production hardware

---

## Qwen3 Optimal (1.8B)

⭐ **Best opensource dense model - Optimized via backward N-D grid search for 1.36e21 FLOPs budget**

**Architecture:** 24L-16H-2048D-6144ff with GQA (8 KV heads, 2:1 ratio)  
**Expected Loss:** 2.340 (best achievable with this budget!)  
**Optimal Tokens:** 81.727B  
**Training Time:** ~110-120 hours (2× A6000) or ~4-6 hours (8× B200)

### 1. Download Tokenizer (First time only)

```bash
cd /root/llm_TII/enhanced_training_system

# Install transformers if needed
pip install transformers

# Download Qwen3 tokenizer (152K vocab)
python << 'EOF'
from transformers import AutoTokenizer

print("Downloading Qwen3 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
tokenizer.save_pretrained("./qwen3_tokenizer")
print(f"✓ Saved to ./qwen3_tokenizer/ (vocab={tokenizer.vocab_size})")
EOF
```

### 2. Prepare Dataset

```bash
# For testing: Prepare SlimPajama-6B with Qwen3 tokenizer (20-40 min)
cd data/slimpajama_6b_qwen3
python prepare.py

cd ../..

# For optimal training (82B tokens): Use slimpajama_627b_qwen3 instead
# (Requires 1.2TB storage and 10-20 hours preparation)
```

### 3. Smoke Test (10 iterations, 1 minute)

```bash
python train.py config/full_qwen3_1.8b_optimal.py \
  --max_iters=10 \
  --compile=False

# Expected: ✓ Architecture: 24L-16H-2048D with GQA (8 KV)
#           ✓ Parameters: ~1.830B
#           ✓ Extended RoPE (theta=1M)
#           ✓ No errors
```

### 4. Quick Test (100 iterations, ~15 minutes)

```bash
# 2× A6000 with ZeRO-1
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --max_iters=100 \
  --eval_interval=50 \
  --use_zero1=True
```

### 5. Full Training (2000 iterations, ~10-12 hours on 2× A6000)

```bash
# 2× A6000 with ZeRO-1 (Current testing setup)
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --max_iters=2000 \
  --use_zero1=True

torchrun --standalone --nproc_per_node=2 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --use_zero1=True \
  --max_iters=2000 \
  --compile=True \
  --batch_size=2 \
  --gradient_accumulation_steps=96
```

### 6. Monitor Training

```bash
# Watch progress
tail -f out-qwen3-1.8b-optimal/run_*.json

# Check GPU usage
nvidia-smi -l 5
```

### 7. Evaluation

```bash
python train.py config/full_qwen3_1.8b_optimal.py \
  --init_from=resume \
  --eval_only=True
```

**Expected Results (6B tokens with 2× A6000):**
- Loss after 2000 iters: ~7.0-8.0 (early training)
- MFU: 38-42% (with ZeRO-1)
- Tokens/sec: ~9,000-11,000
- Memory: ~36-40 GB/GPU with ZeRO-1

**Why This Config:**
- ✅ **Best loss** for 1.36e21 FLOPs budget (2.340)
- ✅ **Deeper architecture** (24 vs 18 layers) - better representation
- ✅ **Extended RoPE** (1M theta) - superior long-context handling
- ✅ **Best opensource tokenizer** (152K vocab, BBPE)
- ✅ **20% fewer tokens** needed than LLaMA 3 optimal
- ✅ Proven Qwen family design principles

**Architecture Highlights:**
- **Deeper**: 24 layers (vs 18 for LLaMA 3)
- **GQA**: 8 KV heads, 16 Q heads (2:1 ratio)
- **RoPE theta**: 1,000,000 (2× LLaMA 3's 500K)
- **Vocabulary**: 151,936 tokens (best compression)
- **FFN Type**: SwiGLU 3.0× expansion

**Comparison with LLaMA 3 Optimal:**
- Similar expected loss (2.340 vs 2.335)
- 19% more parameters (1.8B vs 1.5B)
- 20% fewer training tokens (82B vs 102B)
- Better tokenization efficiency

**Note:** For full convergence (82B tokens), would need ~39,000 iterations on production hardware

---

## Hardware Requirements

### Current Testing Setup: 2× A6000 (48GB each)

| Model | Memory/GPU | Batch Size | ZeRO-1 | MFU | Tokens/sec |
|-------|-----------|------------|--------|-----|------------|
| **GPT-2 1.36B** | 25-28 GB | 4 | ✅ Yes | 35-40% | ~12-15k |
| **LLaMA 2 1.36B** | 28-32 GB | 6 | ✅ Yes | 35-40% | ~10-13k |
| **LLaMA 3 Optimal (1.5B)** | 30-35 GB | 8 | ✅ Yes | 40-45% | ~10-12k |
| **LLaMA 3 Chinchilla (2.2B)** | 40-42 GB | 4 | ✅ Yes | 35-40% | ~8-9k |
| **Qwen3 Optimal (1.8B)** | 36-40 GB | 6 | ✅ Yes | 38-42% | ~9-11k |

**Key Points:**
- **ZeRO-1 is essential** for all models (saves ~50% optimizer memory)
- Larger models need smaller batch sizes to fit in memory
- All configurations tested and working on 2× A6000
- 2000 iterations = ~6-16 hours depending on model size
- **Qwen3** has best loss for compute budget (deeper architecture)

---

## Troubleshooting

### OOM (Out of Memory)

```bash
# Solution 1: Reduce batch size
--batch_size=2 --gradient_accumulation_steps=64

# Solution 2: Enable ZeRO-1
--use_zero1=True

# Solution 3: Enable FSDP (8+ GPUs)
--use_fsdp=True

# Solution 4: Reduce sequence length (testing only)
--block_size=1024
```

### Dataset Not Found

```bash
# Verify dataset exists
ls -lh data/slimpajama_6b_*/

# If missing, run prepare.py
cd data/slimpajama_6b_llama  # or _gpt2, _llama3
python prepare.py
```

### Tokenizer Not Found

```bash
# For LLaMA 2/3: Download locally first (see sections above)
# Then verify:
ls -lh llama2_tokenizer/  # or llama3_tokenizer/

# If blocked, modify prepare.py to use local path:
# tokenizer = LlamaTokenizer.from_pretrained("../../llama2_tokenizer")
```

### Low MFU (<20%)

**Check:**
1. Is `compile=True`? (should be enabled for speed)
2. Is GPU utilization <50%? Run `nvidia-smi`
3. Is data loading slow? Check CPU usage

**Solutions:**
- Enable torch.compile: `--compile=True`
- Check for CPU bottlenecks
- Use faster storage (SSD/NVMe)

### Loss Not Decreasing

**Check:**
1. Learning rate: Should be ~3e-4
2. Gradient clipping: Should be 1.0
3. Warmup: Should be ~2000 iterations

**Solutions:**
```bash
# Try higher learning rate
--learning_rate=4e-4

# Increase warmup
--warmup_iters=3000

# Check gradient norms in logs (should be 0.5-2.0)
```

---

## Quick Command Reference

```bash
# === Data Preparation ===
cd data/slimpajama_6b_gpt2 && python prepare.py && cd ../..
cd data/slimpajama_6b_llama && python prepare.py && cd ../..
cd data/slimpajama_6b_llama3 && python prepare.py && cd ../..
cd data/slimpajama_6b_qwen3 && python prepare.py && cd ../..

# === Smoke Tests (10 iterations) ===
python train.py config/full_gpt2_1.36b.py --max_iters=10 --compile=False
python train.py config/full_llama_1.36b.py --max_iters=10 --compile=False
python train.py config/full_llama3_1.5b_optimal.py --max_iters=10 --compile=False
python train.py config/full_qwen3_1.8b_optimal.py --max_iters=10 --compile=False

# === Full Training (2000 iterations on 2× A6000) ===
torchrun --standalone --nproc_per_node=2 train.py config/full_gpt2_1.36b.py \
  --max_iters=2000 --batch_size=4 --gradient_accumulation_steps=32 --use_zero1=True

torchrun --standalone --nproc_per_node=2 train.py config/full_llama_1.36b.py \
  --max_iters=2000 --batch_size=6 --gradient_accumulation_steps=32 --use_zero1=True

torchrun --standalone --nproc_per_node=2 train.py config/full_llama3_1.5b_optimal.py \
  --max_iters=2000 --use_zero1=True

torchrun --standalone --nproc_per_node=2 train.py config/full_llama3_2.2b_chinchilla.py \
  --max_iters=2000 --use_zero1=True --batch_size=4 --gradient_accumulation_steps=32

torchrun --standalone --nproc_per_node=2 train.py config/full_qwen3_1.8b_optimal.py \
  --max_iters=2000 --use_zero1=True

# === Monitoring ===
tail -f out-*/run_*.json
nvidia-smi -l 5

# === Evaluation ===
python train.py config/full_llama3_1.5b_optimal.py --init_from=resume --eval_only=True
python train.py config/full_qwen3_1.8b_optimal.py --init_from=resume --eval_only=True
```

---

## Additional Resources

- **Detailed Guide**: See `docs/TRAINING_GUIDE_DETAILED.md` for in-depth explanations
- **System Overview**: See `SYSTEM_OVERVIEW.md` for architecture details
- **Testing Guide**: See `TESTING.md` for all test commands
- **LLaMA 3 Implementation**: See `docs/LLAMA3_AND_SCALING_LAW_IMPLEMENTATION.md`

---

**Need help?** Check the troubleshooting section above or refer to the detailed documentation.

