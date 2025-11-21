# B200 MFU Testing Guide: LLaMA 2 1.36B

**Purpose:** Systematic MFU testing on 8× B200 GPUs to measure the impact of each optimization technique.

**Prerequisites:**
- ✅ Dataset prepared: `data/slimpajama_6b_llama/train.bin` exists
- ✅ Virtual environment activated: `source venv/bin/activate`
- ✅ PyTorch with B200 support installed (CUDA 12.8, sm_100)
- ✅ FlashAttention-2 installed (FA3 optional)

**Model:** LLaMA 2 1.36B (18L-18H-2304D-6144ff)  
**Hardware:** 8× NVIDIA B200 GPUs (192GB each)  
**Test Duration:** 100-200 iterations per test (~5-10 minutes each)

---

## 📋 Table of Contents

1. [Baseline Tests (2 GPUs)](#baseline-tests-2-gpus)
2. [Scaling Tests (8 GPUs)](#scaling-tests-8-gpus)
3. [Optimization Tiers (8 GPUs)](#optimization-tiers-8-gpus)
4. [Ablation Studies](#ablation-studies)
5. [Memory Limit Tests](#memory-limit-tests)
6. [Analysis & Comparison](#analysis--comparison)

---

## 🔧 Setup

```bash
# Navigate to training directory
cd /home/zhf004/llm_TII/enhanced_training_system

# Activate environment
source ../venv/bin/activate

# Verify dataset
ls -lh data/slimpajama_6b_llama/train.bin
# Should show ~6GB file

# Check available GPUs
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

---

## 1. Baseline Tests (2 GPUs)

**Purpose:** Establish baseline performance before scaling to 8 GPUs.

### Test 1.1: Minimal Configuration (Pure DDP)

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --gradient_log_interval=50 \
  --eval_iters=20 \
  --compile=False \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_fsdp=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected MFU: ~40-50% (2 GPUs)
# Save output: out-llama-1.36b/run_*.json
```

### Test 1.2: With torch.compile()

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_fsdp=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected MFU: ~50-60% (+10-20% vs no compile)
# Expected tokens/sec: Higher than Test 1.1
```

### Test 1.3: Baseline + DataLoader

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_fsdp=False \
  --use_cuda_graphs=False \
  --use_dataloader=True \
  --dataloader_num_workers=4 \
  --dataloader_prefetch_factor=2 \
  --always_save_checkpoint=False

# Expected MFU: ~52-62% (slight improvement if CPU-bound)
# Expected tokens/sec: Similar or slightly higher than 1.2
```

---

## 2. Scaling Tests (8 GPUs)

**Purpose:** Measure how MFU scales from 2 to 8 GPUs with same per-GPU workload.

### Test 2.1: 8 GPU Pure DDP (No Optimizations)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=False \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_fsdp=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected MFU: ~35-45% (lower than 2 GPU due to DDP overhead)
# Expected global tokens/sec: ~4× higher than 2 GPU test
# Key metric: Compare per-GPU efficiency vs Test 1.1
```

### Test 2.2: 8 GPU with torch.compile()

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_fsdp=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected MFU: ~45-55%
# Key question: Does compile() scale well to 8 GPUs?
```

---

## 3. Optimization Tiers (8 GPUs)

**Purpose:** Three-tier optimization strategy for production use.

### 🥉 Tier 1: Conservative (Proven Stable)

**Target MFU:** 45-55%  
**Risk:** Low  
**Use when:** First time testing, production runs

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_fsdp=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Optimizations enabled:
# ✅ FlashAttention-2
# ✅ torch.compile()
# ✅ bfloat16
# ❌ DataLoader (not needed)
# ❌ CUDA Graphs (stability risk)
```

### 🥈 Tier 2: Balanced (Better CPU Efficiency)

**Target MFU:** 50-60%  
**Risk:** Medium  
**Use when:** CPUs are slower than GPUs (likely on B200)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_fsdp=False \
  --use_cuda_graphs=False \
  --use_dataloader=True \
  --dataloader_num_workers=4 \
  --dataloader_prefetch_factor=2 \
  --always_save_checkpoint=False

# Additional optimizations:
# ✅ PyTorch DataLoader (4 workers)
# ✅ Prefetching (2× batches per worker)
```

### 🥇 Tier 3: Maximum Performance (Experimental)

**Target MFU:** 55-65%  
**Risk:** High  
**Use when:** Stability confirmed, need maximum speed

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_fsdp=False \
  --use_cuda_graphs=True \
  --use_dataloader=True \
  --dataloader_num_workers=4 \
  --dataloader_prefetch_factor=2 \
  --always_save_checkpoint=False

# All optimizations enabled:
# ✅ FlashAttention-2
# ✅ torch.compile()
# ✅ bfloat16
# ✅ PyTorch DataLoader
# ✅ CUDA Graphs (5-15% speedup)

# Warning: CUDA Graphs may fail with DDP + gradient accumulation
# If error occurs, fallback to Tier 2
```

---

## 4. Ablation Studies

**Purpose:** Isolate the impact of each optimization technique.

### Test 4.1: FlashAttention Impact

```bash
# WITHOUT FlashAttention (SDPA fallback)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=sdpa \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Compare with Tier 1
# Expected: 20-30% slower than FlashAttention-2
```

### Test 4.2: Precision Impact (float32 vs bfloat16)

```bash
# float32 (highest precision, slowest)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=100 \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=float32 \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Compare with Tier 1 (bfloat16)
# Expected: 2× slower, 2× more memory
```

### Test 4.3: Gradient Accumulation Impact

```bash
# Test A: gradient_accumulation_steps=1 (frequent DDP sync)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=1 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Test B: gradient_accumulation_steps=16 (less frequent sync)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=16 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Compare: Higher grad_accum = fewer DDP syncs = higher MFU
# Expected: Test B should have 5-10% higher MFU
```

### Test 4.4: Batch Size Impact

```bash
# Small batch (more iterations, less GPU utilization)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=8 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Large batch (fewer iterations, better GPU utilization)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --attention_backend=flash_attn_2 \
  --max_iters=200 \
  --batch_size=32 \
  --gradient_accumulation_steps=8 \
  --eval_interval=200 \
  --log_interval=10 \
  --compile=True \
  --dtype=bfloat16 \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected: batch_size=32 should have 10-20% higher MFU
# Trade-off: Larger batch uses more memory
```

---

## 5. Memory Limit Tests

**Purpose:** Find the maximum batch size that fits in 192GB B200 memory.

### Test 5.1: Binary Search for Max Batch Size

```bash
# Test batch_size in sequence until OOM
for BS in 32 48 64 96 128; do
  echo "=== Testing batch_size=$BS ==="
  torchrun --standalone --nproc_per_node=8 train.py \
    config/full_llama_1.36b.py \
    --attention_backend=flash_attn_2 \
    --max_iters=20 \
    --batch_size=$BS \
    --gradient_accumulation_steps=8 \
    --eval_interval=200 \
    --log_interval=10 \
    --compile=True \
    --dtype=bfloat16 \
    --use_zero1=False \
    --use_cuda_graphs=False \
    --use_dataloader=False \
    --always_save_checkpoint=False
  
  if [ $? -ne 0 ]; then
    echo "❌ OOM at batch_size=$BS"
    break
  else
    echo "✅ batch_size=$BS fits!"
  fi
  
  sleep 5
done

# Monitor with: watch -n 1 nvidia-smi
# Goal: Find optimal batch_size that maximizes MFU without OOM
```

### Test 5.2: Memory vs Performance Trade-off

```bash
# Conservative (50% memory usage)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --batch_size=16 \
  --gradient_accumulation_steps=8 \
  --max_iters=200 \
  --compile=True \
  --always_save_checkpoint=False

# Balanced (70% memory usage)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --batch_size=24 \
  --gradient_accumulation_steps=8 \
  --max_iters=200 \
  --compile=True \
  --always_save_checkpoint=False

# Aggressive (90% memory usage)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama_1.36b.py \
  --batch_size=32 \
  --gradient_accumulation_steps=8 \
  --max_iters=200 \
  --compile=True \
  --always_save_checkpoint=False

# Compare: MFU vs memory usage vs stability
```

---

## 6. Analysis & Comparison

### Extract Performance Metrics

```bash
cd /home/zhf004/llm_TII/enhanced_training_system

python << 'EOF'
import json
import glob
import os
from pathlib import Path

# Find all recent run logs
log_files = sorted(glob.glob('out-llama-1.36b/run_*.json'), 
                   key=os.path.getmtime, reverse=True)

print("=" * 80)
print("B200 MFU TESTING SUMMARY (LLaMA 2 1.36B)")
print("=" * 80)

for i, log_path in enumerate(log_files[:10]):  # Last 10 runs
    try:
        with open(log_path) as f:
            data = json.load(f)
        
        # Extract metadata
        startup = data.get('startup_info', {})
        config = startup.get('config', {})
        iters = data.get('training_iterations', [])
        
        if not iters or len(iters) < 50:
            continue
        
        # Skip warmup iterations
        valid_iters = iters[50:]
        
        if not valid_iters:
            continue
        
        # Calculate averages
        avg_mfu = sum(i.get('mfu', 0) for i in valid_iters) / len(valid_iters)
        avg_tokens = sum(i.get('tokens_per_sec', 0) for i in valid_iters) / len(valid_iters)
        avg_mem = sum(i.get('memory_allocated_gb', 0) for i in valid_iters) / len(valid_iters)
        avg_time = sum(i.get('dt_ms', 0) for i in valid_iters) / len(valid_iters)
        
        # Print summary
        print(f"\n{i+1}. Run: {Path(log_path).stem}")
        print(f"   Config:")
        print(f"      GPUs: {config.get('world_size', 'N/A')}")
        print(f"      Batch size: {config.get('batch_size', 'N/A')}")
        print(f"      Grad accum: {config.get('gradient_accumulation_steps_per_gpu', 'N/A')} (per GPU)")
        print(f"      Attention: {config.get('attention_backend', 'N/A')}")
        print(f"      Compile: {config.get('compile', 'N/A')}")
        print(f"      DataLoader: {config.get('use_dataloader', 'N/A')}")
        print(f"      CUDA Graphs: {config.get('use_cuda_graphs', 'N/A')}")
        print(f"      Precision: {config.get('dtype', 'N/A')}")
        print(f"   Performance:")
        print(f"      Avg MFU: {avg_mfu:.2f}%")
        print(f"      Avg tokens/sec: {avg_tokens:,.0f}")
        print(f"      Avg memory/GPU: {avg_mem:.1f} GB")
        print(f"      Avg time/iter: {avg_time:.0f} ms")
        print(f"   Iterations: {len(iters)} total, {len(valid_iters)} analyzed")
    
    except Exception as e:
        print(f"\n{i+1}. Error reading {Path(log_path).stem}: {e}")

print("\n" + "=" * 80)
EOF
```

### Compare Optimization Impact

```bash
# Generate comparison table
python << 'EOF'
import json
import glob
import os

log_files = sorted(glob.glob('out-llama-1.36b/run_*.json'), 
                   key=os.path.getmtime, reverse=True)

print("\n" + "=" * 100)
print("OPTIMIZATION IMPACT COMPARISON")
print("=" * 100)
print(f"{'Config':<35} {'GPUs':>5} {'BS':>4} {'GA':>4} {'MFU':>7} {'Tokens/s':>10} {'Mem/GPU':>9} {'Time/iter':>10}")
print("-" * 100)

for log_path in log_files[:10]:
    try:
        with open(log_path) as f:
            data = json.load(f)
        
        startup = data.get('startup_info', {})
        config = startup.get('config', {})
        iters = data.get('training_iterations', [])
        
        if not iters or len(iters) < 50:
            continue
        
        valid_iters = iters[50:]
        if not valid_iters:
            continue
        
        # Calculate averages
        avg_mfu = sum(i.get('mfu', 0) for i in valid_iters) / len(valid_iters)
        avg_tokens = sum(i.get('tokens_per_sec', 0) for i in valid_iters) / len(valid_iters)
        avg_mem = sum(i.get('memory_allocated_gb', 0) for i in valid_iters) / len(valid_iters)
        avg_time = sum(i.get('dt_ms', 0) for i in valid_iters) / len(valid_iters)
        
        # Build config string
        opts = []
        if config.get('compile'):
            opts.append('compile')
        if config.get('use_dataloader'):
            opts.append('dataloader')
        if config.get('use_cuda_graphs'):
            opts.append('cudagraph')
        if config.get('use_zero1'):
            opts.append('zero1')
        
        attn = config.get('attention_backend', 'N/A')
        dtype = config.get('dtype', 'N/A')
        
        config_str = f"{attn}+{dtype}"
        if opts:
            config_str += f"+{'+'.join(opts)}"
        
        # Print row
        print(f"{config_str:<35} "
              f"{config.get('world_size', 'N/A'):>5} "
              f"{config.get('batch_size', 'N/A'):>4} "
              f"{config.get('gradient_accumulation_steps_per_gpu', 'N/A'):>4} "
              f"{avg_mfu:>6.2f}% "
              f"{avg_tokens:>10,.0f} "
              f"{avg_mem:>8.1f}G "
              f"{avg_time:>9.0f}ms")
    
    except Exception as e:
        pass

print("=" * 100)
print("\nLegend:")
print("  BS = Batch Size (per GPU)")
print("  GA = Gradient Accumulation Steps (per GPU)")
print("  MFU = Model FLOPs Utilization (%)")
print("  Mem/GPU = GPU Memory Allocated")
print("\n")
EOF
```

### Plot MFU Over Time

```bash
# Generate plot for latest run
cd plots
python plot_b200_run.py ../out-llama-1.36b/run_*.json
cd ..

# View plot
ls -lh out-llama-1.36b/*_b200_analysis.png
```

---

## 📊 Expected Results Summary

| Test | Configuration | Expected MFU | Tokens/sec | Notes |
|------|--------------|--------------|------------|-------|
| **Baseline (2 GPU)** |
| 1.1 | Pure DDP, no compile | 40-50% | ~40K | Baseline |
| 1.2 | + torch.compile() | 50-60% | ~50K | +20% speedup |
| 1.3 | + DataLoader | 52-62% | ~52K | Slight improvement |
| **Scaling (8 GPU)** |
| 2.1 | Pure DDP, no compile | 35-45% | ~140K | DDP overhead |
| 2.2 | + torch.compile() | 45-55% | ~180K | Compile scales well |
| **Production Tiers** |
| Tier 1 | Conservative | 45-55% | ~180K | **Recommended start** |
| Tier 2 | + DataLoader | 50-60% | ~195K | If CPU-bound |
| Tier 3 | + CUDA Graphs | 55-65% | ~210K | Maximum speed |
| **Ablations** |
| 4.1 | SDPA (no FA2) | 30-40% | ~120K | -20-30% vs FA2 |
| 4.2 | float32 | 20-30% | ~90K | 2× slower |
| 4.3 | grad_accum=1 | 35-45% | ~160K | Frequent DDP sync |
| 4.3 | grad_accum=16 | 50-60% | ~200K | Less sync overhead |
| 4.4 | batch_size=8 | 35-45% | ~140K | Small batch inefficient |
| 4.4 | batch_size=32 | 55-65% | ~220K | Large batch efficient |

**Key Findings:**
- **FlashAttention-2**: 20-30% speedup vs SDPA
- **torch.compile()**: 20% speedup
- **CUDA Graphs**: 5-15% speedup (if stable)
- **DataLoader**: 3-5% speedup (if CPU-bound)
- **Larger batch size**: 10-20% higher MFU (if fits in memory)
- **More gradient accumulation**: 5-10% higher MFU (less DDP sync)

---

## 🔍 Monitoring Commands

```bash
# Terminal 1: Watch GPU usage
watch -n 1 nvidia-smi

# Terminal 2: Watch training logs (last 50 lines)
tail -n 50 -f out-llama-1.36b/run_*.json

# Terminal 3: Live MFU tracking
watch -n 2 'tail -1 out-llama-1.36b/run_*.json | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Iter: {d.get(\"training_iterations\", [{}])[-1].get(\"iter\", \"N/A\")} | MFU: {d.get(\"training_iterations\", [{}])[-1].get(\"mfu\", 0):.2f}% | Loss: {d.get(\"training_iterations\", [{}])[-1].get(\"loss\", 0):.4f}\") if d.get(\"training_iterations\") else print(\"Waiting...\")"'

# Check system resources
htop  # CPU usage
iotop # Disk I/O
```

---

## 🐛 Troubleshooting

### OOM (Out of Memory)

```bash
# Reduce batch size
--batch_size=16

# Or enable ZeRO-1 (slight MFU penalty)
--use_zero1=True
```

### CUDA Graph Capture Failed

```bash
# Disable CUDA Graphs (fallback to Tier 2)
--use_cuda_graphs=False
```

### NVLink Hardware Errors

```bash
# Start without torch.compile()
--compile=False

# If persists, contact admin
```

### Low MFU (<30%)

**Check:**
1. GPU utilization: `nvidia-smi`
2. CPU utilization: `htop`
3. Disk I/O: `iotop`

**Common causes:**
- Batch size too small → increase batch_size
- Frequent DDP sync → increase gradient_accumulation_steps
- CPU bottleneck → enable DataLoader
- Disk I/O bottleneck → move data to faster storage

---

## 📝 Notes

1. **Always skip first 50 iterations** when calculating average MFU (warmup period)
2. **torch.compile()** takes 2-5 minutes on first run (one-time cost)
3. **CUDA Graphs** may not work with all configurations (experimental)
4. **Save checkpoint disabled** for testing speed (use `--always_save_checkpoint=True` for real training)
5. **Gradient accumulation** is now **per-GPU** (not divided by world size)
6. **MFU calculation** is now **global** (total throughput vs total peak FLOPS)

---

## 🎯 Recommended Testing Workflow

```bash
# 1. Start with 2 GPU baseline
bash -c "$(sed -n '/Test 1.1/,/^# Expected/p' docs/39_b200_mfu_testing_guide.md | grep -A 20 'torchrun')"

# 2. Scale to 8 GPUs (Tier 1)
bash -c "$(sed -n '/Tier 1:/,/^# Optimizations/p' docs/39_b200_mfu_testing_guide.md | grep -A 20 'torchrun')"

# 3. Try Tier 2 if stable
bash -c "$(sed -n '/Tier 2:/,/^# Additional/p' docs/39_b200_mfu_testing_guide.md | grep -A 20 'torchrun')"

# 4. Analyze results
python << 'EOF'
# (Use analysis script from Section 6)
EOF

# 5. If MFU looks good, try Tier 3
# 6. Run ablation studies to understand bottlenecks
```

---

**Questions?** Refer to main `TRAINING_GUIDE.md` or `docs/34_b200_mfu_optimization_implementation.md`

