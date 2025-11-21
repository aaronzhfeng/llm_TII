# MFU Testing Guide: Qwen3 1.8B on B200

Quick reference for systematic MFU testing on 8× B200 GPUs.

**Prerequisites:** Dataset prepared, venv activated, PyTorch with B200 support installed.

---

## Setup

```bash
cd /home/zhf004/llm_TII/enhanced_training_system
source ../venv/bin/activate

# Check GPU availability
nvidia-smi
# 2-GPU tests will use GPUs 5 and 6
# 8-GPU tests will use all GPUs (0-7)
```

---

## 1. Baseline (2 GPUs: 5 and 6)

### 1.1 Minimal (No Compile)

```bash
CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc_per_node=2 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --max_iters=200 \
  --batch_size=20 \
  --gradient_accumulation_steps=8 \
  --compile=False \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected: 40-50% MFU
# Using GPUs 5 and 6 (others may be busy)
```

### 1.2 With Compile

```bash
CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc_per_node=2 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --max_iters=200 \
  --batch_size=20 \
  --gradient_accumulation_steps=8 \
  --compile=True \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected: 50-60% MFU (+20% speedup)
# Using GPUs 5 and 6 (others may be busy)
```

---

## 2. Scaling (8 GPUs)

### 2.1 Pure DDP

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --max_iters=200 \
  --batch_size=20 \
  --gradient_accumulation_steps=8 \
  --compile=False \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected: 35-45% MFU (DDP overhead)
```

### 2.2 DDP + Compile

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --max_iters=200 \
  --batch_size=20 \
  --gradient_accumulation_steps=8 \
  --compile=True \
  --use_zero1=False \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected: 45-55% MFU
```

---

## 3. Production Tiers (8 GPUs)

### 🥉 Tier 1: Conservative (Recommended Start)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --max_iters=100 \
  --batch_size=20 \
  --gradient_accumulation_steps=8 \
  --compile=True \
  --use_cuda_graphs=False \
  --use_dataloader=False \
  --always_save_checkpoint=False

# Expected: 45-55% MFU, ~165K tokens/sec
# Most stable, proven configuration
```

### 🥈 Tier 2: Balanced (+ DataLoader)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --max_iters=1000 \
  --batch_size=20 \
  --gradient_accumulation_steps=8 \
  --compile=True \
  --use_cuda_graphs=False \
  --use_dataloader=True \
  --dataloader_num_workers=4 \
  --always_save_checkpoint=False

# Expected: 50-60% MFU, ~180K tokens/sec
# Better CPU efficiency
```

### 🥇 Tier 3: Maximum (+ CUDA Graphs)

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --max_iters=1000 \
  --batch_size=20 \
  --gradient_accumulation_steps=8 \
  --compile=True \
  --use_cuda_graphs=True \
  --use_dataloader=True \
  --dataloader_num_workers=4 \
  --always_save_checkpoint=False

# Expected: 55-65% MFU, ~195K tokens/sec
# Maximum performance (may be unstable)
```

---

## 4. Ablation Studies

### FlashAttention vs SDPA

```bash
# Without FlashAttention
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --attention_backend=sdpa \
  --max_iters=200 \
  --batch_size=20 \
  --compile=True \
  --always_save_checkpoint=False

# Expected: 30-40% MFU (20-30% slower)
```

### Gradient Accumulation Impact

```bash
# Few steps (frequent sync)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --gradient_accumulation_steps=1 \
  --max_iters=200 \
  --batch_size=20 \
  --compile=True \
  --always_save_checkpoint=False

# Expected: 35-45% MFU

# Many steps (less sync)
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --gradient_accumulation_steps=16 \
  --max_iters=200 \
  --batch_size=20 \
  --compile=True \
  --always_save_checkpoint=False

# Expected: 50-60% MFU (5-10% improvement)
```

### Batch Size Impact

```bash
# Small batch
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --batch_size=8 \
  --max_iters=200 \
  --compile=True \
  --always_save_checkpoint=False

# Expected: 35-45% MFU

# Large batch
torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_optimal.py \
  --batch_size=32 \
  --max_iters=200 \
  --compile=True \
  --always_save_checkpoint=False

# Expected: 55-65% MFU (10-20% improvement)
```

---

## 5. Memory Limit Test

```bash
# Find max batch size (test until OOM)
for BS in 24 32 48 64 96; do
  echo "Testing batch_size=$BS"
  torchrun --standalone --nproc_per_node=8 train.py \
    config/full_qwen3_1.8b_optimal.py \
    --batch_size=$BS \
    --max_iters=20 \
    --compile=True \
    --always_save_checkpoint=False
  
  [ $? -ne 0 ] && echo "OOM at $BS" && break
  sleep 5
done
```

---

## 6. Analysis

### Quick Stats (Latest Run)

```bash
python << 'EOF'
import json, glob, os
log = sorted(glob.glob('out-qwen3-1.8b/run_*.json'), key=os.path.getmtime)[-1]
data = json.load(open(log))
iters = data['training_iterations'][50:]  # Skip warmup
avg_mfu = sum(i['mfu'] for i in iters) / len(iters)
avg_tokens = sum(i['tokens_per_sec'] for i in iters) / len(iters)
print(f"Avg MFU: {avg_mfu:.2f}%")
print(f"Avg tokens/sec: {avg_tokens:,.0f}")
EOF
```

### Compare All Runs

```bash
python << 'EOF'
import json, glob, os
logs = sorted(glob.glob('out-qwen3-1.8b/run_*.json'), key=os.path.getmtime, reverse=True)[:5]
print(f"{'Config':<30} {'GPUs':>5} {'BS':>4} {'MFU':>7} {'Tokens/s':>10}")
print("-" * 62)
for log in logs:
    data = json.load(open(log))
    cfg = data['startup_info']['config']
    iters = data['training_iterations'][50:]
    mfu = sum(i['mfu'] for i in iters) / len(iters)
    tok = sum(i['tokens_per_sec'] for i in iters) / len(iters)
    opts = []
    if cfg['compile']: opts.append('compile')
    if cfg['use_dataloader']: opts.append('DL')
    if cfg['use_cuda_graphs']: opts.append('CG')
    name = '+'.join(opts) if opts else 'baseline'
    print(f"{name:<30} {cfg['world_size']:>5} {cfg['batch_size']:>4} {mfu:>6.2f}% {tok:>10,.0f}")
EOF
```

### Plot Results

```bash
cd plots
python plot_b200_run.py ../out-qwen3-1.8b/run_*.json
cd ..
```

---

## Monitoring

```bash
# Terminal 1: GPU usage
watch -n 1 nvidia-smi

# Terminal 2: Training logs
tail -f out-qwen3-1.8b/run_*.json

# Terminal 3: Live MFU
watch -n 2 'tail -1 out-qwen3-1.8b/run_*.json | python3 -c "import sys,json; d=json.load(sys.stdin); it=d[\"training_iterations\"][-1]; print(f\"Iter {it[\"iter\"]} | MFU: {it[\"mfu\"]:.2f}% | Loss: {it[\"loss\"]:.4f}\")"'
```

---

## Expected Results Summary

| Test | Config | MFU | Tokens/sec |
|------|--------|-----|------------|
| **Baseline (2 GPU)** |
| 1.1 | Pure DDP, no compile | 40-50% | ~35K |
| 1.2 | + compile | 50-60% | ~45K |
| **Scaling (8 GPU)** |
| 2.1 | Pure DDP | 35-45% | ~130K |
| 2.2 | + compile | 45-55% | ~165K |
| **Production** |
| Tier 1 | Conservative | 45-55% | ~165K |
| Tier 2 | + DataLoader | 50-60% | ~180K |
| Tier 3 | + CUDA Graphs | 55-65% | ~195K |

**Key Insights:**
- FlashAttention-2: +20-30% vs SDPA
- torch.compile(): +20%
- CUDA Graphs: +5-15%
- DataLoader: +3-5%
- Larger batch: +10-20%
- More grad_accum: +5-10%

**Note:** Qwen3 (1.8B) is ~30% larger than LLaMA2 (1.36B), so slightly lower throughput but similar MFU expected.

---

## Troubleshooting

**OOM:** Reduce `--batch_size` or enable `--use_zero1=True`

**CUDA Graph fails:** Disable `--use_cuda_graphs=False`

**Low MFU (<30%):** Increase batch_size or gradient_accumulation_steps

**NVLink errors:** Try `--compile=False`

---

**Recommended workflow:** Test 1.1 → 2.1 → Tier 1 → Tier 2 → Ablations

