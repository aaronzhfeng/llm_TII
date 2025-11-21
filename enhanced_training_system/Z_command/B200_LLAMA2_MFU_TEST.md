# B200 LLaMA 1.36B MFU Testing Guide

**Model**: LLaMA 1.36B (18L-18H-2304D-6144ff)  
**Hardware**: 8× NVIDIA B200 (192GB each, 2,250 TFLOPS dense BF16)  
**Config**: `config/full_llama2_1.36b_b200_optimal.py`  
**Strategy**: Progressive tuning from conservative → optimal

---

## Quick Reference

| Phase | Batch Size | Grad Accum | CUDA Graphs | Expected MFU | Expected Tokens/s |
|-------|------------|------------|-------------|--------------|-------------------|
| **Phase 1** | 64 | 4 | ❌ | 38-42% | 280,000-320,000 |
| **Phase 2** | 96 | 2 | ❌ | 42-48% | 320,000-370,000 |
| **Phase 3** | 128 | 2 | ✅ | 52-58% | 420,000-480,000 |

**Note**: LLaMA 1.36B has lower FLOPs/token than Qwen3 1.8B (18 layers vs 24), so expect ~3-5% higher MFU for same throughput.

---

## Phase 1: Conservative Validation (Start Here)

**Config**: Already set in `full_llama2_1.36b_b200_optimal.py`

### Quick Test (200 iters, ~5-10 minutes)

```bash
cd /home/zhf004/llm_TII/enhanced_training_system

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama2_1.36b_b200_optimal.py \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=2 \
  --compile=True \
  --always_save_checkpoint=False
```

**Watch for:**
- ✅ MFU: 38-42%
- ✅ Memory: ~45-55 GB per GPU
- ✅ Tokens/sec: ~280,000-320,000

---

## Phase 2: Increase Saturation

**Config changes**:
```python
batch_size = 96
gradient_accumulation_steps = 2
```

### Quick Test

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama2_1.36b_b200_optimal.py \
  --max_iters=200 \
  --always_save_checkpoint=False
```

**Expected**: MFU 42-48%, Memory ~60-70 GB

---

## Phase 3: Maximum Performance

**Config changes**:
```python
batch_size = 128
gradient_accumulation_steps = 2
use_cuda_graphs = True
```

### Quick Test

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama2_1.36b_b200_optimal.py \
  --max_iters=200 \
  --always_save_checkpoint=False
```

**Expected**: MFU 52-58%, Memory ~70-80 GB

---

## Iteration Calculations

```python
# Phase 1 (BS=64, Accum=4):
tokens_per_iter = 64 × 4 × 8 × 2048 = 4,194,304 (~4.2M)

# For 6B dataset:
max_iters = 6,000,000,000 / 4,194,304 ≈ 1,431 iterations

# For 85B (Chinchilla optimal for 1.36B):
max_iters = 85,000,000,000 / 4,194,304 ≈ 20,265 iterations
```

---

## Quick Commands

```bash
# Phase 1 Quick Test (200 iters)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama2_1.36b_b200_optimal.py --max_iters=200 --always_save_checkpoint=False

# Full 6B Dataset
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_llama2_1.36b_b200_optimal.py --max_iters=1500
```

---

## Expected Performance Summary

### Phase 1 (Conservative)
- MFU: 38-42%
- Tokens/sec: 280,000-320,000
- Memory: 45-55 GB/GPU
- Stability: High ✅

### Phase 2 (Recommended)
- MFU: 42-48%
- Tokens/sec: 320,000-370,000
- Memory: 60-70 GB/GPU
- Stability: High ✅

### Phase 3 (Maximum)
- MFU: 52-58%
- Tokens/sec: 420,000-480,000
- Memory: 70-80 GB/GPU
- Stability: Medium ⚠️

