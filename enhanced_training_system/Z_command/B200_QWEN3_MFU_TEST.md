# B200 Qwen3 1.8B MFU Testing Guide

**Model**: Qwen3 1.8B (24L-16H-2048D-GQA)  
**Hardware**: 8× NVIDIA B200 (192GB each, 2,250 TFLOPS dense BF16)  
**Config**: `config/full_qwen3_1.8b_b200_optimal.py`  
**Strategy**: Progressive tuning from conservative → optimal

---

## Quick Reference

| Phase | Batch Size | Grad Accum | CUDA Graphs | Expected MFU | Expected Tokens/s |
|-------|------------|------------|-------------|--------------|-------------------|
| **Phase 1** | 64 | 4 | ❌ | 35-40% | 250,000-300,000 |
| **Phase 2** | 96 | 2 | ❌ | 40-45% | 300,000-350,000 |
| **Phase 3** | 128 | 2 | ✅ | 50-55% | 400,000-450,000 |

**Goal**: Verify each phase before moving to next. Only proceed if stable (no OOM, no errors).

---

## Phase 1: Conservative Validation (Start Here)

**Config**: Already set in `full_qwen3_1.8b_b200_optimal.py`
- `batch_size = 64`
- `gradient_accumulation_steps = 4`
- `use_cuda_graphs = False`
- `use_dataloader = True`, `dataloader_num_workers = 16`

### Quick Test (200 iters, ~5-10 minutes)

```bash
cd /home/zhf004/llm_TII/enhanced_training_system

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=200 \
  --batch_size=22 \
  --gradient_accumulation_steps=2 \
  --compile=True \
  --always_save_checkpoint=False
```

### 🔬 MFU Version Comparison Test (Optional)

Test all three MFU calculation methods with identical parameters to compare:

**Version 1: Legacy nanoGPT** (6N heuristic)  
_Uses PaLM's parameter-count formula: FLOPs = 6N + 12LHQT where N is total parameters (ignores GQA structure)._

```bash
cd /home/zhf004/llm_TII/enhanced_training_system
cp mfu_versions/model_builder_v1.py model_builder.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=200 \
  --batch_size=22 \
  --gradient_accumulation_steps=2 \
  --compile=True \
  --always_save_checkpoint=False

# Check logs for: "calculation_method": "nanogpt_legacy_v1"
# Expected MFU: ~52% (overestimated - no GQA correction)
```

**Version 2: Combined Formula** (algebraic)  
_Uses component summation with algebraic GQA formula: Attention = 2SH²(2 + 2/G) + 4S²H, where G = n_head/n_kv_head._

```bash
cd /home/zhf004/llm_TII/enhanced_training_system
cp mfu_versions/model_builder_v2.py model_builder.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=200 \
  --batch_size=22 \
  --gradient_accumulation_steps=2 \
  --compile=True \
  --always_save_checkpoint=False

# Check logs for: "calculation_method": "combined_formula_v2"
# Expected MFU: ~44% (correct)
# Expected attention_to_ffn_ratio: 2.50 (inflated)
```

**Version 3: Gold Standard** (explicit) ⭐ **RECOMMENDED**  
_Uses explicit component breakdown: Q=2SH², K=2SH²/G, V=2SH²/G, O=2SH², summing all projections separately for maximum clarity._

```bash
cd /home/zhf004/llm_TII/enhanced_training_system
cp mfu_versions/model_builder_v3.py model_builder.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=200 \
  --batch_size=22 \
  --gradient_accumulation_steps=2 \
  --compile=True \
  --always_save_checkpoint=False

# Check logs for: "calculation_method": "component_summation_v2025"
# Expected MFU: ~44% (correct)
# Expected attention_to_ffn_ratio: 0.50 (correct!)
```

**Comparison Summary**:

| Version | MFU % | Attn/FFN Ratio | Method ID | Status |
|---------|-------|----------------|-----------|---------|
| v1 | ~52% | N/A | `nanogpt_legacy_v1` | ❌ Overestimated |
| v2 | ~44% | 2.50 | `combined_formula_v2` | ⚠️ MFU ok, ratio inflated |
| v3 | ~44% | 0.50 | `component_summation_v2025` | ✅ All correct |

**After testing, restore v3 (recommended)**:
```bash
cp mfu_versions/model_builder_v3.py model_builder.py
```

**Watch for:**
- ✅ MFU: 35-40% (if lower, something's wrong)
- ✅ Memory: ~50-60 GB per GPU (plenty of headroom)
- ✅ Tokens/sec: ~250,000-300,000
- ❌ No OOM errors
- ❌ No NVLink errors

**If successful → Proceed to Phase 2**

---

## Phase 2: Increase Saturation

**Config changes** (edit `full_qwen3_1.8b_b200_optimal.py`):
```python
batch_size = 96                       # Increased from 64
gradient_accumulation_steps = 2       # Reduced from 4
```

### Quick Test (200 iters)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=200 \
  --always_save_checkpoint=False
```

**Expected improvements:**
- MFU: 40-45% (+5-10% over Phase 1)
- Memory: ~65-75 GB per GPU
- Tokens/sec: ~300,000-350,000

**If successful → Proceed to Phase 3**

---

## Phase 3: Maximum Performance

**Config changes**:
```python
batch_size = 128                      # Further increased (optional: can stay at 96)
gradient_accumulation_steps = 2       # Keep at 2
use_cuda_graphs = True                # Enable CUDA Graphs
```

### Quick Test (200 iters)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=200 \
  --always_save_checkpoint=False
```

**Expected improvements:**
- MFU: 50-55% (+10% over Phase 2)
- Memory: ~75-85 GB per GPU
- Tokens/sec: ~400,000-450,000

**⚠️ Note:** CUDA Graphs can be finicky with DDP. If errors occur, keep Phase 2 settings.

---

## Full Training Run (Once Validated)

After finding your optimal phase, run full training on SlimPajama-6B:

```bash
# Example: Using Phase 2 settings (conservative but proven)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=3000 \
  --always_save_checkpoint=True \
  --eval_interval=500
```

**Calculate max_iters for desired tokens:**
```python
# Your current config (Phase 1):
tokens_per_iter = batch_size × grad_accum × num_gpus × seq_len
                = 64 × 4 × 8 × 2048
                = 4,194,304 tokens (~4.2M)

# For 6B dataset:
max_iters = 6_000_000_000 / 4_194_304 ≈ 1,431 iterations

# For testing (1B tokens):
max_iters = 1_000_000_000 / 4_194_304 ≈ 238 iterations

# For full training (82B tokens, Chinchilla optimal):
max_iters = 82_000_000_000 / 4_194_304 ≈ 19,550 iterations
```

**Adjust for your phase:**

| Phase | Tokens/Iter | Iters for 6B | Iters for 82B | Time for 82B |
|-------|-------------|--------------|---------------|--------------|
| Phase 1 (BS=64, Accum=4) | 4.2M | 1,431 | 19,550 | ~6-8 hours |
| Phase 2 (BS=96, Accum=2) | 3.9M | 1,538 | 21,000 | ~5-6 hours |
| Phase 3 (BS=128, Accum=2) | 5.2M | 1,154 | 15,770 | ~4-5 hours |

---

## Monitoring During Training

### GPU Status
```bash
# Watch GPU utilization (should be >95%)
watch -n 1 nvidia-smi

# Detailed stats
nvidia-smi dmon -s pucvmet -d 2
```

### Check MFU in Logs
Look for output like:
```
⚡ MFU: 42.5% │ Achieved: 7650 TF │ Peak: 18000.0 TF
   Tokens/s: 320450 │ FLOPs/token: 23.9 GF
```

**Target MFU by Phase:**
- Phase 1: 35-40%
- Phase 2: 40-45%
- Phase 3: 50-55%

### Memory Usage
```
💾 Memory: 68.2 GB alloc │ 72.5 GB peak │ 78.0 GB reserved
```

Should be well below 192 GB in all phases.

---

## Troubleshooting

### Issue: MFU < 35% (Phase 1)

**Possible causes:**
1. DataLoader not enabled → Check `use_dataloader = True`
2. Too few workers → Check `dataloader_num_workers = 16`
3. CPU bottleneck → Monitor CPU usage with `htop`
4. Slow storage → Check I/O with `iostat -x 2`

### Issue: OOM (Out of Memory)

**Solutions:**
1. Reduce `batch_size` (try 48 or 32)
2. Disable `compile = False`
3. Check for memory leaks (restart training)

### Issue: NVLink Errors

**Solutions:**
1. Disable `compile = False`
2. Disable `use_cuda_graphs = False`
3. Check `nvidia-smi topo -m` for topology issues
4. Contact admin if hardware issue suspected

### Issue: CUDA Graphs Crash

**Solution:**
Keep `use_cuda_graphs = False` and use Phase 2 settings. CUDA Graphs are optional optimization.

---

## Expected Performance Summary

### Phase 1 (Conservative, Current Config)
```
Hardware Peak:  18,000 TFLOPS (8× B200 @ 2,250 TFLOPS each)
Achieved:       ~7,000 TFLOPS
MFU:            35-40%
Tokens/sec:     250,000-300,000
Memory/GPU:     50-60 GB (26-31% of 192 GB)
Stability:      High ✅
```

### Phase 2 (Recommended)
```
Hardware Peak:  18,000 TFLOPS
Achieved:       ~8,000 TFLOPS
MFU:            40-45%
Tokens/sec:     300,000-350,000
Memory/GPU:     65-75 GB (34-39% of 192 GB)
Stability:      High ✅
```

### Phase 3 (Maximum)
```
Hardware Peak:  18,000 TFLOPS
Achieved:       ~9,500 TFLOPS
MFU:            50-55%
Tokens/sec:     400,000-450,000
Memory/GPU:     75-85 GB (39-44% of 192 GB)
Stability:      Medium ⚠️ (CUDA Graphs can be tricky)
```

**Recommendation**: Phase 2 is the sweet spot (40-45% MFU, stable, good throughput).

---

## Quick Commands Cheatsheet

```bash
# Phase 1 Quick Test (200 iters, ~5-10 min)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py --max_iters=200 --always_save_checkpoint=False

# Phase 1 Short Run (1000 iters, ~30-40 min, ~4.2B tokens)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py --max_iters=1000

# Full 6B Dataset (1431 iters, ~2-3 hours)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py --max_iters=1500

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor detailed stats
nvidia-smi dmon -s pucvmet -d 2
```

---

## Next Steps After Testing

1. ✅ **Run Phase 1** (200 iters) to validate stability
2. ✅ **Check MFU** - Should be 35-40%
3. ✅ **If stable** → Try Phase 2 (96/2 config)
4. ✅ **If Phase 2 stable** → Try Phase 3 (128/2 + CUDA Graphs)
5. ✅ **Pick best stable phase** for full training
6. 🎯 **Run full training** on 6B dataset (or 82B for optimal)

**Note**: The goal is NOT to chase maximum MFU at the cost of stability. 40-45% MFU with stable training is better than 55% MFU with crashes.

