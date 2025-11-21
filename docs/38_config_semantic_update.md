# 38 — Configuration File Updates for New Semantics

**Date:** 2025-11-20  
**Related Documents:** [36_mfu_global_calculation_update.md](./36_mfu_global_calculation_update.md), [37_gradient_accumulation_per_gpu.md](./37_gradient_accumulation_per_gpu.md)

## Summary

Updated configuration files to align with the new gradient accumulation semantics (per-GPU instead of global).

## Changes Made

### LLaMA 1.36B Config (`config/full_llama_1.36b.py`)

**Before:**
```python
gradient_accumulation_steps = 512  # Total gradient accumulation steps (divided by num_gpus in DDP)
                                    # For 8 GPUs: 512 / 8 = 64 steps per GPU
                                    # For 2 GPUs: 128 / 2 = 64 steps per GPU
```

**After:**
```python
gradient_accumulation_steps = 64   # Gradient accumulation steps PER GPU
                                    # Global steps = 64 * num_gpus
                                    # For 8 GPUs: 64 per GPU × 8 = 512 global steps
                                    # For 2 GPUs: 64 per GPU × 2 = 128 global steps
```

**Impact:**
- Same effective workload (64 gradient accumulation steps per GPU)
- Now explicit: value is per-GPU, scaled by world_size for global count
- No change to actual training behavior

### Qwen3 1.8B Config (`config/full_qwen3_1.8b_optimal.py`)

**Before:**
```python
gradient_accumulation_steps = 128   # Total gradient accumulation steps (divided by num_gpus in DDP)
                                     # For 8 GPUs: 128 / 8 = 16 steps per GPU
                                     # For 2 GPUs: Set to 32 for 16 steps per GPU
```

**After:**
```python
gradient_accumulation_steps = 16    # Gradient accumulation steps PER GPU
                                     # Global steps = 16 * num_gpus
                                     # For 8 GPUs: 16 per GPU × 8 = 128 global steps
                                     # For 2 GPUs: 16 per GPU × 2 = 32 global steps
```

**Impact:**
- Same effective workload (16 gradient accumulation steps per GPU)
- Now explicit: value is per-GPU, scaled by world_size for global count
- No change to actual training behavior

## Rationale

### Old Semantics (Pre-Document 37)
```python
# Config: gradient_accumulation_steps = 512
if ddp:
    gradient_accumulation_steps //= ddp_world_size  # Automatic division
# Result: 512 / 8 = 64 per GPU (hidden from user)
```

**Problems:**
- User specified 512 but got 64 per GPU (implicit division)
- Scaling from 2 → 8 GPUs required changing config (512 → 2048) to maintain per-GPU workload
- Confusing when comparing runs with different GPU counts

### New Semantics (Post-Document 37)
```python
# Config: gradient_accumulation_steps = 64
gradient_accumulation_steps_per_gpu = gradient_accumulation_steps  # No division
gradient_accumulation_steps_global = gradient_accumulation_steps_per_gpu * ddp_world_size
# Result: 64 per GPU (explicit, as specified)
```

**Benefits:**
- User specifies exactly what they want per GPU
- Scaling from 2 → 8 GPUs requires NO config changes
- Clear separation between per-GPU and global values in logs
- Consistent with how `batch_size` works (always per-GPU)

## Verification

### Expected Training Logs (8 GPUs)

**Old format:**
```
Gradient accum steps:  64
```

**New format (with Document 37 changes):**
```
Gradient accum steps:  64 per GPU │ 512 global
```

### Effective Batch Size Calculation

**LLaMA 1.36B (8 GPUs):**
- Batch size per GPU: 8
- Gradient accumulation per GPU: 64
- Sequence length: 2048
- World size: 8
- **Effective batch**: 8 × 64 × 8 = 4,096 samples = 8,388,608 tokens

**Qwen3 1.8B (8 GPUs):**
- Batch size per GPU: 6
- Gradient accumulation per GPU: 16
- Sequence length: 2048
- World size: 8
- **Effective batch**: 6 × 16 × 8 = 768 samples = 1,572,864 tokens

## Migration Guide

If you have custom configs using the old semantics:

1. **Identify your target per-GPU gradient accumulation**
   - Old: `gradient_accumulation_steps = N * num_gpus`
   - Target per-GPU: `N`

2. **Update config value**
   ```python
   # Old (for 8 GPUs targeting 64 per GPU):
   gradient_accumulation_steps = 512
   
   # New (explicitly 64 per GPU):
   gradient_accumulation_steps = 64
   ```

3. **Update comments**
   - Remove references to "divided by num_gpus"
   - Add explicit per-GPU and global examples

## Compatibility

**No breaking changes to existing trained models or checkpoints.**

The actual training loop behavior is identical:
- Same number of gradient accumulation steps per GPU
- Same synchronization behavior
- Same effective batch size
- Same tokens per iteration

Only the **config interpretation** changed, not the underlying math.

