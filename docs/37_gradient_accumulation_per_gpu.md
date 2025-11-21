# 37 — Gradient Accumulation Semantics Update

## What changed
- `gradient_accumulation_steps` is now interpreted per GPU (no automatic division by world size).
- Explicit fields are logged for both per-GPU and global accumulation counts.
- Tokens-per-iteration uses the per-GPU value multiplied by world size for clarity.

## Original logic (auto-divided)
```python
if ddp:
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size  # per-GPU steps hidden
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
```

## New logic (per-GPU explicit)
```python
if ddp:
    gradient_accumulation_steps_per_gpu = gradient_accumulation_steps
else:
    gradient_accumulation_steps_per_gpu = gradient_accumulation_steps
gradient_accumulation_steps_global = gradient_accumulation_steps_per_gpu * ddp_world_size
tokens_per_iter = gradient_accumulation_steps_per_gpu * ddp_world_size * batch_size * block_size
config['gradient_accumulation_steps_per_gpu'] = gradient_accumulation_steps_per_gpu
config['gradient_accumulation_steps_global'] = gradient_accumulation_steps_global
```

## Impact
- CLI/config now specifies per-GPU accumulation directly; scaling across GPU counts no longer changes the per-GPU workload implicitly.
- Logs and startup prints show both per-GPU and global values, preventing misreads during scaling studies.
