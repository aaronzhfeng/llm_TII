# 36 — MFU Calculation Update

## What changed
- MFU now uses global tokens/FLOPs when comparing to global hardware peak.
- The estimator returns both global and per-GPU throughput for clarity.
- Logging calls send global sequence counts instead of per-rank counts.

## Original formula (per-rank numerator vs global denominator)
```python
# tokens_per_iter (per rank)
tokens_per_iter = S * fwdbwd_per_iter
flops_per_iter = training_flops_per_token * tokens_per_iter
flops_achieved = flops_per_iter / dt          # per rank
hardware_peak_flops = hardware_peak_flops_per_gpu * num_gpus
mfu = flops_achieved / hardware_peak_flops    # shrinks with num_gpus
```

## New formula (global numerator matches global denominator)
```python
# fwdbwd_per_iter is global sequences = micro_batch * grad_accum_per_gpu * world_size
tokens_per_iter = S * fwdbwd_per_iter
flops_per_iter = training_flops_per_token * tokens_per_iter
flops_achieved = flops_per_iter / dt              # global
flops_achieved_per_gpu = flops_achieved / num_gpus
tokens_per_sec = tokens_per_iter / dt             # global
tokens_per_sec_per_gpu = tokens_per_sec / num_gpus
hardware_peak_flops = hardware_peak_flops_per_gpu * num_gpus
mfu = flops_achieved / hardware_peak_flops
```

## Code reference
- `enhanced_training_system/model_builder.py`: `estimate_mfu_detailed` now computes global throughput and adds per-GPU fields.
- `enhanced_training_system/model.py`: legacy estimator updated the same way.
- `enhanced_training_system/train.py`: MFU calls now pass global sequence counts (`batch_size * grad_accum_per_gpu * world_size`) so numerator and denominator are aligned.
