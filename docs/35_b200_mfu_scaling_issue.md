# B200 MFU Scaling Issue - Diagnostic Report

**Date:** 2025-11-20  
**System:** DGX B200 (8× NVIDIA B200 GPUs, 192GB each)  
**Model:** LLaMA 1.36B (1,294,159,104 parameters)  
**Dataset:** SlimPajama-6B

## Issue Summary

Significant MFU degradation observed when scaling from 2 GPUs to 8 GPUs, despite identical per-GPU configuration.

## Configuration Details

### Common Settings (Both Runs)
- **Model**: LLaMA 1.36B (18L-18H-2304D-RoPE-RMS-SwiGLU-PreNorm)
- **Attention Backend**: FlashAttention-2
- **Batch Size (per GPU)**: 24
- **Gradient Accumulation (per GPU)**: 64 steps
- **torch.compile()**: Enabled
- **Precision**: bfloat16
- **CUDA Graphs**: Disabled
- **DataLoader**: Disabled

### Per-GPU Workload (Identical)
- Micro-batch size: 24
- Gradient accumulation: 64 steps
- Tokens per micro-batch: 24 × 2048 = 49,152 tokens
- Total tokens per GPU per iteration: 24 × 64 × 2048 = 3,145,728 tokens

## Performance Comparison

### 2 GPU Run

```
🖥️  HARDWARE:
  Device:                NVIDIA B200
  GPUs:                  2
  Memory per GPU:        191.5 GB
  Precision:             bfloat16
  TF32 enabled:          True
  Compile:               True
  Parallelism:           DDP

📈 THEORETICAL PERFORMANCE:
  Hardware peak:         9000.0 TFLOPS (B200 bf16)
  FLOPs per token:       8.78 GFLOPs
  Attention/FFN ratio:   2.50
  Expected tokens/s @50% MFU: 512285

================================================================================
🏁 STARTING TRAINING
================================================================================

Iter 0: loss 10.8422, time 32054ms (warming up...)

────────────────────────────────────────────────────────────────────────────────
📍 Iter     10 │ Loss: 10.7652 │ Time: 1838ms │ LR: 1.65e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 10.44% │ Achieved: 939.7 TF │ Peak: 9000.0 TF
   Tokens/s: 106979 │ FLOPs/token: 8.78 GF (6N+12LHQT: N=1.294B)
💾 Memory: 13.53 GB alloc │ 103.26 GB peak │ 111.23 GB reserved
```

**Key Metrics (2 GPU):**
- **MFU**: 10.44%
- **Achieved TFLOPS**: 939.7
- **Time per iteration**: ~1.8 seconds
- **Tokens/sec**: 106,979
- **Memory usage**: 103.26 GB peak (53.8% of 192 GB)

### 8 GPU Run

```
🖥️  HARDWARE:
  Device:                NVIDIA B200
  GPUs:                  8
  Memory per GPU:        191.5 GB
  Precision:             bfloat16
  TF32 enabled:          True
  Compile:               True
  Parallelism:           DDP

📈 THEORETICAL PERFORMANCE:
  Hardware peak:         36000.0 TFLOPS (B200 bf16)
  FLOPs per token:       8.78 GFLOPs
  Attention/FFN ratio:   2.50
  Expected tokens/s @50% MFU: 2049141

================================================================================
🏁 STARTING TRAINING
================================================================================

Iter 0: loss 10.8414, time 59848ms (warming up...)

────────────────────────────────────────────────────────────────────────────────
📍 Iter     10 │ Loss: 10.7715 │ Time: 29452ms │ LR: 1.65e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 2.61% │ Achieved: 938.2 TF │ Peak: 36000.0 TF
   Tokens/s: 106809 │ FLOPs/token: 8.78 GF (6N+12LHQT: N=1.294B)
💾 Memory: 13.53 GB alloc │ 103.26 GB peak │ 111.23 GB reserved
```

**Key Metrics (8 GPU):**
- **MFU**: 2.61%
- **Achieved TFLOPS**: 938.2
- **Time per iteration**: ~29.5 seconds
- **Tokens/sec**: 106,809
- **Memory usage**: 103.26 GB peak (53.8% of 192 GB)

## Statistical Analysis

### Absolute Compute Performance
| Metric | 2 GPU | 8 GPU | Ratio |
|--------|-------|-------|-------|
| Achieved TFLOPS | 939.7 | 938.2 | 1.00× (identical) |
| Tokens/sec | 106,979 | 106,809 | 1.00× (identical) |
| Time per iteration | 1.84s | 29.45s | **16.0×** |

### Relative Performance (MFU)
| Metric | 2 GPU | 8 GPU | Ratio |
|--------|-------|-------|-------|
| MFU | 10.44% | 2.61% | **0.25×** (4× degradation) |
| Hardware Peak | 9,000 TFLOPS | 36,000 TFLOPS | 4.0× |
| MFU × Peak | 939.6 TFLOPS | 939.6 TFLOPS | 1.00× |

### Memory Utilization
| Metric | 2 GPU | 8 GPU |
|--------|-------|-------|
| Peak allocated | 103.26 GB | 103.26 GB |
| % of total | 53.8% | 53.8% |
| Reserved | 111.23 GB | 111.23 GB |

### Per-Iteration Breakdown
| Phase | 2 GPU | 8 GPU | Expected | Observed/Expected |
|-------|-------|-------|----------|-------------------|
| Iteration time | 1.84s | 29.45s | ~0.46s (4× faster) | **64×** slower |
| Tokens/iteration | 6.3M | 25.2M | Same total | Correct |
| TFLOPS achieved | 940 | 938 | ~3760 (4× more GPUs) | **0.25×** |

## Observations

1. **Identical absolute compute**: Both configurations achieve ~940 TFLOPS total
2. **Identical throughput**: Both process ~107k tokens/second
3. **Identical memory footprint**: Both use 103 GB peak per GPU
4. **16× iteration time increase**: 8-GPU run takes 16× longer per iteration
5. **No GPU scaling**: Adding 4× more GPUs provides 0× speedup
6. **MFU calculation discrepancy**: 
   - 2 GPU: 940 TFLOPS / 9,000 TFLOPS = 10.44% ✓
   - 8 GPU: 940 TFLOPS / 36,000 TFLOPS = 2.61% ✓
   - Both calculations are mathematically correct

## Configuration Verification

### Gradient Accumulation
- **Config setting**: `gradient_accumulation_steps = 512`
- **2 GPU**: 512 / 2 = **64 steps per GPU** ✓
- **8 GPU**: 512 / 8 = **64 steps per GPU** ✓

### Effective Batch Size
- **2 GPU**: 24 × 64 × 2 = 3,072 samples (6.3M tokens)
- **8 GPU**: 24 × 64 × 8 = 12,288 samples (25.2M tokens)

### DDP Configuration
Both runs use:
- `DistributedDataParallel` (DDP)
- NCCL backend
- `model.no_sync()` for gradient accumulation
- Synchronization only on final micro-step

## Environment Details

### Software Stack
- PyTorch: 2.7.0+cu128
- CUDA: 12.8
- FlashAttention: 2.8.3
- Python: 3.12
- Driver: 560.35.03

### Hardware
- GPU: NVIDIA B200 (Blackwell architecture)
- Interconnect: NVLink 5.0 / NVSwitch
- Memory: 192 GB HBM3e per GPU
- Compute: sm_100

## Raw Terminal Logs

### 2 GPU Run - Full Output
```
⚙️  TRAINING CONFIGURATION:
  Batch size (micro):    24
  Gradient accum steps:  64
  Effective batch size:  3072
  Tokens per iteration:  6,291,456
  Max iterations:        1,000
  Learning rate:         0.0003
  Weight decay:          0.1
  Gradient clip:         1.0
  LR warmup iters:       2000
  LR decay iters:        25000

Training:   0%|                                                     | 0/1000 [00:00<?, ?iter/s]
Iter 0: loss 10.8422, time 32054ms (warming up...)
Training:   1%|▌                                                   | 10/1000 [00:48<36:31,  2.21s/iter]
────────────────────────────────────────────────────────────────────────────────
📍 Iter     10 │ Loss: 10.7652 │ Time: 1838ms │ LR: 1.65e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 10.44% │ Achieved: 939.7 TF │ Peak: 9000.0 TF
   Tokens/s: 106979 │ FLOPs/token: 8.78 GF (6N+12LHQT: N=1.294B)
💾 Memory: 13.53 GB alloc │ 103.26 GB peak │ 111.23 GB reserved
```

### 8 GPU Run - Full Output
```
⚙️  TRAINING CONFIGURATION:
  Batch size (micro):    24
  Gradient accum steps:  64
  Effective batch size:  12288
  Tokens per iteration:  25,165,824
  Max iterations:        100
  Learning rate:         0.0003
  Weight decay:          0.1
  Gradient clip:         1.0
  LR warmup iters:       2000
  LR decay iters:        25000

Training:   0%|                                                     | 0/100 [00:00<?, ?iter/s]
Iter 0: loss 10.8414, time 59848ms (warming up...)
Training:  10%|████▍                                       | 10/100 [05:23<44:35, 29.73s/iter]
────────────────────────────────────────────────────────────────────────────────
📍 Iter     10 │ Loss: 10.7715 │ Time: 29452ms │ LR: 1.65e-06
────────────────────────────────────────────────────────────────────────────────
⚡ MFU: 2.61% │ Achieved: 938.2 TF │ Peak: 36000.0 TF
   Tokens/s: 106809 │ FLOPs/token: 8.78 GF (6N+12LHQT: N=1.294B)
💾 Memory: 13.53 GB alloc │ 103.26 GB peak │ 111.23 GB reserved
Training:  12%|█████▎                                      | 12/100 [06:22<43:17, 29.52s/iter]
```

## Next Steps

Further investigation required to determine root cause of scaling failure.

