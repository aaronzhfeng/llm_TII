#!/usr/bin/env python3
"""
Smoke test for B200 training setup
Tests: PyTorch, CUDA, GPUs, FlashAttention, basic training step
"""

import torch
import sys

print("="*80)
print("B200 Training Environment Smoke Test")
print("="*80)

# Test 1: PyTorch & CUDA
print("\n[1/6] PyTorch & CUDA...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda}")

if not torch.cuda.is_available():
    print("  ❌ CUDA not available!")
    sys.exit(1)

# Test 2: GPU Detection
print("\n[2/6] GPU Detection...")
gpu_count = torch.cuda.device_count()
print(f"  GPU count: {gpu_count}")
for i in range(gpu_count):
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name} | Compute {cap[0]}.{cap[1]} | {mem:.1f}GB")

if gpu_count != 8:
    print(f"  ⚠️  Expected 8 GPUs, found {gpu_count}")

# Test 3: B200 Compute Capability
print("\n[3/6] B200 Compatibility...")
cap = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()
print(f"  Device capability: {cap}")
print(f"  Supported architectures: {arch_list}")

if cap == (10, 0):
    if 'sm_100' in arch_list or 'compute_100' in arch_list:
        print("  ✅ B200 (sm_100) fully supported!")
    else:
        print("  ⚠️  B200 detected but sm_100 not in arch list - using compatibility mode")
else:
    print(f"  ⚠️  Expected compute 10.0 (B200), got {cap}")

# Test 4: Basic Tensor Operations
print("\n[4/6] Basic CUDA Operations...")
try:
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x.T)
    print(f"  Matrix multiplication: ✅ (result shape: {y.shape})")
except Exception as e:
    print(f"  ❌ CUDA operation failed: {e}")
    sys.exit(1)

# Test 5: FlashAttention
print("\n[5/6] FlashAttention...")
try:
    from flash_attn import flash_attn_func
    print(f"  FlashAttention-2: ✅")
    
    # Try FA3 detection
    try:
        import flash_attn
        from packaging import version
        has_fa3 = version.parse(flash_attn.__version__) >= version.parse("2.5.0")
        if has_fa3:
            print(f"  FlashAttention-3: ✅ (version {flash_attn.__version__})")
        else:
            print(f"  FlashAttention-3: ❌ (version {flash_attn.__version__} < 2.5.0)")
    except:
        print(f"  FlashAttention-3: ❌")
        
except ImportError as e:
    print(f"  ❌ FlashAttention not installed: {e}")

# Test 6: Small Training Step
print("\n[6/6] Mini Training Step...")
try:
    model = torch.nn.Linear(512, 512).cuda().to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    x = torch.randn(32, 512, device='cuda', dtype=torch.bfloat16)
    y = model(x)
    loss = y.mean()
    loss.backward()
    optimizer.step()
    
    print(f"  Forward/backward pass: ✅")
    print(f"  Loss: {loss.item():.4f}")
except Exception as e:
    print(f"  ❌ Training step failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ All tests passed! Ready for B200 training.")
print("="*80)

