# Qwen3 Implementation Plan

**Date:** November 13, 2025  
**Phase:** Architecture Extension - Implementation Roadmap  
**Status:** Planning

---

## Overview

This document outlines the complete implementation plan for adding Qwen3 architecture support to our modular training system, including tokenizer integration, configuration files, dataset preparation, and training workflows.

---

## 1. Implementation Goals

### Primary Objectives

1. **Add Qwen3 architecture support** to the modular system
2. **Integrate Qwen3 tokenizer** (151K vocab, BBPE)
3. **Create configuration files** for 0.6B, 1.5B, and 2B variants
4. **Prepare datasets** with Qwen3 tokenization
5. **Validate training** on 2× A6000 (testing) and document for DGX B200 (production)
6. **Compare performance** with GPT-2, LLaMA 2, and LLaMA 3 architectures

### Non-Goals (Out of Scope)

- ❌ Chain-of-thought fine-tuning (separate phase if desired)
- ❌ Multi-stage training pipelines (use single-stage initially)
- ❌ MoE (Mixture of Experts) variants (Qwen3-MoE is separate)
- ❌ Long-context training >32K (can reduce to 2-8K for testing)

---

## 2. Architecture Implementation

### 2.1 Extended RoPE Support

**Current Status:**
- ✅ RoPE already implemented in `model_components.py`
- ⚠️ Need to verify `rope_theta` parameter is exposed and configurable

**Required Changes:**

```python
# In model_components.py: RoPEPositionEncoding

class RoPEPositionEncoding(nn.Module):
    def __init__(self, dim, max_seq_len, theta=10000.0):  # Add theta parameter
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta  # Make configurable
        
        # Compute position encodings with configurable theta
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        # ... rest of implementation
```

**Validation:**
- [ ] Test with `rope_theta=10000` (LLaMA 2 default)
- [ ] Test with `rope_theta=500000` (LLaMA 3)
- [ ] Test with `rope_theta=1000000` (Qwen3)
- [ ] Verify no numerical instability at high theta values

### 2.2 SwiGLU 3× Expansion Ratio

**Current Status:**
- ✅ SwiGLU already implemented in `model_components.py`
- ✅ Expansion ratio is configurable via `d_ff` parameter

**Required Changes:**
- ✅ No code changes needed
- [ ] Document 3× ratio as Qwen3 default in configs

**Note:** Our SwiGLU uses classic 8/3 formula by default, but `d_ff` override allows any ratio.

### 2.3 GQA 2:1 Ratio Verification

**Current Status:**
- ✅ GQA implemented in `model_builder.py`
- ✅ Tested with 4:1 ratio (LLaMA 3: 32 Q heads, 8 KV heads)

**Required Testing:**
- [ ] Verify 2:1 ratio works (16 Q heads, 8 KV heads)
- [ ] Test parameter counting is correct
- [ ] Test FLOPs calculation is accurate
- [ ] Verify memory usage is as expected

### 2.4 Qwen3 Preset Configuration

**Required:** Add `get_qwen3_style_config()` to `model_config.py`

```python
def get_qwen3_style_config() -> 'ModelArchitectureConfig':
    """
    Qwen3 architecture preset (based on Qwen3-0.6B).
    
    Specifications:
    - 28 layers, 1024 hidden, 16 Q heads, 8 KV heads (2:1 GQA)
    - RMSNorm (pre-norm), SwiGLU (3× expansion), RoPE (theta=1M)
    - 151K vocabulary, no bias, no weight tying
    - Context: 32K (can reduce for training)
    """
    return ModelArchitectureConfig(
        # Model dimensions
        n_layer=28,
        n_head=16,
        n_embd=1024,
        num_key_value_heads=8,  # GQA 2:1 ratio
        block_size=32768,       # Can override to 2048 for testing
        vocab_size=151936,      # Qwen3 tokenizer
        
        # Architecture choices
        normalization='rmsnorm',
        activation='silu',
        position_encoding='rope',
        norm_position='pre',
        ffn_type='swiglu',
        attention_backend='flash_attn_2',  # or 'sdpa'
        
        # Component options
        bias=False,
        weight_tying=False,     # Qwen doesn't tie embeddings
        dropout=0.0,
        
        # FFN dimension (3× expansion for Qwen3)
        d_ff=3072,  # 3 × 1024
        
        # RoPE configuration
        rope_theta=1_000_000,   # Extended RoPE base
        
        # Normalization epsilon
        norm_eps=1e-6,
    )
```

**Update:** Add 'qwen3' to `PRESET_CONFIGS` dictionary and `get_preset_config()` function.

---

## 3. Tokenizer Integration

### 3.1 Download and Save Tokenizer

**Implementation:**

```bash
# Script: data/download_qwen3_tokenizer.py

from transformers import AutoTokenizer
import os

def download_qwen3_tokenizer(save_path="./qwen3_tokenizer"):
    """Download and save Qwen3 tokenizer locally."""
    print("Downloading Qwen3 tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        trust_remote_code=True  # May be needed for custom tokenizers
    )
    
    # Save locally
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    
    print(f"✓ Saved Qwen3 tokenizer to {save_path}/")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Special tokens: {tokenizer.special_tokens_map}")
    
    return tokenizer

if __name__ == "__main__":
    download_qwen3_tokenizer()
```

**Expected Output:**
```
Vocabulary size: 151936
Special tokens: 
  - bos_token: <|endoftext|>
  - eos_token: <|endoftext|>
  - pad_token: <|endoftext|>
  - unk_token: <|endoftext|>
```

### 3.2 Dataset Preparation

**Create:** `data/slimpajama_6b_qwen3/prepare.py`

**Key Differences from LLaMA 3:**
- Vocabulary: 151,936 (vs 128,256 for LLaMA 3)
- Token IDs: Need `uint32` (vs `uint16` which only goes to 65K)
- Special tokens: Different from LLaMA format

**Implementation Template:**

```python
# data/slimpajama_6b_qwen3/prepare.py

import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Load Qwen3 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "../../qwen3_tokenizer",  # Local saved tokenizer
    trust_remote_code=True
)

print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
assert tokenizer.vocab_size == 151936, "Unexpected vocab size"

# Load dataset
print("Loading SlimPajama-6B dataset...")
dataset = load_dataset("DKYoon/SlimPajama-6B", split="train")

# Split into train/validation
print("Splitting dataset...")
split_dataset = dataset.train_test_split(test_size=0.005, seed=42)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

# Tokenize function
def tokenize_function(examples):
    # Qwen3 uses same format as other models
    outputs = tokenizer(
        examples['text'],
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    return {'input_ids': outputs['input_ids']}

# Process datasets
print("Tokenizing train dataset...")
train_tokenized = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train",
)

print("Tokenizing validation dataset...")
val_tokenized = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation",
)

# Flatten and save as binary
def save_to_bin(dataset, filename):
    """Save tokenized dataset to binary file."""
    all_tokens = []
    for example in tqdm(dataset, desc=f"Flattening {filename}"):
        all_tokens.extend(example['input_ids'])
    
    # Convert to numpy array (IMPORTANT: uint32 for 151K vocab)
    arr = np.array(all_tokens, dtype=np.uint32)
    
    print(f"Saving {filename}: {len(arr):,} tokens ({arr.nbytes / 1e9:.2f} GB)")
    arr.tofile(filename)
    
    return len(arr)

# Save train and validation
train_tokens = save_to_bin(train_tokenized, "train.bin")
val_tokens = save_to_bin(val_tokenized, "val.bin")

# Save metadata
import pickle
meta = {
    'vocab_size': tokenizer.vocab_size,
    'tokenizer': 'Qwen3-BBPE',
    'train_tokens': train_tokens,
    'val_tokens': val_tokens,
    'dtype': 'uint32',  # Important for >65K vocab
}

with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("✓ Dataset preparation complete!")
print(f"  Train: {train_tokens:,} tokens")
print(f"  Validation: {val_tokens:,} tokens")
print(f"  Vocabulary: {tokenizer.vocab_size:,}")
```

**Storage Requirements:**
- SlimPajama-6B with Qwen3 tokenizer:
  - Train: ~6 GB (uint32 format)
  - Val: ~30 MB (uint32 format)
  - Total: ~6.03 GB

**Time Estimate:** 20-40 minutes (similar to LLaMA 3)

### 3.3 Create Larger Dataset (Optional)

**For production training:** `data/slimpajama_627b_qwen3/`

- Same process, larger dataset
- Storage: ~1.2 TB (uint32 format)
- Time: 10-20 hours preparation

---

## 4. Configuration Files

### 4.1 Qwen3-0.6B Baseline (`config/full_qwen3_0.6b.py`)

**Purpose:** Exact reproduction of official Qwen3-0.6B for validation

```python
"""
Qwen3-0.6B - Baseline Validation Configuration
================================================

Official Qwen3-0.6B architecture for validation testing.
"""

# === ARCHITECTURE ===
arch_preset = 'qwen3'  # Use Qwen3 preset

# Model dimensions (official Qwen3-0.6B)
n_layer = 28
n_head = 16
n_embd = 1024
num_key_value_heads = 8
d_ff = 3072
block_size = 2048           # Reduced from 32K for testing
vocab_size = 151936
rope_theta = 1_000_000
dropout = 0.0
bias = False

# === TRAINING ===
dataset = 'slimpajama_6b_qwen3'
batch_size = 8
gradient_accumulation_steps = 16
max_iters = 25000           # ~25B tokens (6.5B tokens total)
learning_rate = 3e-4
weight_decay = 1e-1

# === SYSTEM ===
dtype = 'bfloat16'
compile = True
use_zero1 = True            # For 2× A6000 testing
use_fsdp = False

# === OUTPUT ===
out_dir = 'out-qwen3-0.6b'
eval_interval = 1000
log_interval = 10
always_save_checkpoint = False
```

**Expected Performance (2× A6000, ZeRO-1):**
- Memory: ~32-36 GB/GPU
- MFU: 38-42%
- Tokens/sec: ~9,000-11,000
- Time for 6B tokens: ~7-8 hours

### 4.2 Qwen3-1.5B Scaled (`config/full_qwen3_1.5b.py`)

**Purpose:** Scaled-up Qwen3-style model for main experiments

```python
"""
Qwen3-1.5B - Scaled Configuration
===================================

Qwen3-style architecture scaled to ~1.5B parameters.
"""

# === ARCHITECTURE ===
arch_preset = 'qwen3'

# Scaled dimensions
n_layer = 28                # Keep depth
n_head = 16                 # Keep head count
n_embd = 2048               # Scale width: 1024 → 2048
num_key_value_heads = 8     # Keep GQA ratio 2:1
d_ff = 6144                 # Scale FFN: 3 × 2048
block_size = 2048
vocab_size = 151936
rope_theta = 1_000_000
dropout = 0.0
bias = False

# === TRAINING ===
dataset = 'slimpajama_6b_qwen3'
batch_size = 6
gradient_accumulation_steps = 32
max_iters = 25000
learning_rate = 3e-4
weight_decay = 1e-1

# === SYSTEM ===
dtype = 'bfloat16'
compile = True
use_zero1 = True
use_fsdp = False

# === OUTPUT ===
out_dir = 'out-qwen3-1.5b'
eval_interval = 1000
log_interval = 10
always_save_checkpoint = False
```

**Expected Performance (2× A6000, ZeRO-1):**
- Parameters: ~1.5B
- Memory: ~36-40 GB/GPU
- MFU: 40-45%
- Tokens/sec: ~8,000-10,000
- Time for 6B tokens: ~8-10 hours

### 4.3 Qwen3-2B Variant (`config/full_qwen3_2.0b.py`)

**Purpose:** Larger variant to test scaling (optional)

```python
"""
Qwen3-2B - Large Variant Configuration
========================================

Qwen3-style architecture scaled to ~2B parameters.
"""

# === ARCHITECTURE ===
arch_preset = 'qwen3'

# Larger dimensions
n_layer = 28
n_head = 16
n_embd = 2304               # Scale to 2304
num_key_value_heads = 8
d_ff = 6912                 # 3 × 2304
block_size = 2048
vocab_size = 151936
rope_theta = 1_000_000
dropout = 0.0
bias = False

# === TRAINING ===
dataset = 'slimpajama_6b_qwen3'
batch_size = 4              # Reduced for memory
gradient_accumulation_steps = 32
max_iters = 25000
learning_rate = 3e-4
weight_decay = 1e-1

# === SYSTEM ===
dtype = 'bfloat16'
compile = True
use_zero1 = True
use_fsdp = False

# === OUTPUT ===
out_dir = 'out-qwen3-2.0b'
eval_interval = 1000
log_interval = 10
always_save_checkpoint = False
```

**Expected Performance (2× A6000, ZeRO-1):**
- Parameters: ~2.0B
- Memory: ~42-46 GB/GPU (tight!)
- MFU: 35-40%
- Tokens/sec: ~6,000-8,000
- Time for 6B tokens: ~10-12 hours

**Warning:** May hit OOM on 2× A6000. Consider:
- Reducing `batch_size` to 2-3
- Using `compile=False`
- Or use FSDP with 4+ GPUs

---

## 5. Implementation Checklist

### Phase 1: Core Architecture Support

**Model Components:**
- [ ] Verify `rope_theta` parameter is exposed in `RoPEPositionEncoding`
- [ ] Add `rope_theta` to `ModelArchitectureConfig` dataclass
- [ ] Create `get_qwen3_style_config()` preset function
- [ ] Add 'qwen3' to `PRESET_CONFIGS` dictionary
- [ ] Update `get_preset_config()` to handle 'qwen3'

**Testing:**
- [ ] Test Qwen3 preset loads without errors
- [ ] Verify GQA 2:1 ratio (16:8) works correctly
- [ ] Verify parameter count matches expectations
- [ ] Test with different `rope_theta` values

### Phase 2: Tokenizer Integration

**Download and Setup:**
- [ ] Create `data/download_qwen3_tokenizer.py` script
- [ ] Run script to download Qwen3 tokenizer locally
- [ ] Verify vocabulary size is 151,936
- [ ] Document special tokens and format

**Dataset Preparation:**
- [ ] Create `data/slimpajama_6b_qwen3/` directory
- [ ] Create `data/slimpajama_6b_qwen3/prepare.py` script
- [ ] Run dataset preparation (20-40 minutes)
- [ ] Verify train.bin and val.bin files created
- [ ] Check meta.pkl contains correct vocabulary size
- [ ] Create `data/slimpajama_6b_qwen3/README.md`

### Phase 3: Configuration Files

**Create Config Files:**
- [ ] `config/full_qwen3_0.6b.py` (baseline validation)
- [ ] `config/full_qwen3_1.5b.py` (main experiment)
- [ ] `config/full_qwen3_2.0b.py` (optional large variant)

**Verify Configs:**
- [ ] Test each config loads without errors
- [ ] Check parameter counts are as expected
- [ ] Verify dataset paths are correct

### Phase 4: Training Validation

**Smoke Tests (10 iterations):**
- [ ] Test Qwen3-0.6B: `python train.py config/full_qwen3_0.6b.py --max_iters=10 --compile=False`
- [ ] Test Qwen3-1.5B: `python train.py config/full_qwen3_1.5b.py --max_iters=10 --compile=False`
- [ ] Verify no errors, memory usage reasonable
- [ ] Check MFU is in expected range

**Short Training Runs (100 iterations):**
- [ ] Qwen3-0.6B: 100 iterations (~30 minutes)
- [ ] Qwen3-1.5B: 100 iterations (~30 minutes)
- [ ] Monitor loss, gradients, memory
- [ ] Compare with GPT-2/LLaMA baselines

**Full Training Runs (2000 iterations):**
- [ ] Qwen3-0.6B: 2000 iterations (~7-8 hours)
- [ ] Qwen3-1.5B: 2000 iterations (~8-10 hours)
- [ ] Collect comprehensive metrics
- [ ] Generate comparison plots

### Phase 5: Documentation Updates

**Update Core Docs:**
- [ ] Add Qwen3 section to `SYSTEM_OVERVIEW.md`
- [ ] Add Qwen3 commands to `TRAINING_GUIDE.md`
- [ ] Add Qwen3 tests to `TESTING.md`
- [ ] Update architecture comparison tables

**Update docs/README.md:**
- [ ] Add entries for documents 26, 27, 28
- [ ] Update total document count
- [ ] Add to appropriate phase grouping

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Architecture Tests:**
```bash
# Test Qwen3 preset loading
python -c "
from model_config import get_preset_config
config = get_preset_config('qwen3')
print(f'Qwen3 preset: {config.get_architecture_name()}')
assert config.rope_theta == 1_000_000, 'RoPE theta incorrect'
assert config.num_key_value_heads == 8, 'GQA heads incorrect'
print('✓ Qwen3 preset validation passed')
"

# Test GQA 2:1 ratio
python -c "
from model_builder import ConfigurableGPT
from model_config import get_preset_config
config = get_preset_config('qwen3')
model = ConfigurableGPT(config)
print(f'Parameters: {model.get_num_params() / 1e6:.2f}M')
print('✓ Qwen3 model construction passed')
"
```

### 6.2 Integration Tests

**Smoke Test (10 iterations):**
```bash
# Test all three configs
python train.py config/full_qwen3_0.6b.py --max_iters=10 --compile=False
python train.py config/full_qwen3_1.5b.py --max_iters=10 --compile=False
python train.py config/full_qwen3_2.0b.py --max_iters=10 --compile=False
```

**Short Run (100 iterations):**
```bash
# Test on 2× A6000 with ZeRO-1
torchrun --standalone --nproc_per_node=2 train.py \
  config/full_qwen3_1.5b.py \
  --max_iters=100 \
  --use_zero1=True
```

### 6.3 Comparison Tests

**Compare Qwen3 vs LLaMA 3 vs GPT-2:**
```bash
# Run all three for 1000 iterations
torchrun --standalone --nproc_per_node=2 train.py config/full_gpt2_1.36b.py --max_iters=1000 --use_zero1=True
torchrun --standalone --nproc_per_node=2 train.py config/full_llama3_1.5b_optimal.py --max_iters=1000 --use_zero1=True
torchrun --standalone --nproc_per_node=2 train.py config/full_qwen3_1.5b.py --max_iters=1000 --use_zero1=True

# Compare results
python compare_architectures.py --latest 3
```

**Expected Comparison:**
| Metric | GPT-2 1.36B | LLaMA 3 1.5B | Qwen3 1.5B |
|--------|-------------|--------------|------------|
| Params | 1.29B | 1.54B | ~1.50B |
| Depth | 18 layers | 18 layers | 28 layers |
| MFU | 30-35% | 40-45% | 40-45% |
| Loss (1K iters) | ~8.0 | ~7.5 | ~7.3 (expected) |

---

## 7. Success Criteria

### Minimum Viable Implementation

✅ **Must Have:**
- [ ] Qwen3 preset loads and creates model successfully
- [ ] Extended RoPE (theta=1M) works without errors
- [ ] GQA 2:1 ratio (16:8) validated
- [ ] Qwen3 tokenizer (151K vocab) integrated
- [ ] At least one config (0.6B or 1.5B) trains successfully
- [ ] Loss decreases over training iterations
- [ ] No OOM errors on 2× A6000 with ZeRO-1

### Desired Features

✅ **Should Have:**
- [ ] All three configs (0.6B, 1.5B, 2B) tested
- [ ] Training runs complete 2000 iterations
- [ ] MFU comparable to or better than LLaMA 3
- [ ] Loss comparable to or better than LLaMA 3
- [ ] Documentation complete and accurate
- [ ] Comparison analysis with existing models

### Stretch Goals

✅ **Nice to Have:**
- [ ] Long-context testing (8K or 16K context)
- [ ] Ablation studies (rope_theta, FFN ratio, etc.)
- [ ] Validation on DGX B200 (if accessible)
- [ ] Extended training (10K+ iterations, 25B+ tokens)
- [ ] Downstream task evaluation

---

## 8. Timeline Estimate

### Conservative Estimate (2× A6000)

**Phase 1: Implementation** (~2-3 hours)
- Code changes to support Qwen3
- Testing and debugging
- Documentation updates

**Phase 2: Tokenizer & Dataset** (~1-2 hours)
- Download tokenizer (5 minutes)
- Prepare SlimPajama-6B dataset (30-45 minutes)
- Create README files (15 minutes)

**Phase 3: Configuration & Testing** (~1 hour)
- Create config files (30 minutes)
- Smoke tests (30 minutes)

**Phase 4: Training Validation** (~20-25 hours)
- Qwen3-0.6B short run (100 iters): 30 minutes
- Qwen3-1.5B short run (100 iters): 30 minutes
- Qwen3-0.6B full run (2000 iters): 7-8 hours
- Qwen3-1.5B full run (2000 iters): 8-10 hours
- Analysis and comparison: 2-3 hours

**Total: ~25-30 hours** (wall-clock time, including compute)

---

## 9. Risk Mitigation

### Potential Issues

**1. Extended RoPE Numerical Stability**
- Risk: High `rope_theta` (1M) may cause numerical issues
- Mitigation: Test incrementally (10K → 500K → 1M)
- Fallback: Use 500K if 1M is unstable

**2. Large Vocabulary Memory Usage**
- Risk: 151K vocab larger than previous models
- Mitigation: Monitor memory during training
- Fallback: Reduce batch size if needed

**3. Deeper Model (28 layers) Training Instability**
- Risk: Deeper models may have gradient issues
- Mitigation: Monitor gradient norms carefully
- Fallback: Increase warmup, reduce learning rate

**4. OOM on 2× A6000**
- Risk: Larger models may not fit
- Mitigation: Test with smaller batch sizes
- Fallback: Use Qwen3-0.6B only if needed

---

## 10. Next Steps

### Immediate Actions

1. **Review this plan** with team/advisor
2. **Verify hardware availability** (2× A6000 or DGX B200)
3. **Check HuggingFace access** for downloading Qwen3 tokenizer
4. **Prepare development environment** (ensure transformers library updated)

### Implementation Order

1. ✅ Create documentation (this document)
2. → Update model architecture code (Phase 1)
3. → Download and integrate tokenizer (Phase 2)
4. → Create configuration files (Phase 3)
5. → Run smoke tests and validation (Phase 4)
6. → Execute full training runs (Phase 5)
7. → Analysis and comparison (Phase 6)
8. → Update documentation with results (Phase 7)

---

## Conclusion

This implementation plan provides a complete roadmap for adding Qwen3 architecture support to our modular training system. The approach is:

- **Incremental:** Build on existing GQA and SwiGLU implementations
- **Validated:** Test at each stage before proceeding
- **Documented:** Maintain clear records of decisions and results
- **Comparable:** Enable direct comparison with GPT-2, LLaMA 2, and LLaMA 3

The estimated timeline of 25-30 hours is realistic for the 2× A6000 testing environment, with most time spent on actual training runs rather than implementation.

**Ready to proceed when approved! 🚀**

