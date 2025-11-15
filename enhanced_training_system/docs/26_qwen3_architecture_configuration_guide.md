# Qwen3 Architecture Configuration Guide

**Date:** November 13, 2025  
**Phase:** Architecture Extension - Qwen3 Integration  
**Status:** Planning

---

## Overview

This document provides guidance on configuring a Qwen3-style architecture using our modular training system, based on the official Qwen3 technical specifications and feasibility analysis for training on DGX B200 hardware.

---

## 1. Qwen3 Architecture Specifications

From the Qwen3 technical report and HuggingFace configurations, the dense models use:

### Core Components

- **Normalization:** RMSNorm (pre-norm)
- **Activation:** SiLU (used in SwiGLU)
- **FFN Type:** SwiGLU with `hidden_act="silu"`
- **Position Encoding:** RoPE with `rope_theta = 1,000,000` (extended)
- **Attention:** GQA (Grouped Query Attention) - more Q heads than KV heads
- **Bias:** No bias (attention_bias = false)
- **Precision:** bfloat16
- **Context Length:** 32K–128K tokens
- **Tokenizer:** BBPE (Byte-level BPE), vocab = 151,643 tokens

### Qwen3-0.6B Concrete Configuration

Official specifications from HuggingFace `Qwen/Qwen3-0.6B`:

```python
num_hidden_layers = 28
hidden_size = 1024
intermediate_size = 3072          # ~3× hidden_size
num_attention_heads = 16          # Q heads
num_key_value_heads = 8           # KV heads (2:1 GQA ratio)
head_dim = 128                    # hidden_size / num_attention_heads
hidden_act = "silu"
vocab_size = 151643
rope_theta = 1_000_000           # Extended RoPE base
torch_dtype = "bfloat16"
max_position_embeddings = 32768  # Context window
tie_word_embeddings = False      # No weight tying
```

**Parameter Count:** ~0.6B (600M parameters)

---

## 2. Adapting to Our Modular System

### Configuration Mapping

To match Qwen3 dense architecture in `config/full_custom.py`:

```python
# === ARCHITECTURE ===
arch_preset = 'custom'  # Qwen3-style dense

# Qwen3-style components:
normalization = 'rmsnorm'           # RMSNorm (pre-norm)
activation = 'silu'                 # SwiGLU uses SiLU
attention_backend = 'flash_attn_2'  # Efficient attention (or 'sdpa')
position_encoding = 'rope'          # RoPE with extended theta
norm_position = 'pre'               # Pre-norm architecture
ffn_type = 'swiglu'                 # SwiGLU FFN
bias = False                        # No bias in projections
weight_tying = False                # Qwen doesn't tie embeddings

# Qwen3-0.6B size parameters:
n_layer = 28
n_head = 16                         # Query heads
n_embd = 1024                       # Hidden dimension
num_key_value_heads = 8             # KV heads (GQA 2:1 ratio)
d_ff = 3072                         # FFN dimension (~3× hidden)
block_size = 32768                  # Context window (can reduce for training)
vocab_size = 151643                 # Qwen3 tokenizer vocab
rope_theta = 1_000_000              # Extended RoPE base
dropout = 0.0
```

### Scaling to 1-2B Parameters

To create Qwen3-style models at 1-2B scale:

**Keep constant:**
- Architecture components (RMSNorm, SwiGLU, RoPE, GQA)
- Depth: `n_layer = 28` (same as 0.6B)
- Head pattern: `n_head = 16`, `num_key_value_heads = 8` (2:1 GQA)

**Scale up:**
- `n_embd` (hidden dimension):
  - For ~1B: `n_embd = 1536`
  - For ~1.5B: `n_embd = 2048`
  - For ~2B: `n_embd = 2304`
- `d_ff = 3 × n_embd` (maintain 3× expansion ratio)

**Example: Qwen3-1.5B-style**
```python
n_layer = 28
n_head = 16
n_embd = 2048
num_key_value_heads = 8
d_ff = 6144  # 3 × 2048
# Results in ~1.5B parameters
```

---

## 3. Tokenizer Integration

### Qwen3 Tokenizer (BBPE)

Qwen3 uses a Byte-level BPE tokenizer with:
- **Vocabulary:** 151,643 tokens
- **Special tokens:** Includes system, user, assistant markers
- **Type:** BBPE (similar to GPT-2 but expanded vocab)

### Integration Path

Use HuggingFace's tokenizer directly:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
# vocab_size: 151643
# Special tokens: <|im_start|>, <|im_end|>, etc.
```

Save locally for dataset preparation:
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
tokenizer.save_pretrained('./qwen3_tokenizer')
print(f'Saved Qwen3 tokenizer (vocab={tokenizer.vocab_size})')
"
```

---

## 4. Training Feasibility on DGX B200

### Hardware Specifications

**DGX B200 (8× B200 GPUs):**
- Single B200: 4.5 PFLOPs FP16/BF16 tensor cores
- Total system: 8 × 4.5 = **36 PFLOPs** peak
- Memory per GPU: 192 GB HBM3e

### Compute Analysis for 60 Hours

**Time budget:**
- 60 hours = 216,000 seconds

**Theoretical FLOPs:**
- Raw: `36e15 FLOPs/s × 2.16e5 s ≈ 7.8e21 FLOPs`

**Realistic MFU (30-50%):**
- Conservative (30% MFU): **2.3×10²¹ FLOPs**
- Optimistic (50% MFU): **3.9×10²¹ FLOPs**

### Training Capacity Estimates

Using the rule: `FLOPs per token ≈ 6 × N_params`

**For 1B parameters:**
- Tokens (30% MFU): `2.3e21 / 6e9 ≈ 380B tokens`
- Tokens (50% MFU): `3.9e21 / 6e9 ≈ 650B tokens`

**For 2B parameters:**
- Tokens (30% MFU): `2.3e21 / 1.2e10 ≈ 190B tokens`
- Tokens (50% MFU): `3.9e21 / 1.2e10 ≈ 325B tokens`

**Chinchilla-optimal training:**
- 1B model: ~20B tokens (data-optimal)
- 2B model: ~40B tokens (data-optimal)

### Conclusion

**Yes, Qwen3-style 1-2B models are very feasible for 60 hours on DGX B200:**

- ✅ Even at 30% MFU, you can train **far beyond** Chinchilla-optimal token counts
- ✅ A 2B model with 200B tokens (5× optimal) easily fits in the budget
- ✅ Plenty of headroom for multiple ablation runs
- ⚠️ **Bottleneck is data quality/quantity, not compute**

---

## 5. Recommended Training Strategy

### Phase 1: Baseline Validation (10-20 hours)

**Goal:** Verify we can reproduce Qwen-style training

**Config:** Exact Qwen3-0.6B clone
```python
n_layer = 28
n_head = 16
n_embd = 1024
num_key_value_heads = 8
d_ff = 3072
vocab_size = 151643
rope_theta = 1_000_000
```

**Training:** 20-40B tokens (~20 hours)
- Establishes baseline performance
- Validates architecture + tokenizer integration
- Tests MFU on DGX B200

### Phase 2: Scale to 1-2B (30-40 hours)

**Config:** Qwen3-1.5B-style
```python
n_layer = 28
n_head = 16
n_embd = 2048
num_key_value_heads = 8
d_ff = 6144
```

**Training:** 60-100B tokens (~30-40 hours)
- Target: 2-2.5× Chinchilla-optimal
- Compare scaling behavior vs LLaMA 3
- Measure efficiency gains from Qwen architecture

### Phase 3: Ablations (10-20 hours)

**Variables to test:**
1. `rope_theta` variations (1e5, 5e5, 1e6, 2e6)
2. FFN expansion ratio (2.67×, 3×, 3.5×)
3. GQA ratios (4:1, 2:1, 1:1/MHA)
4. Context length impact (2K, 8K, 32K)

Each ablation: ~5-10 hours with reduced token count

---

## 6. Comparison with Existing Models

### Architecture Comparison Table

| Feature | GPT-2 1.36B | LLaMA 2 1.36B | LLaMA 3 1.5B | Qwen3 1.5B |
|---------|-------------|---------------|--------------|------------|
| **Layers** | 18 | 18 | 18 | 28 |
| **Hidden** | 2304 | 2304 | 2048 | 2048 |
| **Heads** | 18 | 18 | 16 | 16 |
| **KV Heads** | 18 (MHA) | 18 (MHA) | 8 (GQA) | 8 (GQA) |
| **FFN Type** | Standard 4× | SwiGLU 2.67× | SwiGLU 3.5× | SwiGLU 3× |
| **FFN Dim** | 9216 | 6144 | 7168 | 6144 |
| **Norm** | LayerNorm | RMSNorm | RMSNorm | RMSNorm |
| **Pos Enc** | Learned | RoPE 10K | RoPE 500K | RoPE 1M |
| **Vocab** | 50K | 32K | 128K | 152K |
| **Params** | 1.29B | 1.36B | 1.54B | ~1.50B |

### Key Qwen3 Advantages

1. **Deeper architecture** (28 layers vs 18) - better representation capacity
2. **Extended RoPE** (1M theta) - superior long-context handling
3. **Larger vocabulary** (152K) - better tokenization efficiency
4. **Proven scaling** - Qwen family has strong empirical results
5. **Modern design** - incorporates latest best practices

---

## 7. Expected Performance

### Training Metrics (1.5B model, 80B tokens)

**On DGX B200 (8 GPUs, FSDP):**
- MFU: 45-55% (with FlashAttention-2)
- Tokens/sec: 60,000-80,000
- Time per iteration: ~8-12s (batch_size=16, grad_accum=8)
- Memory per GPU: 30-40 GB
- Total training time: ~30-40 hours

**Expected Loss (80B tokens):**
- Theoretical minimum: ~2.1-2.2
- Better than LLaMA 2 1.36B (deeper architecture)
- Comparable to LLaMA 3 1.5B (similar GQA efficiency)

### Downstream Task Performance (estimated)

Compared to our existing models:
- **Better than:** GPT-2 1.36B (modern architecture)
- **Similar to:** LLaMA 3 1.5B (comparable params + efficiency)
- **Competitive with:** Official Qwen3-0.6B (more parameters + tokens)

---

## 8. Implementation Checklist

### Prerequisites
- [ ] Qwen3 tokenizer downloaded and saved locally
- [ ] Dataset prepared with Qwen3 tokenizer (151K vocab)
- [ ] FSDP/ZeRO-1 tested on DGX B200
- [ ] FlashAttention-2 installed and verified

### Architecture Support
- [ ] Extended RoPE theta (1M) support in `model_components.py`
- [ ] Verify GQA works with 2:1 ratio (16:8)
- [ ] Test SwiGLU with 3× expansion ratio
- [x] Confirmed vocab_size=151643 (actual Qwen3 tokenizer)

### Configuration Files
- [ ] `config/full_qwen3_0.6b.py` - baseline validation
- [ ] `config/full_qwen3_1.5b.py` - scaled version
- [ ] `config/full_qwen3_2.0b.py` - (optional) larger variant

### Training Infrastructure
- [ ] Dataset: SlimPajama tokenized with Qwen3 tokenizer
- [ ] Multi-stage training pipeline (optional)
- [ ] Evaluation benchmarks setup
- [ ] Logging and monitoring configured

---

## References

- Qwen Technical Report: https://qwenlm.github.io/
- Qwen3 HuggingFace: https://huggingface.co/Qwen/Qwen3-0.6B
- Extended RoPE: "Extending Context Window of Large Language Models"
- SwiGLU: "GLU Variants Improve Transformer" (Shazeer, 2020)

---

**Next Steps:** See `28_qwen3_implementation_plan.md` for detailed implementation roadmap.

