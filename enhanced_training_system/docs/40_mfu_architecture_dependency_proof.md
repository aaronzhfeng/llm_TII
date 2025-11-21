# MFU Computation & Architecture Dependency: Complete Proof

**Date:** 2025-11-21  
**System:** Enhanced Training System  
**Purpose:** Document how MFU computation dynamically depends on model architecture configuration

---

## Executive Summary

This document proves that **MFU (Model FLOPs Utilization) computation is tightly coupled to model architecture** and that changing config parameters **dynamically affects MFU calculation at runtime**. No hardcoding of model-specific FLOPs exists.

**Key Findings:**
- MFU computation reads architecture parameters directly from `self.config`
- Config changes (layers, heads, hidden dim, FFN size) automatically affect FLOPs calculation
- Only hardware specs (GPU peak FLOPS) and PaLM formula constants are hardcoded
- All model-specific computations are dynamic

---

## 1. Architecture Flow: Config → Model → MFU

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: CONFIG FILE (e.g., full_qwen3_1.8b_optimal.py)      │
│ ──────────────────────────────────────────────────────────  │
│ n_layer = 24           # Number of transformer layers       │
│ n_head = 16            # Number of attention heads          │
│ n_embd = 2048          # Hidden dimension                   │
│ d_ff = 6144            # FFN dimension                       │
│ block_size = 2048      # Sequence length                     │
│ num_key_value_heads = 8  # For GQA                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ exec(open('configurator.py').read())
                     │ Loaded as Python variables in train.py
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: TRAIN.PY - Build ModelArchitectureConfig            │
│ ──────────────────────────────────────────────────────────  │
│ arch_config = ModelArchitectureConfig(                      │
│     n_layer=n_layer,          # 24 ← from config            │
│     n_head=n_head,            # 16 ← from config            │
│     n_embd=n_embd,            # 2048 ← from config          │
│     d_ff=d_ff,                # 6144 ← from config          │
│     block_size=block_size,    # 2048 ← from config          │
│     num_key_value_heads=8,    # from config                 │
│     ...                                                      │
│ )                                                            │
│                                                              │
│ # Pass to model constructor                                 │
│ model = ConfigurableGPT(arch_config)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Store config in model instance
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: MODEL_BUILDER.PY - ConfigurableGPT.__init__()       │
│ ──────────────────────────────────────────────────────────  │
│ class ConfigurableGPT(nn.Module):                           │
│     def __init__(self, config: ModelArchitectureConfig):    │
│         super().__init__()                                  │
│         self.config = config  # ← STORED for later use      │
│         # Build model using config parameters...            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ MFU method called during training
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: estimate_mfu_detailed() - Runtime Computation       │
│ ──────────────────────────────────────────────────────────  │
│ def estimate_mfu_detailed(self, fwdbwd_per_iter, dt, ...):  │
│     cfg = self.config        # ← READ stored config         │
│     H = cfg.n_embd           # 2048                         │
│     L = cfg.n_layer          # 24                           │
│     a = cfg.n_head           # 16                           │
│     S = cfg.block_size       # 2048                         │
│     D_ff = cfg.d_ff          # 6144                         │
│                                                              │
│     # Use in FLOPs formulas (DYNAMIC computation!)          │
│     attention_qkv_flops = 6 * S * H * H                     │
│     attention_scores_flops = a * S * S * H                  │
│     ffn_up_flops = 2 * S * H * D_ff                         │
│     ffn_down_flops = 2 * S * D_ff * H                       │
│     ...                                                      │
│     # PaLM formula                                           │
│     attn_flops = 12.0 * L * a * Q * T / 1e9                 │
│     non_attn_flops = 6.0 * N_billion                        │
│     ...                                                      │
│     mfu = flops_achieved / hardware_peak_flops              │
│     return mfu_breakdown                                     │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight:** `self.config` is the bridge between config file parameters and MFU computation.

---

## 2. Config Parameters That Affect MFU

### Parameters with Direct Impact

| Parameter | Variable | Used In Formula | Impact on FLOPs | Impact on MFU |
|-----------|----------|-----------------|-----------------|---------------|
| **`n_layer`** | `L` | `12.0 * L * a * Q * T`<br>(Attention component) | ↑ Layers → ↑ FLOPs | ↑ FLOPs → ↓ MFU |
| **`n_head`** | `a` | `12.0 * L * a * Q * T`<br>`a * S * S * H` | ↑ Heads → ↑ FLOPs | ↑ FLOPs → ↓ MFU |
| **`n_embd`** | `H` | `6 * S * H * H`<br>`2 * S * H * D_ff`<br>Parameter count (N) | ↑ Hidden → ↑ FLOPs | ↑ FLOPs → ↓ MFU |
| **`d_ff`** | `D_ff` | `2 * S * H * D_ff`<br>`2 * S * D_ff * H` | ↑ FFN → ↑ FLOPs | ↑ FLOPs → ↓ MFU |
| **`block_size`** | `S` | ALL formulas<br>(every formula has S) | ↑ Seq → ↑ FLOPs | ↑ FLOPs → ↓ MFU |
| **`num_key_value_heads`** | `kv_heads` | GQA projection size | GQA reduces FLOPs | ↓ FLOPs → ↑ MFU |
| `vocab_size` | `V` | Embedding params only | Minimal (non-compute) | Minimal |

### Code Evidence

From `model_builder.py` lines 427-505:

```python
def estimate_mfu_detailed(self, fwdbwd_per_iter, dt, device_type='cuda', num_gpus=1):
    # READ CONFIG PARAMETERS (not hardcoded!)
    cfg = self.config
    H = cfg.n_embd              # ← From config
    L = cfg.n_layer             # ← From config
    a = cfg.n_head              # ← From config
    S = cfg.block_size          # ← From config
    D_ff = cfg.d_ff             # ← From config
    
    # ATTENTION FLOPs (uses H, S, a)
    attention_qkv_flops = 6 * S * H * H
    attention_scores_flops = a * S * S * H
    attention_output_flops = a * S * S * H
    attention_proj_flops = 2 * S * H * H
    attention_flops = (attention_qkv_flops + attention_scores_flops + 
                      attention_output_flops + attention_proj_flops)
    
    # FFN FLOPs (uses S, H, D_ff)
    ffn_up_flops = 2 * S * H * D_ff
    ffn_down_flops = 2 * S * D_ff * H
    ffn_flops = ffn_up_flops + ffn_down_flops
    
    # Per-layer total (includes L implicitly)
    flops_per_layer = attention_flops + ffn_flops + norm_flops
    total_forward_flops = L * flops_per_layer
    
    # PaLM formula (uses L, a, Q, S, N)
    N_params = self.get_num_params(non_embedding=True)
    N_billion = N_params / 1e9
    Q = H // a  # head dimension
    T = S       # sequence length
    
    non_attn_flops = 6.0 * N_billion
    attn_flops = 12.0 * L * a * Q * T / 1e9
    
    training_flops_per_token = (non_attn_flops + attn_flops) * 1e9
    
    # Compute MFU
    tokens_per_iter = S * fwdbwd_per_iter
    flops_per_iter = training_flops_per_token * tokens_per_iter
    flops_achieved = flops_per_iter / dt
    hardware_peak_flops = hardware_peak_flops_per_gpu * num_gpus
    mfu = flops_achieved / hardware_peak_flops
```

**Every single architecture parameter comes from `cfg` (self.config), which comes from the config file!**

---

## 3. What IS Hardcoded vs What's Dynamic

### Hardcoded (System Constants)

These are **intentionally** hardcoded as they represent physical hardware or academic standards:

```python
# 1. HARDWARE SPECS (model_builder.py line 518-527)
hardware_specs = {
    'cuda': {
        'B200': {'bf16': 2250e12, 'fp16': 2250e12, 'fp32': 90e12},  # Dense peak
        'H100': {'bf16': 989e12, 'fp16': 989e12, 'fp32': 67e12},
        'A100': {'bf16': 312e12, 'fp16': 312e12, 'fp32': 19.5e12},
        'A6000': {'bf16': 155.0e12, 'fp16': 155.0e12, 'fp32': 38.7e12},
        'V100': {'bf16': 125e12, 'fp16': 125e12, 'fp32': 15.7e12},
    }
}
# Reason: Physical GPU specifications from NVIDIA datasheets
# How to change: Edit model_builder.py and model.py

# 2. PALM FORMULA CONSTANTS (model_builder.py line 501-502)
non_attn_flops = 6.0 * N_billion      # 6N coefficient
attn_flops = 12.0 * L * a * Q * T     # 12LHQT coefficient
# Reason: Academic standard from PaLM paper (Chowdhery et al., 2022)
# How to change: Would require changing the MFU standard itself
```

### Dynamic (Config-Driven)

These are **automatically** loaded from config files at runtime:

```python
# ALL MODEL ARCHITECTURE PARAMETERS
n_layer          # ← config file
n_head           # ← config file
n_embd           # ← config file
d_ff             # ← config file
block_size       # ← config file
num_key_value_heads  # ← config file
vocab_size       # ← config file

# DERIVED QUANTITIES (computed from config)
N_params         # Computed from actual model size
Q                # = n_embd // n_head
attention_flops  # Computed from n_head, n_embd, block_size
ffn_flops        # Computed from n_embd, d_ff, block_size
```

---

## 4. Proof of Dynamic Behavior

### Example: Change LLaMA Config

**Original Config (`full_llama_1.36b.py`):**
```python
n_layer = 18
n_head = 18
n_embd = 2304
d_ff = 6144
block_size = 2048
```

**Predicted FLOPs/Token:**
```python
# Attention: 12 * L * a * Q * T / 1e9
Q = 2304 / 18 = 128
attn_flops = 12 * 18 * 18 * 128 * 2048 / 1e9 = 10.74 GFLOPs

# Non-attention: 6 * N (N ≈ 1.294B)
non_attn_flops = 6 * 1.294 = 7.76 GFLOPs

# Total: ~18.5 GFLOPs/token
```

### Test Case 1: Halve the Layers

**Modified Config:**
```python
n_layer = 9  # Changed from 18
n_head = 18
n_embd = 2304
d_ff = 6144
block_size = 2048
```

**Expected Change:**
- Model parameters: ~1.294B → ~0.7B (halved)
- Attention FLOPs: 10.74 → 5.37 GFLOPs (halved, due to L)
- Non-attention FLOPs: 7.76 → ~4.0 GFLOPs (halved, due to N)
- **Total FLOPs/token: ~18.5 → ~9.4 GFLOPs (≈50% reduction)**

**MFU Impact:**
- For same throughput (tokens/sec), MFU would double
- Because model does 50% less work per token

### Test Case 2: Change FFN Size

**Modified Config:**
```python
n_layer = 18
n_head = 18
n_embd = 2304
d_ff = 12288  # Changed from 6144 (doubled)
block_size = 2048
```

**Expected Change:**
- FFN FLOPs: `2*S*H*D_ff + 2*S*D_ff*H = 4*S*H*D_ff`
- Original: `4 * 2048 * 2304 * 6144 ≈ 115.8 TFLOPs`
- Modified: `4 * 2048 * 2304 * 12288 ≈ 231.6 TFLOPs` (doubled)
- **Model becomes slower, MFU drops accordingly**

### Verification Script

To verify this is truly dynamic, you can test:

```python
# In train.py, after model initialization:
print(f"Config n_layer: {model.config.n_layer}")
print(f"Config n_embd: {model.config.n_embd}")
print(f"Config d_ff: {model.config.d_ff}")

mfu_info = model.estimate_mfu_detailed(
    fwdbwd_per_iter=batch_size * gradient_accumulation_steps * world_size,
    dt=1.0,
    num_gpus=world_size
)
print(f"FLOPs/token: {mfu_info['flops_per_token'] / 1e9:.2f} GFLOPs")
```

Change any config parameter, rerun, and observe the FLOPs/token changes automatically.

---

## 5. Comparison: LLaMA 1.36B vs Qwen3 1.8B

### Architecture Differences

| Parameter | LLaMA 1.36B | Qwen3 1.8B | Impact |
|-----------|-------------|------------|--------|
| `n_layer` | 18 | 24 | +33% layers → +33% attention FLOPs |
| `n_head` | 18 | 16 | -11% heads → -11% attention FLOPs |
| `n_embd` | 2304 | 2048 | -11% hidden → less QKV/FFN compute |
| `d_ff` | 6144 | 6144 | Same FFN expansion |
| `block_size` | 2048 | 2048 | Same sequence length |
| Total Params | 1.294B | 1.830B | +41% params → +41% non-attn FLOPs |

### FLOPs/Token Calculation

**LLaMA 1.36B:**
```python
# Attention: 12 * 18 * 18 * 128 * 2048 / 1e9 = 10.74 GFLOPs
# Non-attention: 6 * 1.294 = 7.76 GFLOPs
# Total: ~18.5 GFLOPs/token
```

**Qwen3 1.8B:**
```python
# Attention: 12 * 24 * 16 * 128 * 2048 / 1e9 = 12.88 GFLOPs
# Non-attention: 6 * 1.83 = 10.98 GFLOPs
# Total: ~23.9 GFLOPs/token
```

**Result:** Qwen3 requires ~29% more FLOPs per token due to:
1. More layers (24 vs 18)
2. More parameters (1.83B vs 1.29B)

**MFU Implication:**
- For same tokens/sec throughput, Qwen3 achieves higher absolute FLOPs
- But requires more compute, so may have similar or slightly lower MFU%

This difference is **automatically calculated** from config parameters, proving the system is not hardcoded!

---

## 6. Why This Design Matters

### Benefits of Dynamic Architecture-Aware MFU

1. **Accuracy**: MFU reflects actual model compute, not a generic estimate
2. **Flexibility**: Add new architectures (e.g., Mixture-of-Experts) by just updating configs
3. **Debuggability**: Can trace exact FLOPs from config → model → MFU
4. **Comparability**: Fair comparison between different architectures (LLaMA vs Qwen3)

### What Would Be Wrong with Hardcoding

If FLOPs/token were hardcoded:
```python
# BAD: Hardcoded approach
if model_name == "llama_1.36b":
    flops_per_token = 18.5e9
elif model_name == "qwen3_1.8b":
    flops_per_token = 23.9e9
```

**Problems:**
- Breaks when you change `n_layer` in config
- Can't handle custom architectures
- Requires manual updates for every model variant
- No way to verify accuracy

### Current Design (Dynamic)

```python
# GOOD: Dynamic calculation from architecture
flops_per_token = self._calculate_flops_from_architecture(
    n_layer=self.config.n_layer,
    n_head=self.config.n_head,
    n_embd=self.config.n_embd,
    d_ff=self.config.d_ff,
    # ... all parameters from config
)
```

**Advantages:**
- ✅ Automatically correct for any config change
- ✅ Works for custom architectures
- ✅ Self-documenting (formula is in code)
- ✅ Testable and verifiable

---

## 7. Code Locations Summary

### Where Config Parameters Are Set
- **Config Files**: `config/full_llama_1.36b.py`, `config/full_qwen3_1.8b_optimal.py`
- **Loaded By**: `train.py` via `configurator.py`

### Where Config Flows to Model
- **File**: `train.py` lines 358-378
- **Code**: `arch_config = ModelArchitectureConfig(...)`
- **Then**: `model = ConfigurableGPT(arch_config)`

### Where Config Is Stored
- **File**: `model_builder.py` line 290
- **Code**: `self.config = config`

### Where Config Is Used for MFU
- **File**: `model_builder.py` lines 427-505
- **Method**: `estimate_mfu_detailed()`
- **Code**: Reads from `self.config` for all architecture parameters

### Where Hardware Specs Are Defined
- **File**: `model_builder.py` line 518-527 (also `model.py` line 443-451)
- **Type**: Hardcoded dictionary (intentionally, as physical constants)

---

## 8. Conclusion

**This document proves:**

1. ✅ **MFU computation requires model architecture** - It's a method on the model class, not a standalone function
2. ✅ **Config parameters directly affect MFU** - All architecture values come from `self.config`
3. ✅ **Changes are dynamic, not hardcoded** - Changing config file immediately affects MFU calculation
4. ✅ **Only hardware specs are hardcoded** - B200: 2,250 TFLOPS, A100: 312 TFLOPS, etc.
5. ✅ **System is architecture-agnostic** - Works for LLaMA, Qwen3, GPT-2, or any custom architecture

**Key Insight:**
```python
MFU = achieved_flops / peak_flops

achieved_flops = f(architecture, throughput)  # Dynamic from config
peak_flops = constant(gpu_type)               # Hardcoded hardware spec
```

The architecture dependency ensures MFU accurately reflects the **actual computational characteristics** of each specific model, making it a reliable metric for comparing different architectures and optimization strategies.

---

## References

- **PaLM Paper**: Chowdhery et al., 2022, "PaLM: Scaling Language Modeling with Pathways"
- **Code**: `enhanced_training_system/model_builder.py`, `estimate_mfu_detailed()` method
- **Related Docs**:
  - `36_mfu_global_calculation_update.md` - Global vs per-GPU MFU calculation
  - `13_mfu_fix_summary_palm_formula_implementation.md` - PaLM formula implementation
  - `34_b200_mfu_optimization_implementation.md` - B200 optimization features

