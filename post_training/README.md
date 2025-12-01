# Post-Training System for LLM Alignment

A comprehensive system for **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)** to transform base language models into instruction-following assistants.

## 📋 Overview

This system provides everything needed to post-train your pre-trained base model:

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Base Model        │     │   SFT Model         │     │   Aligned Model     │
│   (Pre-trained)     │ ──► │   (Instruction      │ ──► │   (Preference       │
│                     │     │    following)       │     │    aligned)         │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                    │                           │
                            Stage 1: SFT                Stage 2: DPO
                            ~30-60 min                  ~15-30 min
                            (8× B200)                   (8× B200)
```

### Why Post-Training?

| Issue with Base Model | Solution |
|----------------------|----------|
| No instruction following | **SFT** teaches the model to follow instructions |
| Hallucinations | **DPO** reduces hallucinations via preference learning |
| Inconsistent outputs | **SFT + DPO** produces more reliable responses |
| Safety concerns | **DPO** can align with harmlessness preferences |

## 📁 Directory Structure

```
post_training/
├── README.md                    # This file
├── train_sft.py                 # SFT training script
├── train_dpo.py                 # DPO training script
├── configs/
│   ├── sft_qwen3_1.8b.py       # SFT configuration
│   └── dpo_qwen3_1.8b.py       # DPO configuration
├── data/
│   ├── prepare_sft.py          # SFT data preparation
│   └── prepare_dpo.py          # DPO data preparation
└── scripts/                     # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

```bash
# Activate your environment
source /raid/zhf004/llm_TII/venv/bin/activate

# Install additional requirements (if needed)
pip install datasets transformers tqdm
```

### Step 1: Prepare SFT Data (~5 minutes)

```bash
cd /raid/zhf004/llm_TII/post_training

# Prepare Alpaca dataset (52K examples - good starting point)
python data/prepare_sft.py \
    --dataset alpaca \
    --output_dir data/sft_alpaca \
    --max_length 2048

# Or use a larger dataset
python data/prepare_sft.py \
    --dataset ultrachat \
    --output_dir data/sft_ultrachat \
    --max_samples 100000
```

**Supported SFT Datasets:**
| Dataset | Examples | Description |
|---------|----------|-------------|
| `alpaca` | 52K | Stanford Alpaca (recommended for testing) |
| `alpaca_cleaned` | 52K | Cleaned Alpaca (higher quality) |
| `dolly` | 15K | Databricks Dolly (human-generated) |
| `ultrachat` | 200K | UltraChat (high-quality synthetic) |
| `slimorca` | 517K | SlimOrca (cleaned OpenOrca) |

### Step 2: Run SFT Training (~30-60 minutes on 8× B200)

```bash
# Update paths in config first!
# Edit: configs/sft_qwen3_1.8b.py
#   - checkpoint_path: Path to your pre-trained model
#   - sft_data_dir: Path to prepared SFT data

# Single GPU (for testing)
python train_sft.py configs/sft_qwen3_1.8b.py --max_iters=100 --compile=False

# Multi-GPU (production)
torchrun --standalone --nproc_per_node=8 train_sft.py configs/sft_qwen3_1.8b.py
```

### Step 3: (Optional) Prepare DPO Data

```bash
# Prepare UltraFeedback preference dataset
python data/prepare_dpo.py \
    --dataset ultrafeedback \
    --output_dir data/dpo_ultrafeedback \
    --max_samples 50000
```

**Supported DPO Datasets:**
| Dataset | Examples | Description |
|---------|----------|-------------|
| `ultrafeedback` | 60K | UltraFeedback binarized preferences |
| `hh_rlhf` | 170K | Anthropic HH-RLHF |
| `orca_dpo` | 12K | Intel Orca DPO pairs |

### Step 4: (Optional) Run DPO Training (~15-30 minutes)

```bash
# Update paths in config:
# Edit: configs/dpo_qwen3_1.8b.py
#   - sft_checkpoint_path: Path to your SFT model
#   - dpo_data_dir: Path to prepared DPO data

# Run DPO
torchrun --standalone --nproc_per_node=8 train_dpo.py configs/dpo_qwen3_1.8b.py
```

## ⚙️ Configuration

### SFT Configuration (`configs/sft_qwen3_1.8b.py`)

Key parameters to adjust:

```python
# REQUIRED - Update these paths!
checkpoint_path = '/path/to/pretrained/ckpt.pt'  # Your base model
sft_data_dir = '/path/to/sft_data'               # Prepared SFT data

# Training
batch_size = 4                    # Per-GPU batch size
gradient_accumulation_steps = 4   # Effective batch = 4×4×8 = 128
learning_rate = 2e-5              # Lower than pre-training!
max_iters = 3000                  # ~3 epochs over 50K examples

# Regularization
dropout = 0.05                    # Slight regularization for fine-tuning
```

### DPO Configuration (`configs/dpo_qwen3_1.8b.py`)

```python
# REQUIRED
sft_checkpoint_path = '/path/to/sft/ckpt.pt'
dpo_data_dir = '/path/to/dpo_data'

# DPO-specific
beta = 0.1                        # KL penalty (0.05-0.5)
learning_rate = 5e-7              # Even lower than SFT
max_iters = 1000                  # Fewer iterations needed
```

## 📊 Compute Requirements

| Stage | Dataset Size | Time (8× B200) | GPU Hours |
|-------|-------------|----------------|-----------|
| **SFT** | 50K examples | 30-60 min | ~4-8 |
| **SFT** | 200K examples | 2-3 hours | ~16-24 |
| **DPO** | 50K pairs | 15-30 min | ~2-4 |

**Comparison with Pre-training:**
- Pre-training: 50 hours × 8 GPUs = **400 GPU-hours**
- SFT + DPO: ~1-2 hours × 8 GPUs = **~10 GPU-hours** (~2.5% of pre-training!)

## 🔧 Advanced Usage

### Custom System Prompt

```bash
python data/prepare_sft.py \
    --dataset alpaca \
    --system_prompt "You are a coding assistant. Help users write clean, efficient code."
```

### Hyperparameter Tuning

```bash
# Quick experiments with different learning rates
for lr in 1e-5 2e-5 5e-5; do
    python train_sft.py configs/sft_qwen3_1.8b.py \
        --learning_rate=$lr \
        --max_iters=500 \
        --out_dir=out-sft-lr-$lr
done
```

### Resume Training

```bash
# Resume from checkpoint
python train_sft.py configs/sft_qwen3_1.8b.py \
    --checkpoint_path=out-qwen3-1.8b-sft/ckpt_001000.pt
```

## 📈 Expected Results

### SFT Results

| Metric | Base Model | After SFT |
|--------|-----------|-----------|
| Instruction Following | ❌ Poor | ✅ Good |
| Format Compliance | ❌ Random | ✅ Structured |
| Conversational | ❌ No | ✅ Yes |

### DPO Results (Additional)

| Metric | After SFT | After DPO |
|--------|-----------|-----------|
| Hallucination | Medium | Low |
| Helpfulness | Good | Better |
| Harmlessness | Variable | Improved |

## 🔍 Monitoring Training

### Terminal Output

```
iter   100 | loss 1.8234 | lr 2.00e-05 | 234.5ms | 8742 tok/s
iter   200 | loss 1.5432 | lr 2.00e-05 | 231.2ms | 8863 tok/s
──────────────────────────────────────────────────────────────
Iter 200: train loss 1.5234, val loss 1.6123
──────────────────────────────────────────────────────────────
Saved checkpoint: ckpt_000200.pt
```

### WandB Integration

Enable in config:
```python
wandb_log = True
wandb_project = 'qwen3-sft'
wandb_run_name = 'my-experiment'
```

### Checkpoints

```
out-qwen3-1.8b-sft/
├── ckpt_000200.pt    # Checkpoint at iter 200
├── ckpt_000400.pt    # Checkpoint at iter 400
├── ckpt.pt           # Latest checkpoint
└── run_*.json        # Training logs
```

## 🧪 Testing the Fine-tuned Model

After training, test with the serving system:

```bash
# Start inference server with SFT model
cd /raid/zhf004/llm_TII/serving_system
python serve_qwen3.py \
    --checkpoint /raid/zhf004/llm_TII/post_training/out-qwen3-1.8b-sft/ckpt.pt

# Open http://localhost:8000 for chat interface
```

Or evaluate on benchmarks:

```bash
cd /raid/zhf004/llm_TII/evaluation_system
python eval_benchmarks.py \
    --checkpoint /raid/zhf004/llm_TII/post_training/out-qwen3-1.8b-sft/ckpt.pt \
    --mode generate
```

## 🔬 Technical Details

### Loss Masking (SFT)

Only compute loss on assistant response tokens:

```
<|im_start|>system
You are helpful.<|im_end|>         ← Masked (-100)
<|im_start|>user
What is 2+2?<|im_end|>             ← Masked (-100)
<|im_start|>assistant
2+2 equals 4.<|im_end|>            ← Computed (loss)
```

### DPO Loss Function

```
L_DPO = -log σ(β × (log π(y_w|x) - log π(y_l|x) 
                  - log π_ref(y_w|x) + log π_ref(y_l|x)))
```

Where:
- π = policy model (being trained)
- π_ref = reference model (frozen SFT checkpoint)
- y_w = chosen response
- y_l = rejected response
- β = KL penalty coefficient

## 📚 References

### Papers
1. **SFT/InstructGPT**: Ouyang et al., "Training language models to follow instructions" (2022)
2. **DPO**: Rafailov et al., "Direct Preference Optimization" (2023)
3. **Alpaca**: Taori et al., "Stanford Alpaca" (2023)

### Code References
- Training system: `../enhanced_training_system/`
- Serving system: `../serving_system/`
- Evaluation: `../evaluation_system/`

## ❓ Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train_sft.py configs/sft_qwen3_1.8b.py --batch_size=2

# Or increase gradient accumulation
python train_sft.py configs/sft_qwen3_1.8b.py --gradient_accumulation_steps=16
```

### Slow Training

```bash
# Enable torch.compile (but increases startup time)
python train_sft.py configs/sft_qwen3_1.8b.py --compile=True
```

### Poor Results

1. **Check learning rate**: SFT should use 1e-5 to 5e-5 (much lower than pre-training)
2. **Check data quality**: Inspect a few examples from your prepared data
3. **Check loss masking**: Ensure only response tokens are being trained on
4. **Train longer**: Try more iterations or epochs

## 📄 License

MIT License (same as parent project)

