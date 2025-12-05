# Post-Training System Overview

## Purpose

Transform a pre-trained base language model into an instruction-following assistant through two stages:

1. **SFT (Supervised Fine-Tuning)** - Teaches the model to follow instructions
2. **DPO (Direct Preference Optimization)** - Aligns the model with human preferences, reducing hallucinations

## Pipeline Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Base Model        │     │   SFT Model         │     │   Aligned Model     │
│   (Pre-trained)     │ ──► │   (Instruction      │ ──► │   (Preference       │
│   ckpt_160000.pt    │     │    following)       │     │    aligned)         │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                    │                           │
                            Stage 1: SFT                Stage 2: DPO
                            ~30-60 min                  ~15-30 min
                            (8× B200)                   (8× B200)
```

## Directory Structure

```
post_training/
├── docs/                         # Documentation (this folder)
│   ├── 01_overview.md           # This file
│   ├── 02_sft_implementation.md # SFT technical details
│   ├── 03_dpo_implementation.md # DPO technical details
│   ├── 04_data_preparation.md   # Data pipeline
│   └── 05_configuration_guide.md # Config options
├── train_sft.py                  # SFT training script
├── train_dpo.py                  # DPO training script
├── configs/
│   ├── sft_qwen3_1.8b.py        # SFT configuration
│   └── dpo_qwen3_1.8b.py        # DPO configuration
├── data/
│   ├── prepare_sft.py           # SFT data preparation
│   ├── prepare_dpo.py           # DPO data preparation
│   └── sft_alpaca/              # Prepared Alpaca dataset
├── scripts/                      # Utility scripts
├── requirements.txt              # Dependencies
└── README.md                     # Quick start guide
```

## Why Two Stages?

| Issue with Base Model | Solution |
|----------------------|----------|
| No instruction following | **SFT** teaches structured responses |
| Hallucinations | **DPO** reduces via preference learning |
| Inconsistent outputs | **SFT + DPO** produces reliable responses |
| Safety concerns | **DPO** can align with harmlessness |

## Compute Requirements

| Stage | Dataset | Time (8× B200) | GPU-Hours |
|-------|---------|----------------|-----------|
| SFT | 50K examples | 30-60 min | ~4-8 |
| DPO | 50K pairs | 15-30 min | ~2-4 |
| **Total** | - | **~1-2 hours** | **~10** |

**Comparison**: Pre-training took 50 hours × 8 GPUs = 400 GPU-hours. Post-training is ~2.5% of that cost.

## Quick Start

```bash
# 1. Activate environment
source /raid/zhf004/llm_TII/venv/bin/activate
cd /raid/zhf004/llm_TII/post_training

# 2. Data is already prepared (sft_alpaca/)

# 3. Run SFT training
torchrun --standalone --nproc_per_node=8 train_sft.py configs/sft_qwen3_1.8b.py

# 4. (Optional) Run DPO after SFT
python data/prepare_dpo.py --dataset ultrafeedback --output_dir data/dpo_ultrafeedback
torchrun --standalone --nproc_per_node=8 train_dpo.py configs/dpo_qwen3_1.8b.py
```

## Key Design Decisions

1. **ChatML Format**: Uses Qwen3's native chat template with `<|im_start|>` and `<|im_end|>` tokens
2. **Loss Masking**: Only computes loss on assistant responses, not instructions
3. **Lower Learning Rate**: 2e-5 (vs 3e-4 in pre-training) to prevent catastrophic forgetting
4. **Modular Architecture**: Reuses `model_builder.py` from enhanced_training_system
5. **JSONL Format**: Human-readable data format for easy inspection and debugging

