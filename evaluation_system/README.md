# Qwen3-1.8B Evaluation System

Benchmark and qualitative evaluation suite for the custom-trained Qwen3 model.

## 📁 Directory Structure

```
evaluation_system/
├── README.md
├── scripts/                    # Evaluation scripts
│   ├── eval_benchmarks.py      # Main benchmark evaluation
│   ├── eval_qwen3_official.py  # Official model baseline
│   └── plot_comparison.py      # Visualization
├── results/                    # All evaluation outputs
│   ├── benchmark/              # Benchmark JSON results
│   ├── qualitative/            # Generation comparison results
│   └── plots/                  # Visualizations (PNG, PDF)
├── qualitative_eval/           # Qualitative comparison suite
│   ├── prompts.json            # 20 diverse prompts
│   ├── run_inference.py        # Multi-model inference
│   └── README.md
└── docs/
    └── 01_sample_prompts.md    # Manual testing prompts
```

## 🎯 Supported Benchmarks

| Benchmark | Dataset | Task | Samples |
|-----------|---------|------|---------|
| **OpenBookQA** | `allenai/openbookqa` | Science + common sense reasoning | 500 |
| **ARC-Challenge** | `allenai/ai2_arc` (ARC-Challenge) | Grade-school science (hard) | 1,172 |
| **ARC-Easy** | `allenai/ai2_arc` (ARC-Easy) | Grade-school science (easy) | 2,376 |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Use the training venv (has torch, transformers)
source /raid/zhf004/llm_TII/venv/bin/activate

# Install datasets library if needed
pip install datasets tqdm
```

### 2. Run Benchmark Evaluation

```bash
cd /raid/zhf004/llm_TII/evaluation_system

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Log-Probability Scoring (Default - Deterministic)
# ══════════════════════════════════════════════════════════════════════════════

# Evaluate on ALL benchmarks
CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmarks.py \
    --checkpoint /raid/zhf004/llm_TII/enhanced_training_system/out-qwen3-1.8b-b200-50h/ckpt_160000.pt \
    --tokenizer /raid/zhf004/llm_TII/enhanced_training_system/qwen3_tokenizer \
    --mode logprob

# Evaluate SFT checkpoint
CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmarks.py \
    --checkpoint /raid/zhf004/llm_TII/post_training/out-qwen3-1.8b-sft/ckpt_002800.pt \
    --tokenizer /raid/zhf004/llm_TII/enhanced_training_system/qwen3_tokenizer \
    --output eval_sft.json

# Evaluate DPO checkpoint
CUDA_VISIBLE_DEVICES=1 python scripts/eval_benchmarks.py \
    --checkpoint /raid/zhf004/llm_TII/post_training/out-qwen3-1.8b-dpo/ckpt_000800.pt \
    --tokenizer /raid/zhf004/llm_TII/enhanced_training_system/qwen3_tokenizer \
    --output eval_dpo.json

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: Generation-Based (With Sampling Parameters)
# ══════════════════════════════════════════════════════════════════════════════

CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmarks.py \
    --checkpoint /path/to/ckpt.pt \
    --mode generate \
    --temperature 0.3 \
    --max-tokens 5 \
    --top-k 10
```

### 3. Run Qualitative Evaluation

Compare generations across Base → SFT → DPO:

```bash
cd /raid/zhf004/llm_TII/evaluation_system

# Full evaluation (20 prompts × 3 models × 3 temps × 3 max_tokens)
CUDA_VISIBLE_DEVICES=0 python qualitative_eval/run_inference.py

# Quick test
CUDA_VISIBLE_DEVICES=0 python qualitative_eval/run_inference.py \
    --prompt-ids 1,5,9 \
    --temperatures 0.7 \
    --max-tokens-list 128
```

### 4. Generate Plots

```bash
cd /raid/zhf004/llm_TII/evaluation_system
python scripts/plot_comparison.py
# Output: results/plots/benchmark_comparison.{png,pdf}
```

## 📊 Results

Results are saved to `results/benchmark/`:

```json
{
  "timestamp": "2024-12-05T...",
  "checkpoint": "/path/to/ckpt.pt",
  "benchmarks": {
    "openbookqa": {"accuracy": 0.29, "correct": 145, "total": 500},
    "arc_challenge": {"accuracy": 0.295, "correct": 346, "total": 1172},
    "arc_easy": {"accuracy": 0.55, "correct": 1305, "total": 2376}
  }
}
```

## 📈 Latest Results (Dec 2024)

| Benchmark | Base | SFT | DPO |
|-----------|------|-----|-----|
| OpenBookQA | 26.4% | 29.0% | 27.2% |
| ARC-Challenge | 27.2% | 29.5% | 29.4% |
| ARC-Easy | 52.6% | 52.6% | 54.9% |

**Key Findings:**
- SFT improves OpenBookQA (+2.6%) and ARC-Challenge (+2.3%)
- DPO improves ARC-Easy (+2.3%)

## 📊 Expected Results for 1.8B Base Model

| Benchmark | Random Baseline | Expected Range | Notes |
|-----------|-----------------|----------------|-------|
| OpenBookQA | 25% | 28-35% | Requires world knowledge |
| ARC-Challenge | 25% | 25-32% | Harder reasoning |
| ARC-Easy | 25% | 40-50% | Simpler factual |

**Note:** These are base model expectations. Instruction-tuned models score 10-20% higher.

## 🔧 Evaluation Methods

### Mode 1: Log-Probability Scoring (`--mode logprob`, default)

**Deterministic** - no sampling, standard for benchmarks.

1. For each question, format as: `"Question: X\nAnswer: Y"`
2. Compute average log-prob of the full sequence for each choice
3. Predict the choice with the highest log-prob

### Mode 2: Generation-Based (`--mode generate`)

**Uses sampling** - tests actual text generation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 0.3 | Sampling temperature |
| `--max-tokens` | 5 | Max tokens to generate |
| `--top-k` | 10 | Top-k sampling |
| `--repetition-penalty` | 1.0 | Repetition penalty |

## 📝 Command Reference

```bash
# Full evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmarks.py

# Single benchmark
CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmarks.py --benchmark openbookqa

# Quick test (100 samples)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmarks.py --max-samples 100

# Custom output name
CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmarks.py --output my_results.json
```

## 🔗 Related

- Post-training: `../post_training/`
- Training system: `../enhanced_training_system/`
- Serving system: `../serving_system/`
