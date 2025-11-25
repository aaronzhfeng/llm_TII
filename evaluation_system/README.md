# Qwen3-1.8B Evaluation System

Benchmark evaluation suite for the custom-trained Qwen3 model.

## 📁 Directory Structure

```
llm_TII/
├── enhanced_training_system/   # Training code
├── serving_system/             # Inference server
└── evaluation_system/          # ← You are here
    ├── eval_benchmarks.py      # Main evaluation script
    ├── sample_prompts.md       # Manual testing prompts
    └── README.md
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
cd /raid/zhf004/llm_TII/evaluation_system

# Use the training venv (has torch, transformers)
source /raid/zhf004/llm_TII/venv/bin/activate

# Install datasets library if needed
pip install datasets tqdm
```

### 2. Run Evaluation

```bash
# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Log-Probability Scoring (Default - Deterministic)
# ══════════════════════════════════════════════════════════════════════════════

# Evaluate on ALL benchmarks
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py \
    --checkpoint /raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_160000.pt \
    --tokenizer /raid/zhf004/llm_TII/enhanced_training_system/qwen3_tokenizer \
    --mode logprob

# Evaluate on specific benchmark
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py \
    --checkpoint /raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_160000.pt \
    --tokenizer /raid/zhf004/llm_TII/enhanced_training_system/qwen3_tokenizer \
    --mode logprob \
    --benchmark arc_easy

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: Generation-Based (With Sampling Parameters)
# ══════════════════════════════════════════════════════════════════════════════

# Generation mode with custom temperature, max_tokens, top_k
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py \
    --checkpoint /raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_160000.pt \
    --tokenizer /raid/zhf004/llm_TII/enhanced_training_system/qwen3_tokenizer \
    --mode generate \
    --temperature 0.3 \
    --max-tokens 5 \
    --top-k 10 \
    --repetition-penalty 1.2

# Quick test with generation mode (100 samples)
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py \
    --checkpoint /raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_160000.pt \
    --tokenizer /raid/zhf004/llm_TII/enhanced_training_system/qwen3_tokenizer \
    --mode generate \
    --temperature 0.2 \
    --max-tokens 3 \
    --top-k 5 \
    --max-samples 100 \
    --benchmark arc_easy

# Use different checkpoint
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py \
    --checkpoint /raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_080000.pt
```

### 3. View Results

Results are saved to `eval_results_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "2024-11-24T...",
  "checkpoint": "/raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_160000.pt",
  "benchmarks": {
    "openbookqa": {"accuracy": 0.32, "correct": 160, "total": 500},
    "arc_challenge": {"accuracy": 0.28, "correct": 328, "total": 1172},
    "arc_easy": {"accuracy": 0.45, "correct": 1069, "total": 2376}
  }
}
```

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

**Prompt Format:**
```
Question: What is the largest planet in our solar system?
Answer: Jupiter
```

**Pros:** Deterministic, reproducible, standard academic approach
**Cons:** Doesn't test actual generation capability

---

### Mode 2: Generation-Based (`--mode generate`)

**Uses sampling** - tests actual text generation with format constraints.

1. Format question with choices and instruction: "Answer with only the letter (A, B, C, or D):"
2. Generate response using temperature, top_k, etc.
3. Extract first valid letter (A/B/C/D) from output

**Prompt Format:**
```
Question: What is the largest planet in our solar system?

A. Mars
B. Jupiter
C. Saturn
D. Earth

Answer with only the letter (A, B, C, or D):
```

**Generation Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--temperature` | 0.3 | Sampling temperature (lower = more deterministic) |
| `--max-tokens` | 5 | Max tokens to generate (keep low for letter output) |
| `--top-k` | 10 | Top-k sampling |
| `--repetition-penalty` | 1.0 | Repetition penalty |

**Pros:** Tests real generation, more realistic
**Cons:** Non-deterministic, may output invalid format

## 📝 Command Reference

```bash
# Full evaluation on all benchmarks
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py

# Single benchmark
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py --benchmark openbookqa
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py --benchmark arc_challenge
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py --benchmark arc_easy

# Quick test (100 samples per benchmark)
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py --max-samples 100

# Specific checkpoint
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py \
    --checkpoint /raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_100000.pt

# Save to specific file
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py --output my_results.json

# CPU only (slow)
python eval_benchmarks.py --device cpu --max-samples 50
```

## 🔄 Compare Checkpoints

To compare different training stages:

```bash
# Iteration 80k
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py \
    --checkpoint /raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_080000.pt \
    --output eval_iter80k.json

# Iteration 160k
CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py \
    --checkpoint /raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_160000.pt \
    --output eval_iter160k.json
```

## 📚 Sample Prompts

See `sample_prompts.md` for manual testing prompts optimized for base models.

## 🔗 Related

- Serving system: `../serving_system/`
- Training docs: `../enhanced_training_system/docs/`
- Checkpoints: `/raid/zhf004/out-qwen3-1.8b-b200-50h/`

