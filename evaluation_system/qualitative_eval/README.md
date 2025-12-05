# Qualitative Evaluation

Compare model responses across training stages (Base → SFT → DPO) with varying generation parameters.

Results are saved to `../results/qualitative/`.

## Quick Start

```bash
cd /raid/zhf004/llm_TII/evaluation_system
source /raid/zhf004/llm_TII/venv/bin/activate

# Run all 20 prompts on all 3 models with default settings
CUDA_VISIBLE_DEVICES=0 python qualitative_eval/run_inference.py

# Quick test: 3 prompts, 1 temperature
CUDA_VISIBLE_DEVICES=0 python qualitative_eval/run_inference.py \
    --prompt-ids 1,5,9 \
    --temperatures 0.7 \
    --max-tokens-list 128

# Single model comparison
CUDA_VISIBLE_DEVICES=0 python qualitative_eval/run_inference.py \
    --models base,sft \
    --temperatures 0.3,0.7

# Custom checkpoints
CUDA_VISIBLE_DEVICES=0 python qualitative_eval/run_inference.py \
    --base-ckpt /path/to/base.pt \
    --sft-ckpt /path/to/sft.pt \
    --dpo-ckpt /path/to/dpo.pt
```

## Prompts (20 total, 10 categories)

| Category | Count | Examples |
|----------|-------|----------|
| Coding | 2 | Palindrome function, list vs tuple |
| Math | 2 | Derivatives, word problems |
| Science | 2 | Sky color, earthquakes |
| History | 2 | Moon landing, WWII |
| Creative | 2 | Poetry, jokes |
| Reasoning | 2 | Logic, arithmetic |
| Advice | 2 | Sleep tips, motivation |
| Explanation | 2 | Neural networks, weather vs climate |
| Conversation | 2 | Greetings, preferences |
| Task | 2 | Summarization, lists |

## Default Settings

| Parameter | Default Values |
|-----------|----------------|
| Temperatures | 0.3, 0.7, 1.0 |
| Max tokens | 64, 128, 256 |
| Models | base, sft, dpo |

**Total combinations per prompt:** 3 models × 3 temps × 3 max_tokens = 27

## Output

Results saved to `../results/qualitative/qualitative_results_YYYYMMDD_HHMMSS.json`:

```json
{
  "timestamp": "2024-12-05T...",
  "config": {
    "temperatures": [0.3, 0.7, 1.0],
    "max_tokens": [64, 128, 256],
    "models": ["base", "sft", "dpo"]
  },
  "generations": [
    {
      "model": "base",
      "prompt_id": 1,
      "category": "coding",
      "prompt": "Write a Python function...",
      "temperature": 0.7,
      "max_tokens": 128,
      "response": "def is_palindrome(s):...",
      "generation_time": 2.34
    }
  ]
}
```

## Analysis Tips

```python
import json
import pandas as pd

# Load results
with open('qualitative_results_*.json') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['generations'])

# Compare response lengths by model
df.groupby('model')['response'].apply(lambda x: x.str.len().mean())

# Compare by category
df.groupby(['category', 'model'])['response'].apply(lambda x: x.str.len().mean()).unstack()
```

