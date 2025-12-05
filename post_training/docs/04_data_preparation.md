# Data Preparation Guide

## Overview

Post-training requires specially formatted datasets:
- **SFT**: Instruction-response pairs in ChatML format
- **DPO**: Preference pairs (prompt, chosen, rejected)

## SFT Data Preparation

### Script: `data/prepare_sft.py`

**Usage**:
```bash
python data/prepare_sft.py \
    --dataset alpaca \
    --output_dir data/sft_alpaca \
    --max_length 2048 \
    --format jsonl
```

### Supported Datasets

| Dataset | HuggingFace Path | Examples | Description |
|---------|------------------|----------|-------------|
| `alpaca` | `tatsu-lab/alpaca` | 52K | Stanford Alpaca (recommended for testing) |
| `alpaca_cleaned` | `yahma/alpaca-cleaned` | 52K | Cleaned version, higher quality |
| `dolly` | `databricks/databricks-dolly-15k` | 15K | Human-generated |
| `ultrachat` | `HuggingFaceH4/ultrachat_200k` | 200K | High-quality synthetic |
| `slimorca` | `Open-Orca/SlimOrca` | 517K | Cleaned OpenOrca |

### Processing Pipeline

```
Raw Dataset (HuggingFace)
         │
         ▼
┌─────────────────────┐
│  Format Converter   │  ← Handles different dataset formats
│  (alpaca/dolly/etc) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  ChatML Formatter   │  ← Applies Qwen3 chat template
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│    Tokenizer        │  ← Qwen3 tokenizer
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Label Masking     │  ← Mask instruction tokens (-100)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Save to Disk      │  ← JSONL or binary format
└─────────────────────┘
```

### ChatML Formatting

The `ChatMLFormatter` class handles conversion:

```python
class ChatMLFormatter:
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."
    
    def format_conversation(self, instruction, response, input_text=None):
        """Convert to ChatML format"""
        chat = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
        
        if input_text:
            user_content = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_content = instruction
        
        chat += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        chat += f"<|im_start|>assistant\n{response}<|im_end|>"
        
        return chat
```

### Output Format

**JSONL** (recommended for debugging):
```json
{"input_ids": [151644, 9125, 198, ...], "labels": [-100, -100, -100, ..., 220, 17, 151645]}
```

**Binary** (faster loading):
- `train.bin`: Concatenated input_ids
- `train_labels.bin`: Concatenated labels
- `meta.json`: Statistics

### Current Prepared Data

```
data/sft_alpaca/
├── train.jsonl    # 50,935 examples
├── val.jsonl      # 1,039 examples
└── meta.json      # Statistics
```

**Statistics** (Alpaca):
- Total examples: 51,974
- Total tokens: 5,734,519
- Avg tokens/example: 110.3
- Max tokens: 1,120
- Truncated: 0

## DPO Data Preparation

### Script: `data/prepare_dpo.py`

**Usage**:
```bash
python data/prepare_dpo.py \
    --dataset ultrafeedback \
    --output_dir data/dpo_ultrafeedback \
    --max_samples 50000
```

### Supported Datasets

| Dataset | HuggingFace Path | Pairs | Description |
|---------|------------------|-------|-------------|
| `ultrafeedback` | `argilla/ultrafeedback-binarized-preferences` | 60K | GPT-4 judged preferences |
| `hh_rlhf` | `Anthropic/hh-rlhf` | 170K | Human preferences |
| `orca_dpo` | `Intel/orca_dpo_pairs` | 12K | Curated pairs |

### Data Format

Each example contains:
```json
{
  "prompt": "What is machine learning?",
  "chosen": "Machine learning is a subset of AI that enables systems to learn...",
  "rejected": "Machine learning is when computers learn stuff automatically."
}
```

### Processing

1. Load from HuggingFace
2. Apply ChatML template to prompt
3. Tokenize chosen and rejected responses separately
4. Create labels with loss masking
5. Save as JSONL

## Custom Datasets

### Adding a New SFT Dataset

1. Add entry to `SUPPORTED_DATASETS`:
```python
SUPPORTED_DATASETS = {
    "my_dataset": {
        "hf_name": "username/my-dataset",
        "description": "My custom dataset",
        "format": "alpaca",  # or create new format
    },
}
```

2. If new format, add converter:
```python
def convert_myformat(example, formatter):
    instruction = example.get("question", "")
    response = example.get("answer", "")
    text = formatter.format_conversation(instruction, response)
    return {"text": text, "instruction": instruction, "response": response}

CONVERTERS["myformat"] = convert_myformat
```

### Using Local Files

For local JSONL files:
```python
# Each line should have: instruction, input (optional), output
{"instruction": "Summarize this", "input": "Long text...", "output": "Summary"}
```

Load with:
```bash
python data/prepare_sft.py --dataset local --data_path /path/to/data.jsonl
```

## Tokenizer

Uses Qwen3 tokenizer loaded from HuggingFace:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

**Special Tokens**:
| Token | ID | Usage |
|-------|-----|-------|
| `<\|im_start\|>` | 151644 | Start of message |
| `<\|im_end\|>` | 151645 | End of message |
| `<\|endoftext\|>` | 151643 | Padding / EOS |

## Best Practices

1. **Start with Alpaca**: Good baseline, fast to process
2. **Check data quality**: Inspect random samples before training
3. **Use JSONL format**: Easier to debug than binary
4. **Reasonable max_length**: 2048 is good for most instruction data
5. **Validation split**: Keep ~2% for monitoring overfitting

