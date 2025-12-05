# SFT Implementation Details

## Overview

Supervised Fine-Tuning (SFT) teaches a base language model to follow instructions by training on (instruction, response) pairs.

## Core Components

### 1. Training Script: `train_sft.py`

**Location**: `/raid/zhf004/llm_TII/post_training/train_sft.py`

**Key Classes and Functions**:

```python
class SFTDataset(Dataset):
    """
    Loads instruction-following data in JSONL or binary format.
    Handles padding and truncation to block_size.
    """
    
def compute_sft_loss(model, input_ids, labels, ignore_index=-100):
    """
    Computes cross-entropy loss with masking.
    Labels with value -100 are ignored (instruction tokens).
    """
    
def load_checkpoint(checkpoint_path, device):
    """
    Loads pre-trained model using ConfigurableGPT from model_builder.
    Handles state dict key prefix cleanup.
    """
```

### 2. Loss Masking

The key innovation in SFT is **loss masking** - only computing loss on the assistant's response tokens:

```
<|im_start|>system
You are helpful.<|im_end|>         ← labels = -100 (masked)
<|im_start|>user
What is 2+2?<|im_end|>             ← labels = -100 (masked)
<|im_start|>assistant
2+2 equals 4.<|im_end|>            ← labels = token_ids (loss computed)
```

**Implementation** (in `prepare_sft.py`):

```python
def _create_masked_labels(self, text, tokens):
    labels = [-100] * len(tokens)  # Start all masked
    
    # Find assistant response boundaries
    assistant_marker = "<|im_start|>assistant\n"
    
    # Unmask only assistant response tokens
    for response_span in find_assistant_spans(text):
        for i in range(response_span.start, response_span.end):
            labels[i] = tokens[i]
    
    return labels
```

### 3. ChatML Format

Qwen3 uses the ChatML template:

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_message}<|im_end|>
```

**Special Token IDs** (Qwen3):
- `<|im_start|>`: 151644
- `<|im_end|>`: 151645
- `<|endoftext|>`: 151643 (used for padding)

### 4. Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2e-5 | 10× lower than pre-training to avoid catastrophic forgetting |
| Min LR | 2e-6 | Cosine decay floor |
| Batch Size | 4 per GPU | Balanced for memory/throughput |
| Grad Accum | 4 | Effective batch = 128 (4×4×8 GPUs) |
| Max Iters | 3000 | ~3 epochs over 50K examples |
| Warmup | 100 iters | Short warmup for fine-tuning |
| Dropout | 0.05 | Slight regularization |
| Weight Decay | 0.01 | Lower than pre-training |

### 5. Training Loop

```python
while iter_num < max_iters:
    # 1. Update learning rate (cosine schedule)
    lr = get_lr(iter_num, warmup_iters, lr_decay_iters, learning_rate, min_lr)
    
    # 2. Periodic evaluation
    if iter_num % eval_interval == 0:
        losses = estimate_loss(model, train_loader, val_loader)
        save_checkpoint()
    
    # 3. Gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        batch = get_next_batch()
        loss = compute_sft_loss(model, batch['input_ids'], batch['labels'])
        loss.backward()
    
    # 4. Gradient clipping & optimizer step
    clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
```

## Model Loading

The script loads the pre-trained checkpoint and creates the model using the modular architecture:

```python
from model_builder import ConfigurableGPT
from model_config import ModelArchitectureConfig

checkpoint = torch.load(checkpoint_path)
model_args = checkpoint['model_args']

arch_config = ModelArchitectureConfig.from_dict(model_args)
model = ConfigurableGPT(arch_config)
model.load_state_dict(checkpoint['model'])
```

## Distributed Training

Supports multi-GPU training via PyTorch DDP:

```bash
torchrun --standalone --nproc_per_node=8 train_sft.py configs/sft_qwen3_1.8b.py
```

**DDP Features**:
- Automatic gradient synchronization
- DistributedSampler for data sharding
- Master process handles logging/checkpointing

## Output

After training:

```
out-qwen3-1.8b-sft/
├── ckpt_000200.pt    # Checkpoint at iter 200
├── ckpt_000400.pt    # Checkpoint at iter 400
├── ...
├── ckpt.pt           # Latest/best checkpoint
└── run_*.json        # Training logs
```

## Expected Results

| Metric | Base Model | After SFT |
|--------|-----------|-----------|
| Instruction Following | ❌ Poor | ✅ Good |
| Format Compliance | ❌ Random | ✅ Structured |
| Conversational | ❌ No | ✅ Yes |
| Train Loss | - | ~1.2-1.5 |
| Val Loss | - | ~1.3-1.6 |

