# DPO Implementation Details

## Overview

Direct Preference Optimization (DPO) aligns language models with human preferences without requiring a separate reward model. It's simpler than RLHF while achieving comparable results.

## Core Concept

DPO uses pairs of responses:
- **Chosen (y_w)**: The preferred response
- **Rejected (y_l)**: The less preferred response

The model learns to increase probability of chosen responses relative to rejected ones.

## DPO Loss Function

```
L_DPO = -log σ(β × (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
```

**Components**:
- `π` = Policy model (being trained)
- `π_ref` = Reference model (frozen SFT checkpoint)
- `y_w` = Chosen response
- `y_l` = Rejected response
- `β` = KL penalty coefficient (typically 0.1)
- `σ` = Sigmoid function

**Intuition**: The loss pushes the model to prefer chosen responses over rejected ones, while staying close to the reference model (preventing reward hacking).

## Implementation

### Training Script: `train_dpo.py`

**Key Components**:

```python
class DPODataset(Dataset):
    """
    Loads preference pairs: (prompt, chosen, rejected)
    Returns tokenized versions of both responses.
    """

def compute_dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """
    Compute DPO loss for a batch of preference pairs.
    """
    # Get log probs from policy model
    policy_chosen_logps = get_log_probs(policy_model, chosen_ids, chosen_labels)
    policy_rejected_logps = get_log_probs(policy_model, rejected_ids, rejected_labels)
    
    # Get log probs from reference model (no grad)
    with torch.no_grad():
        ref_chosen_logps = get_log_probs(ref_model, chosen_ids, chosen_labels)
        ref_rejected_logps = get_log_probs(ref_model, rejected_ids, rejected_labels)
    
    # DPO loss
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    
    return losses.mean()
```

### Reference Model

DPO requires keeping a frozen copy of the SFT model as the reference:

```python
# Load SFT checkpoint for both policy and reference
policy_model = load_checkpoint(sft_checkpoint_path)
ref_model = load_checkpoint(sft_checkpoint_path)

# Freeze reference model
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()
```

**Memory Note**: This doubles GPU memory usage since two models are loaded.

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 5e-7 | Very low - DPO is sensitive |
| Beta (β) | 0.1 | KL penalty strength (0.05-0.5 typical) |
| Batch Size | 2 per GPU | Memory constrained (2 models) |
| Grad Accum | 8 | Effective batch = 128 |
| Max Iters | 1000 | Fewer iterations than SFT |
| Warmup | 50 iters | Short warmup |

## Data Format

DPO datasets contain preference pairs:

```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris, a major European city known for...",
  "rejected": "France capital is Paris I think maybe London?"
}
```

### Supported Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| `ultrafeedback` | 60K | UltraFeedback binarized preferences |
| `hh_rlhf` | 170K | Anthropic HH-RLHF |
| `orca_dpo` | 12K | Intel Orca DPO pairs |

## Training Pipeline

```
┌─────────────────────┐
│   SFT Checkpoint    │
│   (from Stage 1)    │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌────────┐
│ Policy │   │  Ref   │
│ Model  │   │ Model  │
│(train) │   │(frozen)│
└───┬────┘   └───┬────┘
    │            │
    └─────┬──────┘
          ▼
    ┌───────────┐
    │ DPO Loss  │
    └─────┬─────┘
          ▼
    ┌───────────┐
    │  Update   │
    │  Policy   │
    └───────────┘
```

## Expected Results

| Metric | After SFT | After DPO |
|--------|-----------|-----------|
| Hallucination | Medium | Low |
| Helpfulness | Good | Better |
| Harmlessness | Variable | Improved |
| Response Quality | Good | Better |

## Usage

```bash
# 1. Prepare DPO data
python data/prepare_dpo.py \
    --dataset ultrafeedback \
    --output_dir data/dpo_ultrafeedback

# 2. Update config with SFT checkpoint path
# Edit configs/dpo_qwen3_1.8b.py

# 3. Run DPO training
torchrun --standalone --nproc_per_node=8 train_dpo.py configs/dpo_qwen3_1.8b.py
```

## Troubleshooting

### Out of Memory
DPO loads two models. Solutions:
- Reduce batch size to 1
- Use gradient checkpointing
- Use fewer GPUs with more memory each

### Reward Hacking
If the model produces degenerate outputs:
- Increase β (stronger KL penalty)
- Reduce learning rate
- Train for fewer iterations

### No Improvement
If metrics don't improve:
- Check data quality (chosen should be clearly better)
- Try different β values
- Ensure SFT model is good first

