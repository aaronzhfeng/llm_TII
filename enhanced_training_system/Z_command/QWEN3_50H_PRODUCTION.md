# 🚀 Qwen3 1.8B — 50 Hour Production Run (117B Tokens)

This playbook launches the finalized configuration for the 50-hour production training window on 8× B200 GPUs. It checkpoints every ~30 minutes (every 1 000 iterations) and streams metrics to Weights & Biases.

## 1. Configuration

- Config file: `config/full_qwen3_1.8b_b200_50h.py`
- Tokens per iteration: `22 × 8 × 2 × 2048 = 720,896`
- Estimated iterations in 50 h: `~162,000` → `~117B` tokens total
- ZeRO-1 remains enabled for memory stability under `torch.compile`
- WandB logging is on (`qwen3-1.8b-b200 / 50h-production-117B`)

## 2. Launch Command

```bash
cd /raid/zhf004/llm_TII/enhanced_training_system
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_50h.py
```

> 🔐 Checkpoints are forced at every evaluation interval (`eval_interval = 1000`). Resume by pointing `init_from` to the latest checkpoint directory if needed.

## 3. Monitoring Checklist

- Watch W&B dashboard `qwen3-1.8b-b200/50h-production-117B`
- Expected throughput: **~650k tokens/s**
- Validation refresh: every 1 000 iterations (~30 min)
- Stop early if validation loss plateaus for >10 k iterations

## 4. Optional Adjustments

- If stability is excellent for >10 k iterations, consider a controlled restart with a modest micro-batch bump (`batch_size = 26`, `grad_accum = 2`) to target ~138B tokens. Test on a short pilot run first.
- Toggle `use_sparse_specs` inside the config if benchmarking sparse MFU denominators on B200.

## 5. Smoke-Test the Pipeline

Before committing a 50 hour run, you can trigger a short “smoke test” that keeps all production hyperparameters (same model, batch size, LR, ZeRO-1, etc.) but runs only 20 iterations with frequent eval/checkpoint/logs. This is useful after regenerating datasets or touching the training stack.

```bash
cd /raid/zhf004/llm_TII/enhanced_training_system
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_50h_smoke.py
```

That job will run for just a few minutes, hit `eval_at_start`, `eval_interval=5`, and `always_save_checkpoint`, so you can confirm logging, evaluation, and checkpoint writing work with the newly prepared `slimpajama_627b_qwen3` dataset.


