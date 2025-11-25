# Z_command: Testing & Training Execution Guides

This directory contains **executable guides** for running training experiments and MFU tests on the enhanced training system.

---

## 📋 Contents

### B200 MFU Testing Guides

- **`B200_LLAMA2_MFU_TEST.md`** - LLaMA 2 1.36B MFU testing on B200 GPUs
- **`B200_QWEN3_MFU_TEST.md`** - Qwen 3 1.8B MFU testing on B200 GPUs

### Legacy MFU Testing Guides (H20/H100)

- **`MFU_TESTING_LLAMA2.md`** - LLaMA 2 MFU testing (legacy hardware)
- **`MFU_TESTING_QWEN3.md`** - Qwen 3 MFU testing (legacy hardware)

### General Guides

- **`TESTING.md`** - Comprehensive testing guide for all architectures
- **`TRAINING_GUIDE.md`** - Complete training workflow and best practices

---

## 🔗 Related Documentation

For analysis, implementation details, and theoretical background, see the **`docs/`** folder:

- `docs/41_mfu_comparison_nanogpt.md` - nanoGPT vs Enhanced MFU comparison
- `docs/42_audit_compliance_summary.md` - 2025 MFU audit compliance report
- `docs/43_mfu_calculation_fix.md` - Bug fixes and methodology
- `docs/44_oom_analysis_zero1.md` - Memory optimization analysis

---

## 🚀 Quick Start

### Test Qwen 3 1.8B on 8× B200

```bash
cd /raid/zhf004/llm_TII/enhanced_training_system

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train.py \
  config/full_qwen3_1.8b_b200_optimal.py \
  --max_iters=200 \
  --batch_size=24 \
  --gradient_accumulation_steps=2 \
  --compile=True \
  --always_save_checkpoint=False
```

See `B200_QWEN3_MFU_TEST.md` for detailed instructions.

---

## 📊 Expected Performance

| Model | Hardware | Expected MFU | Tokens/sec |
|-------|----------|--------------|------------|
| Qwen 3 1.8B | 8× B200 | 43-45% | ~650,000 |
| LLaMA 2 1.36B | 8× B200 | 40-42% | ~700,000 |

---

**Last Updated**: 2025-11-21

