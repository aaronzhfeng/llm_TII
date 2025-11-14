# Training Results Visualization Guide

**Date:** November 14, 2025  
**Phase:** Results Analysis & Visualization  
**Status:** Complete

---

## Overview

This guide describes how to generate comprehensive training analysis plots for all completed training runs. The plotting system automatically generates publication-quality visualizations with proper model names and metrics.

---

## Quick Start

### Generate All Plots

```bash
cd /root/llm_TII/enhanced_training_system/plots
python plot_all_runs.py
```

This will generate 4 plots (one per model) in the `saves/` directory:
- `Qwen3_1.8B_Optimal_analysis.png`
- `GPT2_1.29B_analysis.png`
- `LLaMA3_2.2B_Chinchilla_analysis.png`
- `LLaMA2_1.36B_analysis.png`

---

## Training Runs Analyzed

### 1. Qwen3 1.8B Optimal
- **Architecture:** 24L-16H-2048D-6144ff with GQA (8 KV heads)
- **Parameters:** 1.83B
- **Run Date:** November 13, 2025
- **Optimal Tokens:** 81.7B
- **Training:** 2000 iterations on 2× A6000

### 2. GPT-2 1.29B
- **Architecture:** 18L-18H-2304D-9216ff (standard FFN 4x)
- **Parameters:** 1.29B
- **Run Date:** November 13, 2025
- **Optimal Tokens:** 27B
- **Training:** 2000 iterations on 2× A6000

### 3. LLaMA 3 2.2B Chinchilla
- **Architecture:** 30L-16H-2048D-7168ff with GQA (8 KV heads)
- **Parameters:** 2.22B
- **Run Date:** November 12, 2025
- **Optimal Tokens:** 61.5B
- **Training:** 2000 iterations on 2× A6000

### 4. LLaMA 2 1.36B
- **Architecture:** 18L-18H-2304D-6144ff (SwiGLU)
- **Parameters:** 1.36B
- **Run Date:** November 10, 2025
- **Optimal Tokens:** 27B
- **Training:** 2000 iterations on 2× A6000

---

## What Each Plot Shows

Each comprehensive analysis plot includes **6 subplots**:

### Row 1: Training Progress

1. **Loss & Perplexity (Dual Y-axis)**
   - Training loss (blue line)
   - Perplexity = exp(loss) (green dashed line)
   - Validation loss at checkpoints (red stars)

2. **Learning Rate Schedule**
   - Cosine decay with warmup
   - Warmup end marked (red dashed line)
   - Log scale for clarity

3. **Cumulative Tokens**
   - Total tokens processed (billions)
   - Progress vs optimal target shown
   - Annotation showing % completion

### Row 2: Performance Metrics

4. **Model FLOPs Utilization (MFU)**
   - MFU % over iterations
   - Average MFU shown (blue dashed line)
   - Typical range: 25-40% on 2× A6000

5. **Training Throughput**
   - Tokens per second
   - Average throughput shown
   - Varies by model size: 5k-15k tokens/s

6. **GPU Memory Usage**
   - Allocated memory (blue line)
   - Peak memory (red line)
   - GPU capacity limit (48 GB A6000)

---

## Key Results Summary

### Performance Comparison (2× A6000, ZeRO-1)

| Model | Final Loss | Val Loss | Avg MFU | Throughput | Peak Memory |
|-------|-----------|----------|---------|------------|-------------|
| **Qwen3 1.8B** | ~2.07 | 2.02 | 27-28% | ~7,000 t/s | 28.7 GB |
| **GPT-2 1.29B** | ~2.15 | 2.10 | 35-40% | ~12,000 t/s | 25-28 GB |
| **LLaMA3 2.2B** | ~2.15 | 2.10 | 24-25% | ~5,100 t/s | 40.9 GB |
| **LLaMA2 1.36B** | ~2.20 | 2.15 | 40-45% | ~10,000 t/s | 30-35 GB |

### Key Observations

1. **Qwen3 1.8B** achieved the **best validation loss** (2.02) with:
   - 24 layers (deepest architecture)
   - Extended RoPE (theta=1M)
   - Best tokenizer (152K vocab)
   - Only 27-28% MFU (lower due to larger vocab & deeper model)

2. **GPT-2 1.29B** achieved the **highest MFU** (35-40%) with:
   - Simplest architecture (standard FFN)
   - Smallest vocabulary (50K)
   - Highest throughput (12k tokens/s)
   - Good for testing/debugging

3. **LLaMA3 2.2B** is the **largest model** (2.22B params):
   - Requires most memory (40.9 GB peak)
   - Lowest MFU (24-25%) due to size
   - Lowest throughput (5.1k tokens/s)
   - Best for downstream task performance

4. **LLaMA2 1.36B** is the **best balanced** model:
   - High MFU (40-45%)
   - Good throughput (10k tokens/s)
   - Reasonable loss (2.15)
   - Moderate memory usage

---

## Customizing Plots

### Adding New Runs

To add a new training run to the plotting system:

1. Add entry to `MODEL_INFO` dict in `plot_all_runs.py`:

```python
MODEL_INFO = {
    "run_20251114_xxxxxx": {
        "name": "YourModel_Size",
        "full_name": "Your Model Description (NL-NH-ND)",
        "params": "X.XXB",
        "optimal_tokens": "XXB"
    },
    # ... existing entries
}
```

2. Run the script:

```bash
cd /root/llm_TII/enhanced_training_system/plots
python plot_all_runs.py
```

### Modifying Plot Style

Key parameters to adjust in `plot_all_runs.py`:

```python
# Figure size
fig = plt.figure(figsize=(18, 10))  # Width x Height in inches

# DPI (resolution)
plt.savefig(output_path, dpi=300, ...)  # Higher = sharper but larger file

# Line styles
linewidth=2  # Thickness of lines
marker='o'   # Point markers
markersize=2 # Size of markers
alpha=0.8    # Transparency (0=transparent, 1=opaque)
```

---

## Troubleshooting

### Issue: "File not found" error

**Solution:** Verify JSON files exist in `saves/` directory:

```bash
ls -lh /root/llm_TII/enhanced_training_system/saves/*.json
```

### Issue: Import error (matplotlib)

**Solution:** Install matplotlib:

```bash
pip install matplotlib numpy
```

### Issue: Plot looks wrong

**Solution:** Check data integrity:

```bash
# Verify JSON is valid
python -c "import json; json.load(open('../saves/run_20251113_225009.json'))"
```

### Issue: Out of memory when plotting

**Solution:** Reduce DPI or plot one at a time:

```bash
# Plot single run
python plot_single_run.py ../saves/run_20251113_225009.json
```

---

## Advanced Usage

### Plot Single Run with Custom Name

Modify `plot_single_run.py` or use it directly:

```bash
python plot_single_run.py ../saves/run_20251113_225009.json
```

### Batch Convert to Different Formats

```bash
# Convert all PNG to PDF
for f in ../saves/*_analysis.png; do
    convert "$f" "${f%.png}.pdf"
done
```

### Create Comparison Plot

For a side-by-side comparison of all models, use:

```bash
python compare_runs.py
```

This generates a single plot comparing all 4 models.

---

## File Locations

### Input Files
- Training logs: `/root/llm_TII/enhanced_training_system/saves/run_*.json`

### Output Files
- Individual plots: `/root/llm_TII/enhanced_training_system/saves/*_analysis.png`
- Comparison plot: `/root/llm_TII/enhanced_training_system/plots/training_comparison.png`

### Scripts
- Batch plotting: `/root/llm_TII/enhanced_training_system/plots/plot_all_runs.py`
- Single run: `/root/llm_TII/enhanced_training_system/plots/plot_single_run.py`
- Comparison: `/root/llm_TII/enhanced_training_system/plots/compare_runs.py`

---

## Next Steps

1. **Generate Plots:** Run `python plot_all_runs.py`
2. **Review Results:** Open the PNG files in image viewer
3. **Share Results:** Copy plots to presentation/paper
4. **Document Findings:** Update project documentation with key insights

---

## References

- Training logs format: See `training_logger.py`
- Config files: `enhanced_training_system/config/`
- Training guide: `TRAINING_GUIDE.md`
- System overview: `SYSTEM_OVERVIEW.md`

---

**Ready to visualize!** 📊

Run the plotting script whenever you complete a training run to get instant visual feedback on performance, convergence, and resource utilization.

