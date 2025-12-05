#!/usr/bin/env python3
"""
Plot comparison between Base, SFT, and DPO models on benchmark evaluations.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
SCRIPTS_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPTS_DIR.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Results Data (Dec 2024)
# ══════════════════════════════════════════════════════════════════════════════

benchmarks = ['OpenBookQA', 'ARC-Challenge', 'ARC-Easy']

# Base model (160k iterations, ~26B tokens)
base_model = [26.40, 27.22, 52.57]

# SFT model (2800 iterations on Alpaca)
sft_model = [29.00, 29.52, 52.61]

# DPO model (800 iterations on UltraFeedback)
dpo_model = [27.20, 29.35, 54.92]

# Random baseline (4-choice = 25%)
random_baseline = [25.0, 25.0, 25.0]

# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════

x = np.arange(len(benchmarks))
width = 0.2

fig, ax = plt.subplots(figsize=(14, 9))

# Create bars
bars1 = ax.bar(x - 1.5*width, base_model, width, label='Base (Pre-trained)', 
               color='#3498DB', edgecolor='#2C3E50', linewidth=1.5)
bars2 = ax.bar(x - 0.5*width, sft_model, width, label='+ SFT (Alpaca)', 
               color='#2ECC71', edgecolor='#2C3E50', linewidth=1.5)
bars3 = ax.bar(x + 0.5*width, dpo_model, width, label='+ DPO (UltraFeedback)', 
               color='#E74C3C', edgecolor='#2C3E50', linewidth=1.5)
bars4 = ax.bar(x + 1.5*width, random_baseline, width, label='Random (25%)', 
               color='#95A5A6', edgecolor='#2C3E50', linewidth=1.5, alpha=0.5)

# Customize
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Benchmark', fontsize=14, fontweight='bold')
ax.set_title('Qwen3-1.8B: Post-Training Progression\n(Base → SFT → DPO)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.set_ylim(0, 70)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
def add_labels(bars, offset=0):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 + offset),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Add improvement annotations
def add_delta(x_pos, base_val, new_val, y_offset=0):
    delta = new_val - base_val
    sign = '+' if delta >= 0 else ''
    color = '#27AE60' if delta >= 0 else '#E74C3C'
    ax.annotate(f'{sign}{delta:.1f}%', 
                xy=(x_pos, max(base_val, new_val) + 5 + y_offset),
                ha='center', fontsize=8, color=color, fontweight='bold')

# Training info annotation
training_info = """
Training Pipeline:
• Base: 160k iter (~26B tokens)
• SFT: +2800 iter (Alpaca, 52k samples)
• DPO: +800 iter (UltraFeedback, 60k pairs)
"""
ax.text(0.98, 0.98, training_info, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'post_training_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(PLOTS_DIR / 'post_training_comparison.pdf', bbox_inches='tight')
print(f"✅ Saved to {PLOTS_DIR}/post_training_comparison.{{png,pdf}}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary Table
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("POST-TRAINING BENCHMARK COMPARISON")
print("="*80)
print(f"{'Benchmark':<18} {'Base':<12} {'SFT':<12} {'DPO':<12} {'SFT Δ':<10} {'DPO Δ':<10}")
print("-"*80)
for i, bench in enumerate(benchmarks):
    sft_delta = sft_model[i] - base_model[i]
    dpo_delta = dpo_model[i] - base_model[i]
    sft_sign = '+' if sft_delta >= 0 else ''
    dpo_sign = '+' if dpo_delta >= 0 else ''
    print(f"{bench:<18} {base_model[i]:.2f}%{'':<6} {sft_model[i]:.2f}%{'':<6} {dpo_model[i]:.2f}%{'':<6} {sft_sign}{sft_delta:.2f}%{'':<4} {dpo_sign}{dpo_delta:.2f}%")
print("-"*80)
avg_base = np.mean(base_model)
avg_sft = np.mean(sft_model)
avg_dpo = np.mean(dpo_model)
print(f"{'Average':<18} {avg_base:.2f}%{'':<6} {avg_sft:.2f}%{'':<6} {avg_dpo:.2f}%{'':<6} +{avg_sft-avg_base:.2f}%{'':<4} +{avg_dpo-avg_base:.2f}%")
print("="*80)

# Key findings
print("\n📊 Key Findings:")
print(f"  • SFT improves OpenBookQA by +{sft_model[0]-base_model[0]:.1f}% and ARC-Challenge by +{sft_model[1]-base_model[1]:.1f}%")
print(f"  • DPO improves ARC-Easy by +{dpo_model[2]-base_model[2]:.1f}% (best overall on this benchmark)")
print(f"  • Average improvement: SFT +{avg_sft-avg_base:.1f}%, DPO +{avg_dpo-avg_base:.1f}%")
