#!/usr/bin/env python3
"""
Plot comprehensive training comparison for LLaMA 1.36B vs Qwen3 1.8B.
Shows loss, MFU, throughput, and training time for both models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def load_run(json_path):
    """Load training log JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_training_data(data):
    """Extract all training iteration data."""
    iterations = []
    losses = []
    mfus = []
    times_ms = []
    tokens_per_sec = []
    memory_peak = []
    
    for entry in data['training_iterations']:
        iter_num = entry['iter']
        if iter_num == 0:  # Skip first iteration (initialization)
            continue
        
        iterations.append(iter_num)
        losses.append(entry['loss'])
        times_ms.append(entry['time_ms'])
        
        # Extract MFU
        if isinstance(entry.get('mfu'), dict):
            mfus.append(entry['mfu']['mfu_percent'])
            tokens_per_sec.append(entry['mfu'].get('tokens_per_sec', 0))
        else:
            mfus.append(None)
            tokens_per_sec.append(None)
        
        # Extract memory
        if 'memory' in entry:
            memory_peak.append(entry['memory']['max_allocated_gb'])
        else:
            memory_peak.append(None)
    
    return {
        'iterations': np.array(iterations),
        'losses': np.array(losses),
        'mfus': np.array([m for m in mfus if m is not None]),
        'times_ms': np.array(times_ms),
        'tokens_per_sec': np.array([t for t in tokens_per_sec if t is not None and t > 0]),
        'memory_peak': np.array([m for m in memory_peak if m is not None]),
    }

def compute_training_time(data):
    """Compute total training time from start and end times."""
    try:
        start_str = data['start_time']
        end_str = data['end_time']
        
        # Parse ISO format timestamps
        start_time = datetime.fromisoformat(start_str)
        end_time = datetime.fromisoformat(end_str)
        
        total_seconds = (end_time - start_time).total_seconds()
        return total_seconds
    except:
        # Fallback: sum iteration times
        total_ms = sum([entry['time_ms'] for entry in data['training_iterations'] if entry['iter'] > 0])
        return total_ms / 1000

def main():
    # Load both runs
    llama_path = Path("../saves/run_20251115_145011.json")
    qwen_path = Path("../saves/run_20251115_150458.json")
    
    print(f"Loading LLaMA run: {llama_path}")
    llama_data = load_run(llama_path)
    print(f"Loading Qwen3 run: {qwen_path}")
    qwen_data = load_run(qwen_path)
    
    # Extract data
    llama_train = extract_training_data(llama_data)
    qwen_train = extract_training_data(qwen_data)
    
    # Get model info
    llama_config = llama_data.get('config', {})
    qwen_config = qwen_data.get('config', {})
    
    llama_params = llama_data['startup_info']['model']['total_params']
    qwen_params = qwen_data['startup_info']['model']['total_params']
    
    llama_time_sec = compute_training_time(llama_data)
    qwen_time_sec = compute_training_time(qwen_data)
    
    llama_time_min = llama_time_sec / 60
    qwen_time_min = qwen_time_sec / 60
    
    # Compute tokens processed
    def compute_tokens(data):
        config = data['config']
        block_size = config.get('block_size', 2048)
        batch_size = config.get('batch_size', 6)
        grad_accum = config.get('gradient_accumulation_steps', 32)
        num_gpus = data.get('metadata', {}).get('world_size', 2)
        max_iter = data['training_iterations'][-1]['iter']
        tokens_per_iter = block_size * batch_size * grad_accum * num_gpus
        return max_iter * tokens_per_iter / 1e9  # In billions
    
    llama_tokens = compute_tokens(llama_data)
    qwen_tokens = compute_tokens(qwen_data)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    
    fig.suptitle(f'Training Comparison: LLaMA 1.36B vs Qwen3 1.8B (100 Iterations)\n'
                 f'Hardware: 2× A100 80GB | Precision: bfloat16 | Parallelism: DDP+ZeRO-1',
                 fontsize=18, fontweight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Color scheme
    llama_color = '#2E86AB'  # Blue
    qwen_color = '#A23B72'   # Purple
    
    # ========== Plot 1: Loss Comparison ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(llama_train['iterations'], llama_train['losses'], 
             color=llama_color, linewidth=2.5, label='LLaMA 1.36B', marker='o', markersize=3)
    ax1.plot(qwen_train['iterations'], qwen_train['losses'], 
             color=qwen_color, linewidth=2.5, label='Qwen3 1.8B', marker='s', markersize=3)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add final loss annotations
    final_llama_loss = llama_train['losses'][-1]
    final_qwen_loss = qwen_train['losses'][-1]
    ax1.text(0.98, 0.05, f"LLaMA Final: {final_llama_loss:.3f}\nQwen3 Final: {final_qwen_loss:.3f}",
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ========== Plot 2: Perplexity Comparison ==========
    ax2 = fig.add_subplot(gs[0, 1])
    llama_perp = np.exp(llama_train['losses'])
    qwen_perp = np.exp(qwen_train['losses'])
    
    ax2.plot(llama_train['iterations'], llama_perp, 
             color=llama_color, linewidth=2.5, label='LLaMA 1.36B', marker='o', markersize=3)
    ax2.plot(qwen_train['iterations'], qwen_perp, 
             color=qwen_color, linewidth=2.5, label='Qwen3 1.8B', marker='s', markersize=3)
    
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Perplexity = exp(loss)', fontsize=12)
    ax2.set_title('Perplexity Curve', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # ========== Plot 3: Loss Reduction Rate ==========
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Compute loss reduction percentage from start
    llama_reduction = ((llama_train['losses'][0] - llama_train['losses']) / llama_train['losses'][0]) * 100
    qwen_reduction = ((qwen_train['losses'][0] - qwen_train['losses']) / qwen_train['losses'][0]) * 100
    
    ax3.plot(llama_train['iterations'], llama_reduction, 
             color=llama_color, linewidth=2.5, label='LLaMA 1.36B', marker='o', markersize=3)
    ax3.plot(qwen_train['iterations'], qwen_reduction, 
             color=qwen_color, linewidth=2.5, label='Qwen3 1.8B', marker='s', markersize=3)
    
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Loss Reduction (%)', fontsize=12)
    ax3.set_title('Loss Improvement from Start', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # ========== Plot 4: MFU Comparison ==========
    ax4 = fig.add_subplot(gs[1, 0])
    
    ax4.plot(llama_train['iterations'][:len(llama_train['mfus'])], llama_train['mfus'], 
             color=llama_color, linewidth=2.5, label='LLaMA 1.36B', marker='o', markersize=3)
    ax4.plot(qwen_train['iterations'][:len(qwen_train['mfus'])], qwen_train['mfus'], 
             color=qwen_color, linewidth=2.5, label='Qwen3 1.8B', marker='s', markersize=3)
    
    # Average lines
    llama_avg_mfu = np.mean(llama_train['mfus'])
    qwen_avg_mfu = np.mean(qwen_train['mfus'])
    
    ax4.axhline(y=llama_avg_mfu, color=llama_color, linestyle='--', alpha=0.5,
                label=f'LLaMA Avg: {llama_avg_mfu:.2f}%')
    ax4.axhline(y=qwen_avg_mfu, color=qwen_color, linestyle='--', alpha=0.5,
                label=f'Qwen3 Avg: {qwen_avg_mfu:.2f}%')
    
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('MFU (%)', fontsize=12)
    ax4.set_title(f'Model FLOPs Utilization\n(LLaMA: {llama_avg_mfu:.1f}% | Qwen3: {qwen_avg_mfu:.1f}%)', 
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9, loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])  # MFU ranges from 0-100%
    
    # ========== Plot 5: Throughput Comparison ==========
    ax5 = fig.add_subplot(gs[1, 1])
    
    ax5.plot(llama_train['iterations'][:len(llama_train['tokens_per_sec'])], llama_train['tokens_per_sec'], 
             color=llama_color, linewidth=2.5, label='LLaMA 1.36B', marker='o', markersize=3)
    ax5.plot(qwen_train['iterations'][:len(qwen_train['tokens_per_sec'])], qwen_train['tokens_per_sec'], 
             color=qwen_color, linewidth=2.5, label='Qwen3 1.8B', marker='s', markersize=3)
    
    # Average lines
    llama_avg_tok = np.mean(llama_train['tokens_per_sec'])
    qwen_avg_tok = np.mean(qwen_train['tokens_per_sec'])
    
    ax5.axhline(y=llama_avg_tok, color=llama_color, linestyle='--', alpha=0.5,
                label=f'LLaMA Avg: {llama_avg_tok:.0f} tok/s')
    ax5.axhline(y=qwen_avg_tok, color=qwen_color, linestyle='--', alpha=0.5,
                label=f'Qwen3 Avg: {qwen_avg_tok:.0f} tok/s')
    
    ax5.set_xlabel('Iteration', fontsize=12)
    ax5.set_ylabel('Tokens per Second', fontsize=12)
    ax5.set_title(f'Training Throughput\n(LLaMA: {llama_avg_tok:.0f} | Qwen3: {qwen_avg_tok:.0f} tok/s)', 
                  fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9, loc='lower right')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, None])  # Start from 0, auto-scale upper limit
    
    # ========== Plot 6: Time per Iteration ==========
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Convert to seconds
    llama_time_per_iter = llama_train['times_ms'] / 1000
    qwen_time_per_iter = qwen_train['times_ms'] / 1000
    
    ax6.plot(llama_train['iterations'], llama_time_per_iter, 
             color=llama_color, linewidth=2.5, label='LLaMA 1.36B', marker='o', markersize=3)
    ax6.plot(qwen_train['iterations'], qwen_time_per_iter, 
             color=qwen_color, linewidth=2.5, label='Qwen3 1.8B', marker='s', markersize=3)
    
    # Average lines
    llama_avg_time = np.mean(llama_time_per_iter)
    qwen_avg_time = np.mean(qwen_time_per_iter)
    
    ax6.axhline(y=llama_avg_time, color=llama_color, linestyle='--', alpha=0.5,
                label=f'LLaMA Avg: {llama_avg_time:.2f}s')
    ax6.axhline(y=qwen_avg_time, color=qwen_color, linestyle='--', alpha=0.5,
                label=f'Qwen3 Avg: {qwen_avg_time:.2f}s')
    
    ax6.set_xlabel('Iteration', fontsize=12)
    ax6.set_ylabel('Time per Iteration (seconds)', fontsize=12)
    ax6.set_title(f'Iteration Time\n(LLaMA: {llama_avg_time:.2f}s | Qwen3: {qwen_avg_time:.2f}s)', 
                  fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9, loc='upper right')
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, None])  # Start from 0, auto-scale upper limit
    
    # ========== Plot 7: Memory Usage ==========
    ax7 = fig.add_subplot(gs[2, 0])
    
    ax7.plot(llama_train['iterations'][:len(llama_train['memory_peak'])], llama_train['memory_peak'], 
             color=llama_color, linewidth=2.5, label='LLaMA 1.36B', marker='o', markersize=3)
    ax7.plot(qwen_train['iterations'][:len(qwen_train['memory_peak'])], qwen_train['memory_peak'], 
             color=qwen_color, linewidth=2.5, label='Qwen3 1.8B', marker='s', markersize=3)
    
    # GPU capacity line
    gpu_capacity = 80  # A100 80GB
    ax7.axhline(y=gpu_capacity, color='red', linestyle='--', alpha=0.4, 
                label=f'GPU Capacity: {gpu_capacity} GB')
    
    ax7.set_xlabel('Iteration', fontsize=12)
    ax7.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax7.set_title(f'GPU Memory Usage\n(LLaMA: {llama_train["memory_peak"][0]:.1f} GB | Qwen3: {qwen_train["memory_peak"][0]:.1f} GB)', 
                  fontsize=13, fontweight='bold')
    ax7.legend(fontsize=10, loc='lower right')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, gpu_capacity * 1.05])
    
    # ========== Plot 8: Training Time Breakdown ==========
    ax8 = fig.add_subplot(gs[2, 1])
    
    models = ['LLaMA\n1.36B', 'Qwen3\n1.8B']
    times_minutes = [llama_time_min, qwen_time_min]
    colors = [llama_color, qwen_color]
    
    bars = ax8.bar(models, times_minutes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, time_min) in enumerate(zip(bars, times_minutes)):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_min:.1f} min\n({time_min/60:.2f} hrs)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax8.set_ylabel('Total Training Time (minutes)', fontsize=12)
    ax8.set_ylim([0, 20])  # Extended range for better visibility
    ax8.set_title(f'Total Training Time (100 Iterations)\nQwen3 is {qwen_time_min/llama_time_min:.2f}× slower', 
                  fontsize=13, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # ========== Plot 9: Model Efficiency Metrics ==========
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create comparison table
    table_data = [
        ['Metric', 'LLaMA 1.36B', 'Qwen3 1.8B', 'Ratio'],
        ['─' * 15, '─' * 12, '─' * 12, '─' * 8],
        ['Parameters', f'{llama_params/1e9:.2f}B', f'{qwen_params/1e9:.2f}B', f'{qwen_params/llama_params:.2f}×'],
        ['Final Loss', f'{final_llama_loss:.3f}', f'{final_qwen_loss:.3f}', f'{final_qwen_loss/final_llama_loss:.3f}×'],
        ['Avg MFU', f'{llama_avg_mfu:.2f}%', f'{qwen_avg_mfu:.2f}%', f'{qwen_avg_mfu/llama_avg_mfu:.3f}×'],
        ['Throughput', f'{llama_avg_tok:.0f} tok/s', f'{qwen_avg_tok:.0f} tok/s', f'{qwen_avg_tok/llama_avg_tok:.3f}×'],
        ['Time/Iter', f'{llama_avg_time:.2f}s', f'{qwen_avg_time:.2f}s', f'{qwen_avg_time/llama_avg_time:.2f}×'],
        ['Total Time', f'{llama_time_min:.1f} min', f'{qwen_time_min:.1f} min', f'{qwen_time_min/llama_time_min:.2f}×'],
        ['Peak Memory', f'{llama_train["memory_peak"][0]:.1f} GB', f'{qwen_train["memory_peak"][0]:.1f} GB', f'{qwen_train["memory_peak"][0]/llama_train["memory_peak"][0]:.2f}×'],
        ['Tokens', f'{llama_tokens:.2f}B', f'{qwen_tokens:.2f}B', f'{qwen_tokens/llama_tokens:.2f}×'],
    ]
    
    # Render table
    table = ax9.table(cellText=table_data, cellLoc='left', loc='center',
                      colWidths=[0.35, 0.25, 0.25, 0.15],
                      bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#4A90E2')
        cell.set_text_props(weight='bold', color='white')
    
    # Style separator row
    for i in range(4):
        cell = table[(1, i)]
        cell.set_facecolor('#E8E8E8')
    
    # Alternate row colors
    for i in range(2, len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#F5F5F5')
    
    ax9.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
    
    # Save figure
    output_path = Path("../saves/LLaMA_vs_Qwen3_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Comparison plot saved: {output_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n🔵 LLaMA 1.36B:")
    print(f"   Parameters:      {llama_params/1e9:.3f}B ({llama_params/1e6:.0f}M)")
    print(f"   Final Loss:      {final_llama_loss:.4f}")
    print(f"   Avg MFU:         {llama_avg_mfu:.2f}%")
    print(f"   Throughput:      {llama_avg_tok:.0f} tokens/s")
    print(f"   Time/Iter:       {llama_avg_time:.2f}s")
    print(f"   Total Time:      {llama_time_min:.1f} min ({llama_time_min/60:.2f} hrs)")
    print(f"   Peak Memory:     {llama_train['memory_peak'][0]:.1f} GB / 80 GB")
    print(f"   Tokens:          {llama_tokens:.2f}B")
    
    print(f"\n🟣 Qwen3 1.8B:")
    print(f"   Parameters:      {qwen_params/1e9:.3f}B ({qwen_params/1e6:.0f}M)")
    print(f"   Final Loss:      {final_qwen_loss:.4f}")
    print(f"   Avg MFU:         {qwen_avg_mfu:.2f}%")
    print(f"   Throughput:      {qwen_avg_tok:.0f} tokens/s")
    print(f"   Time/Iter:       {qwen_avg_time:.2f}s")
    print(f"   Total Time:      {qwen_time_min:.1f} min ({qwen_time_min/60:.2f} hrs)")
    print(f"   Peak Memory:     {qwen_train['memory_peak'][0]:.1f} GB / 80 GB")
    print(f"   Tokens:          {qwen_tokens:.2f}B")
    
    print(f"\n📊 Key Insights:")
    print(f"   • Qwen3 has {qwen_params/llama_params:.2f}× more parameters")
    print(f"   • Qwen3 achieves {((final_llama_loss - final_qwen_loss) / final_llama_loss * 100):.1f}% better final loss")
    print(f"   • Qwen3 has {qwen_avg_mfu/llama_avg_mfu:.2f}× higher MFU")
    print(f"   • Qwen3 is {qwen_avg_time/llama_avg_time:.2f}× slower per iteration")
    print(f"   • Qwen3 uses {qwen_train['memory_peak'][0]/llama_train['memory_peak'][0]:.2f}× more memory")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()

