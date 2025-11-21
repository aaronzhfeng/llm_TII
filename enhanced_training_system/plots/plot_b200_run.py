#!/usr/bin/env python3
"""
Plot comprehensive B200 training analysis for a single run.
Shows loss, perplexity, MFU, throughput, learning rate, and memory usage.
Optimized for 8× B200 GPU analysis with 192GB memory per GPU.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

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
    achieved_tflops = []
    memory_alloc = []
    memory_peak = []
    memory_reserved = []
    
    for entry in data['training_iterations']:
        iter_num = entry['iter']
        if iter_num == 0:  # Skip first iteration (initialization)
            continue
        
        # Check if this is an evaluation run (time > 100s indicates eval)
        is_eval_iter = entry['time_ms'] > 100000
            
        iterations.append(iter_num)
        losses.append(entry['loss'])
        
        # Only add time-based metrics if not evaluation
        if not is_eval_iter:
            times_ms.append(entry['time_ms'])
        else:
            times_ms.append(None)  # Mark as None to skip in plots
        
        # Extract MFU
        if isinstance(entry.get('mfu'), dict):
            # Skip MFU from eval iters (it's artificially low)
            if not is_eval_iter:
                mfus.append(entry['mfu']['mfu_percent'])
                tokens_per_sec.append(entry['mfu'].get('tokens_per_sec', 0))
                achieved_tflops.append(entry['mfu'].get('achieved_tflops', 0))
            else:
                mfus.append(None)
                tokens_per_sec.append(None)
                achieved_tflops.append(None)
        else:
            mfus.append(None)
            tokens_per_sec.append(None)
            achieved_tflops.append(None)
        
        # Extract memory
        if 'memory' in entry:
            memory_alloc.append(entry['memory']['allocated_gb'])
            memory_peak.append(entry['memory']['max_allocated_gb'])
            memory_reserved.append(entry['memory'].get('reserved_gb', 0))
        else:
            memory_alloc.append(None)
            memory_peak.append(None)
            memory_reserved.append(None)
    
    return {
        'iterations': iterations,
        'losses': losses,
        'mfus': mfus,
        'times_ms': times_ms,
        'tokens_per_sec': tokens_per_sec,
        'achieved_tflops': achieved_tflops,
        'memory_alloc': memory_alloc,
        'memory_peak': memory_peak,
        'memory_reserved': memory_reserved,
    }

def extract_eval_data(data):
    """Extract evaluation data."""
    if not data.get('eval_steps'):
        return None
    
    eval_iters = []
    train_losses = []
    val_losses = []
    lrs = []
    
    for entry in data['eval_steps']:
        eval_iters.append(entry['iter'])
        train_losses.append(entry['train_loss'])
        val_losses.append(entry['val_loss'])
        if 'lr' in entry:
            lrs.append(entry['lr'])
    
    return {
        'iters': eval_iters,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'lrs': lrs
    }

def compute_learning_rate(iter_num, config):
    """Compute learning rate at given iteration."""
    lr = config.get('learning_rate', 3e-4)
    warmup = config.get('warmup_iters', 2000)
    decay_iters = config.get('lr_decay_iters', 25000)
    min_lr = config.get('min_lr', 3e-5)
    decay_lr = config.get('decay_lr', True)
    
    if not decay_lr:
        return lr
    
    # Warmup
    if iter_num < warmup:
        return lr * (iter_num + 1) / (warmup + 1)
    
    # Decay
    if iter_num > decay_iters:
        return min_lr
    
    # Cosine decay
    decay_ratio = (iter_num - warmup) / (decay_iters - warmup)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

def main():
    # Determine input file
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        # Default to most recent run in out-llama-1.36b
        out_dir = Path(__file__).parent.parent / "out-llama-1.36b"
        json_files = sorted(out_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            print(f"❌ No JSON files found in {out_dir}")
            print(f"Usage: {sys.argv[0]} [path/to/run.json]")
            sys.exit(1)
        json_path = json_files[0]
    
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        print(f"Usage: {sys.argv[0]} [path/to/run.json]")
        sys.exit(1)
    
    print(f"Loading: {json_path}")
    data = load_run(json_path)
    
    # Extract data
    train_data = extract_training_data(data)
    eval_data = extract_eval_data(data)
    config = data.get('config', {})
    startup = data.get('startup_info', {})
    
    # Get model info
    model_info = startup.get('model', {})
    total_params = model_info.get('total_params', 0)
    
    # Get hardware info
    hardware = startup.get('hardware', {})
    num_gpus = hardware.get('num_gpus', data.get('metadata', {}).get('world_size', 8))
    gpu_name = hardware.get('gpu_name', 'B200')
    gpu_memory_gb = hardware.get('gpu_memory_gb', 192)
    
    # Compute derived metrics
    iters = np.array(train_data['iterations'])
    losses = np.array(train_data['losses'])
    perplexities = np.exp(losses)
    
    # Compute learning rates for all iterations
    lrs = [compute_learning_rate(i, config) for i in iters]
    
    # Compute cumulative tokens (accounting for DDP division in config)
    block_size = config.get('block_size', 2048)
    batch_size = config.get('batch_size', 8)
    grad_accum = config.get('gradient_accumulation_steps', 128)
    # Note: grad_accum in config is total (will be divided by num_gpus in train.py)
    # So actual grad_accum_per_gpu = grad_accum / num_gpus
    grad_accum_per_gpu = grad_accum // num_gpus
    tokens_per_iter = block_size * batch_size * grad_accum_per_gpu * num_gpus
    cumulative_tokens = iters * tokens_per_iter / 1e9  # In billions
    
    # Filter out None values for MFU and memory
    mfu_iters = [iters[i] for i, m in enumerate(train_data['mfus']) if m is not None]
    mfu_values = [m for m in train_data['mfus'] if m is not None]
    
    tokens_iters = [iters[i] for i, t in enumerate(train_data['tokens_per_sec']) if t is not None and t > 0]
    tokens_values = [t for t in train_data['tokens_per_sec'] if t is not None and t > 0]
    
    tflops_iters = [iters[i] for i, t in enumerate(train_data['achieved_tflops']) if t is not None and t > 0]
    tflops_values = [t for t in train_data['achieved_tflops'] if t is not None and t > 0]
    
    mem_iters = [iters[i] for i, m in enumerate(train_data['memory_alloc']) if m is not None]
    mem_alloc_values = [m for m in train_data['memory_alloc'] if m is not None]
    mem_peak_values = [train_data['memory_peak'][i] for i, m in enumerate(train_data['memory_alloc']) if m is not None]
    mem_reserved_values = [train_data['memory_reserved'][i] for i, m in enumerate(train_data['memory_reserved']) if m is not None]
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Model name from config
    model_name = f"LLaMA {total_params/1e9:.2f}B"
    attention_backend = config.get('attention_backend', 'unknown')
    compile_status = "Compiled" if config.get('compile', False) else "No Compile"
    dataloader_status = "DataLoader" if config.get('use_dataloader', False) else "No DataLoader"
    
    fig.suptitle(f'{model_name} Training on {num_gpus}× {gpu_name} - {len(iters)} Iterations\n'
                 f'{total_params/1e6:.0f}M params | {attention_backend} | {compile_status} | {dataloader_status} | '
                 f'Batch={batch_size}, GradAccum={grad_accum_per_gpu}/GPU | Total Tokens: {cumulative_tokens[-1]:.2f}B',
                 fontsize=14, fontweight='bold')
    
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35, top=0.92, bottom=0.05, left=0.05, right=0.97)
    
    # ========== Plot 1: Loss & Perplexity (Dual Y-axis) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    
    # Loss on left axis
    line1 = ax1.plot(iters, losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=10)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=10, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Mark eval point
    if eval_data and eval_data['iters']:
        ax1.scatter(eval_data['iters'], eval_data['val_loss'], 
                   color='red', s=100, zorder=5, marker='*', 
                   label=f"Val Loss: {eval_data['val_loss'][0]:.3f}")
    
    # Perplexity on right axis
    line2 = ax1_twin.plot(iters, perplexities, 'g-', linewidth=1.5, 
                          label='Perplexity', alpha=0.6, linestyle='--')
    ax1_twin.set_ylabel('Perplexity = exp(loss)', fontsize=10, color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    ax1_twin.set_yscale('log')
    
    ax1.set_title('Loss & Perplexity', fontsize=11, fontweight='bold')
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    if eval_data and eval_data['iters']:
        labels.insert(1, f"Val @ {eval_data['iters'][0]}")
    ax1.legend(lines, labels, loc='upper right', fontsize=8)
    
    # ========== Plot 2: Learning Rate Schedule ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iters, lrs, 'purple', linewidth=2)
    
    # Mark warmup end
    warmup = config.get('warmup_iters', 2000)
    if max(iters) > warmup:
        ax2.axvline(x=warmup, color='red', linestyle='--', alpha=0.5, label=f'Warmup End: {warmup}')
        ax2.legend(fontsize=8)
    
    ax2.set_xlabel('Iteration', fontsize=10)
    ax2.set_ylabel('Learning Rate', fontsize=10)
    ax2.set_title('Learning Rate Schedule', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # ========== Plot 3: Cumulative Tokens ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iters, cumulative_tokens, 'teal', linewidth=2, marker='o', markersize=2)
    
    ax3.set_xlabel('Iteration', fontsize=10)
    ax3.set_ylabel('Cumulative Tokens (Billions)', fontsize=10)
    ax3.set_title(f'Tokens Processed: {cumulative_tokens[-1]:.3f}B', 
                  fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ========== Plot 4: MFU Over Time ==========
    ax4 = fig.add_subplot(gs[1, 0])
    if mfu_values:
        ax4.plot(mfu_iters, mfu_values, 'orange', linewidth=2, marker='o', markersize=3)
        
        # Average MFU
        avg_mfu = np.mean(mfu_values)
        ax4.axhline(y=avg_mfu, color='blue', linestyle='--', alpha=0.6, 
                   label=f'Average: {avg_mfu:.2f}%')
        
        # B200 target (optimistic)
        target_mfu = 40  # Target for well-optimized B200
        ax4.axhline(y=target_mfu, color='green', linestyle=':', alpha=0.4,
                   label=f'Target: {target_mfu}%')
        
        ax4.set_xlabel('Iteration', fontsize=10)
        ax4.set_ylabel('MFU (%)', fontsize=10)
        ax4.set_title(f'Model FLOPs Utilization (Avg: {avg_mfu:.2f}%)', 
                     fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        mfu_min = min(mfu_values)
        mfu_max = max(mfu_values)
        mfu_range = mfu_max - mfu_min
        margin = max(5, mfu_range * 0.3)
        ax4.set_ylim([max(0, mfu_min - margin), min(100, max(mfu_max + margin, target_mfu + 10))])
    
    # ========== Plot 5: Throughput (Tokens/sec) ==========
    ax5 = fig.add_subplot(gs[1, 1])
    if tokens_values:
        ax5.plot(tokens_iters, [t/1000 for t in tokens_values], 'brown', linewidth=2, marker='o', markersize=3)
        
        # Average throughput
        avg_tokens = np.mean(tokens_values)
        ax5.axhline(y=avg_tokens/1000, color='blue', linestyle='--', alpha=0.6,
                   label=f'Average: {avg_tokens/1000:.0f}K tokens/s')
        
        ax5.set_xlabel('Iteration', fontsize=10)
        ax5.set_ylabel('Tokens per Second (Thousands)', fontsize=10)
        ax5.set_title(f'Training Throughput (Avg: {avg_tokens/1000:.1f}K tokens/s)', 
                     fontsize=11, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
    
    # ========== Plot 6: Achieved TFLOPS ==========
    ax6 = fig.add_subplot(gs[1, 2])
    if tflops_values:
        ax6.plot(tflops_iters, tflops_values, 'purple', linewidth=2, marker='o', markersize=3)
        
        # Average TFLOPS
        avg_tflops = np.mean(tflops_values)
        ax6.axhline(y=avg_tflops, color='blue', linestyle='--', alpha=0.6,
                   label=f'Average: {avg_tflops:.0f} TFLOPS')
        
        # B200 peak (bf16 tensor cores)
        peak_tflops = 36000  # 36 PFLOPS for 8× B200
        ax6.axhline(y=peak_tflops, color='red', linestyle=':', alpha=0.4,
                   label=f'Peak: {peak_tflops/1000:.0f}K TFLOPS')
        
        ax6.set_xlabel('Iteration', fontsize=10)
        ax6.set_ylabel('Achieved TFLOPS', fontsize=10)
        ax6.set_title(f'Compute Performance (Avg: {avg_tflops:.0f} TFLOPS)', 
                     fontsize=11, fontweight='bold')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
    
    # ========== Plot 7: Memory Usage ==========
    ax7 = fig.add_subplot(gs[2, 0])
    if mem_alloc_values:
        ax7.plot(mem_iters, mem_alloc_values, 'steelblue', linewidth=2, 
                label='Allocated', marker='o', markersize=2)
        ax7.plot(mem_iters, mem_peak_values, 'darkred', linewidth=2, 
                label='Peak', marker='s', markersize=2)
        ax7.plot(mem_iters, mem_reserved_values, 'orange', linewidth=1.5, 
                label='Reserved', marker='d', markersize=2, alpha=0.6)
        
        # GPU capacity
        ax7.axhline(y=gpu_memory_gb, color='red', linestyle='--', alpha=0.4, 
                   label=f'GPU Capacity: {gpu_memory_gb:.0f} GB')
        
        ax7.set_xlabel('Iteration', fontsize=10)
        ax7.set_ylabel('Memory (GB)', fontsize=10)
        ax7.set_title(f'GPU Memory Usage (Peak: {max(mem_peak_values):.1f} / {gpu_memory_gb:.0f} GB = {max(mem_peak_values)/gpu_memory_gb*100:.1f}%)', 
                     fontsize=11, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim([0, gpu_memory_gb * 1.05])
    
    # ========== Plot 8: Iteration Time ==========
    ax8 = fig.add_subplot(gs[2, 1])
    valid_times = [(iters[i], t/1000) for i, t in enumerate(train_data['times_ms']) if t is not None]
    if valid_times:
        time_iters, time_values = zip(*valid_times)
        ax8.plot(time_iters, time_values, 'darkgreen', linewidth=2, marker='o', markersize=3)
        
        avg_time = np.mean(time_values)
        ax8.axhline(y=avg_time, color='blue', linestyle='--', alpha=0.6,
                   label=f'Average: {avg_time:.2f}s')
        
        ax8.set_xlabel('Iteration', fontsize=10)
        ax8.set_ylabel('Time (seconds)', fontsize=10)
        ax8.set_title(f'Time per Iteration (Avg: {avg_time:.2f}s)', 
                     fontsize=11, fontweight='bold')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
    
    # ========== Plot 9: Summary Text Box ==========
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create summary text
    summary_lines = []
    summary_lines.append(f"🏗️  MODEL SUMMARY")
    summary_lines.append(f"  • Params: {total_params/1e9:.3f}B ({total_params/1e6:.0f}M)")
    summary_lines.append(f"  • Layers: {config.get('n_layer', '?')} × {config.get('n_embd', '?')}D")
    summary_lines.append(f"  • Heads: {config.get('n_head', '?')}")
    summary_lines.append(f"  • Context: {config.get('block_size', '?')} tokens")
    summary_lines.append(f"")
    summary_lines.append(f"🔧 OPTIMIZATION")
    summary_lines.append(f"  • Attention: {config.get('attention_backend', '?')}")
    summary_lines.append(f"  • Compile: {config.get('compile', False)}")
    summary_lines.append(f"  • CUDA Graphs: {config.get('use_cuda_graphs', False)}")
    summary_lines.append(f"  • DataLoader: {config.get('use_dataloader', False)}")
    summary_lines.append(f"  • Precision: {config.get('dtype', '?')}")
    summary_lines.append(f"")
    summary_lines.append(f"⚡ PERFORMANCE")
    if mfu_values:
        summary_lines.append(f"  • MFU: {np.mean(mfu_values):.2f}%")
    if tokens_values:
        summary_lines.append(f"  • Throughput: {np.mean(tokens_values)/1000:.1f}K tok/s")
    if tflops_values:
        summary_lines.append(f"  • TFLOPS: {np.mean(tflops_values):.0f} / 36000")
    if valid_times:
        summary_lines.append(f"  • Time/iter: {np.mean([t for _, t in valid_times]):.2f}s")
    summary_lines.append(f"")
    summary_lines.append(f"💾 MEMORY (per GPU)")
    if mem_peak_values:
        summary_lines.append(f"  • Peak: {max(mem_peak_values):.1f} GB / {gpu_memory_gb:.0f} GB")
        summary_lines.append(f"  • Utilization: {max(mem_peak_values)/gpu_memory_gb*100:.1f}%")
    summary_lines.append(f"")
    summary_lines.append(f"📊 TRAINING")
    summary_lines.append(f"  • Iterations: {len(iters)}")
    summary_lines.append(f"  • Tokens: {cumulative_tokens[-1]:.2f}B")
    summary_lines.append(f"  • Loss: {losses[0]:.3f} → {losses[-1]:.3f}")
    summary_lines.append(f"  • PPL: {perplexities[0]:.1f} → {perplexities[-1]:.1f}")
    
    summary_text = '\n'.join(summary_lines)
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
            verticalalignment='top', horizontalalignment='left',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Save figure
    output_path = json_path.parent / f"{json_path.stem}_b200_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Plot saved: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("B200 TRAINING ANALYSIS SUMMARY")
    print("="*80)
    
    # Model info
    print(f"\n🏗️  MODEL:")
    print(f"   Total params:       {total_params/1e9:.3f}B ({total_params/1e6:.0f}M)")
    print(f"   Architecture:       {config.get('n_layer', '?')}L-{config.get('n_head', '?')}H-{config.get('n_embd', '?')}D")
    print(f"   Hardware:           {num_gpus}× {gpu_name} ({gpu_memory_gb:.0f} GB/GPU)")
    
    # Optimization settings
    print(f"\n🔧 OPTIMIZATION:")
    print(f"   Attention:          {config.get('attention_backend', '?')}")
    print(f"   torch.compile():    {config.get('compile', False)}")
    print(f"   CUDA Graphs:        {config.get('use_cuda_graphs', False)}")
    print(f"   DataLoader:         {config.get('use_dataloader', False)} (workers={config.get('dataloader_num_workers', 0)})")
    print(f"   Precision:          {config.get('dtype', '?')}")
    print(f"   Batch size:         {batch_size} (per GPU)")
    print(f"   Grad accum:         {grad_accum_per_gpu} (per GPU)")
    print(f"   Effective batch:    {tokens_per_iter:,} tokens/iter")
    
    # Training progress
    print(f"\n📈 TRAINING PROGRESS:")
    print(f"   Iterations:         {len(iters)} ({iters[0]} → {iters[-1]})")
    print(f"   Tokens processed:   {cumulative_tokens[-1]:.2f}B")
    
    # Loss metrics
    print(f"\n📉 LOSS & PERPLEXITY:")
    print(f"   Initial:            {losses[0]:.4f} (PPL: {perplexities[0]:.1f})")
    print(f"   Final:              {losses[-1]:.4f} (PPL: {perplexities[-1]:.1f})")
    print(f"   Reduction:          {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
    
    # Performance
    if mfu_values:
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Avg MFU:            {np.mean(mfu_values):.2f}% (target: 35-50%)")
        print(f"   Min/Max MFU:        {min(mfu_values):.2f}% / {max(mfu_values):.2f}%")
    if tokens_values:
        print(f"   Avg Throughput:     {np.mean(tokens_values):,.0f} tokens/s ({np.mean(tokens_values)/1000:.1f}K)")
    if tflops_values:
        print(f"   Avg TFLOPS:         {np.mean(tflops_values):,.0f} / 36,000 ({np.mean(tflops_values)/36000*100:.2f}%)")
    if valid_times:
        print(f"   Avg Time/Iter:      {np.mean([t for _, t in valid_times]):.2f}s")
    
    # Memory
    if mem_peak_values:
        print(f"\n💾 MEMORY (per GPU):")
        print(f"   Peak allocated:     {max(mem_peak_values):.2f} GB / {gpu_memory_gb:.0f} GB ({max(mem_peak_values)/gpu_memory_gb*100:.1f}%)")
        print(f"   Avg allocated:      {np.mean(mem_alloc_values):.2f} GB")
        if mem_reserved_values:
            print(f"   Reserved (cached):  {max(mem_reserved_values):.2f} GB")
    
    # Training time
    if valid_times:
        total_time_sec = sum([t for _, t in valid_times])
        print(f"\n⏱️  TIME:")
        print(f"   Total training:     {total_time_sec/60:.1f} minutes ({total_time_sec/3600:.2f} hours)")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()

