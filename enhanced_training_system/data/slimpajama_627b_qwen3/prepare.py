"""
Tokenize SlimPajama-627B dataset with Qwen3 tokenizer.

PREREQUISITES: Dataset must be downloaded first using download.py

Usage:
    python3 prepare.py                # Tokenize all splits
    python3 prepare.py --train-only   # Tokenize training set only

Output:
    - train.bin: ~1.2TB (627B tokens as uint32)
    - val.bin: ~1GB (500M tokens) [if validation available]
    - test.bin: ~1GB (500M tokens) [if test available]
    - meta.pkl: Metadata

Time: 2-4 hours with 224 CPU cores
"""

import os
import sys
import pickle
import argparse
import numpy as np
import traceback
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Use /raid/ cache (set by download.py)
os.environ['HF_HOME'] = '/raid/zhf004/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/raid/zhf004/huggingface_cache/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/raid/zhf004/huggingface_cache/transformers'

# Suppress verbose logging
import logging
logging.getLogger('datasets').setLevel(logging.ERROR)

# Auto-detect CPU count
NUM_PROC = os.cpu_count()
print(f"Detected {NUM_PROC} CPU cores")

def parse_args():
    parser = argparse.ArgumentParser(description='Tokenize SlimPajama-627B')
    parser.add_argument('--train-only', action='store_true',
                        help='Tokenize training set only (skip validation/test)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-tokenization even if .bin files exist')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print("="*80)
    print("SlimPajama-627B Tokenization (Qwen3)")
    if args.train_only:
        print("Mode: TRAINING SET ONLY")
    print("="*80)
    
    # Load Qwen3 tokenizer
    print("\n[1/4] Loading Qwen3 tokenizer...")
    local_tokenizer_path = "../../qwen3_tokenizer"
    
    try:
        if os.path.exists(local_tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, 
                                                     trust_remote_code=True, use_fast=True)
            print(f"✓ Loaded from {local_tokenizer_path}")
        else:
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', 
                                                     trust_remote_code=True, use_fast=True)
            print(f"✓ Loaded from HuggingFace")
        print(f"  Vocab size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Load dataset from cache
    print("\n[2/4] Loading dataset from cache...")
    print(f"   Cache: /raid/zhf004/huggingface_cache/datasets")
    print(f"   Using {NUM_PROC} CPU cores for tokenization")
    
    # Check if HF_TOKEN is set (helps with rate limits)
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print(f"   ⚠️  HF_TOKEN not set - you may hit rate limits")
        print(f"   Set it with: export HF_TOKEN=<your_token>")
    else:
        print(f"   ✓ HF_TOKEN detected")
    
    try:
        # Load from cache - will use cached Arrow files if conversion is complete
        dataset = load_dataset("cerebras/SlimPajama-627B", token=hf_token)
        print(f"✓ Dataset loaded: {list(dataset.keys())}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\n📋 Full error traceback:")
        traceback.print_exc()
        print("\n⚠️  Dataset not found in cache or conversion failed!")
        print("   Run download.py first to download the dataset:")
        print("   python3 download.py")
        sys.exit(1)
    
    # Tokenization function
    def process_batch(examples):
        """Tokenize a batch of examples"""
        all_ids = []
        all_lens = []
        
        for text in examples['text']:
            ids = tokenizer(text, truncation=False, add_special_tokens=False)['input_ids']
            ids.append(tokenizer.eos_token_id)
            all_ids.append(ids)
            all_lens.append(len(ids))
        
        return {'ids': all_ids, 'len': all_lens}
    
    # Tokenize and write each split
    print(f"\n[3/4] Tokenizing with {NUM_PROC} cores...")
    total_tokens = {}
    
    # Determine which splits to process
    if args.train_only:
        splits_to_process = ['train']
        print("  Processing TRAIN split only")
    else:
        splits_to_process = ['train', 'validation', 'test']
        print("  Processing all splits (train, validation, test)")
    
    for split_name in splits_to_process:
        if split_name not in dataset:
            print(f"\n  ⚠️  Skipping {split_name} (not found in cache)")
            continue
        
        # Check if already tokenized (skip unless --force)
        output_name = "val" if split_name == "validation" else split_name
        output_file = os.path.join(os.path.dirname(__file__), f'{output_name}.bin')
        
        if os.path.exists(output_file) and not args.force:
            # Get file size for display
            file_size_gb = os.path.getsize(output_file) / 1e9
            print(f"\n  ✓ {split_name}: Already tokenized ({output_name}.bin exists, {file_size_gb:.2f} GB)")
            print(f"    Use --force to re-tokenize")
            
            # Still need to count tokens for meta.pkl
            # Estimate based on file size (uint32 = 4 bytes per token)
            estimated_tokens = int(os.path.getsize(output_file) / 4)
            total_tokens[split_name] = estimated_tokens
            continue
            
        print(f"\n  Processing {split_name}...")
        split_data = dataset[split_name]
        
        # Tokenize
        print(f"    Tokenizing...")
        tokenized = split_data.map(
            process_batch,
            batched=True,
            batch_size=1000,
            remove_columns=list(split_data.features.keys()),
            num_proc=NUM_PROC,
            desc="    Progress"
        )
        
        # Calculate total size
        arr_len = np.sum(tokenized['len'], dtype=np.uint64)
        print(f"    Total tokens: {arr_len:,}")
        
        # Write to binary file (optimized with 32 large shards)
        output_name = "val" if split_name == "validation" else split_name
        output_file = os.path.join(os.path.dirname(__file__), f'{output_name}.bin')
        dtype = np.uint32  # Qwen3 vocab requires uint32
        
        print(f"    Writing {output_name}.bin...")
        arr = np.memmap(output_file, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # Write in 32 large shards (16x faster than 1024 small shards)
        num_shards = 32
        idx = 0
        
        for shard_idx in tqdm(range(num_shards), desc="    Writing"):
            shard = tokenized.shard(num_shards=num_shards, index=shard_idx, 
                                   contiguous=True).with_format('numpy')
            shard_tokens = np.concatenate(shard['ids'])
            arr[idx:idx + len(shard_tokens)] = shard_tokens
            idx += len(shard_tokens)
        
        arr.flush()
        del arr
        
        file_size_gb = os.path.getsize(output_file) / 1e9
        print(f"    ✓ {output_name}.bin: {arr_len:,} tokens ({file_size_gb:.2f} GB)")
        total_tokens[split_name] = arr_len
    
    # Save metadata
    print("\n[4/4] Saving metadata...")
    meta = {
        'vocab_size': tokenizer.vocab_size,
        'eos_token_id': tokenizer.eos_token_id,
        'tokenizer': 'qwen3',
        'dataset': 'slimpajama-627b',
        'total_tokens': total_tokens,
    }
    meta_file = os.path.join(os.path.dirname(__file__), 'meta.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"✓ meta.pkl saved")
    
    # Summary
    print("\n" + "="*80)
    print("✅ TOKENIZATION COMPLETE!")
    print("="*80)
    for split_name, token_count in total_tokens.items():
        print(f"  {split_name}: {token_count:,} tokens")
    
    print(f"\n✓ Ready for training!")
    print(f"  Update config: dataset='slimpajama_627b_qwen3'")
