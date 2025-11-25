#!/usr/bin/env python3
"""
Simple download script for SlimPajama-627B dataset.
No parallel workers, throttled to avoid rate limits on small files.

Usage:
    python3 download.py

Output: Downloads to /raid/zhf004/huggingface_cache/
Time: ~8-12 hours (sequential, but reliable)
"""

import os
import sys
import time

# Set cache to /raid/ (24TB available)
os.environ['HF_HOME'] = '/raid/zhf004/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/raid/zhf004/huggingface_cache/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/raid/zhf004/huggingface_cache/transformers'

# Throttle to avoid rate limits (especially for small validation/test files)
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

print("="*80)
print("SlimPajama-627B Simple Download (Throttled)")
print("="*80)
print()
print(f"Cache location: /raid/zhf004/huggingface_cache/")
print(f"Dataset: cerebras/SlimPajama-627B")
print(f"Size: ~895GB compressed, 627B tokens")
print(f"Throttled: ~2 API calls/sec (under 8.33 limit)")
print()
print("Starting download (sequential, no rate limits)...")
print("This will take 8-12 hours. You can safely Ctrl+C and restart later.")
print("Progress is cached - already downloaded files will be skipped.")
print()

try:
    from datasets import load_dataset
    
    # Simple, sequential download
    # The datasets library handles downloads efficiently with built-in caching
    # No parallel downloads by default, which avoids rate limit issues
    dataset = load_dataset(
        "cerebras/SlimPajama-627B",
        num_proc=1  # Sequential processing to avoid rate limits
    )
    
    print()
    print("="*80)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*80)
    print()
    print(f"Dataset info: {dataset}")
    print()
    print("Next step: Tokenize the data")
    print("  python3 prepare.py")
    
except KeyboardInterrupt:
    print()
    print("Download interrupted. Progress is saved in cache.")
    print("Run this script again to continue from where you left off.")
    sys.exit(0)
    
except Exception as e:
    print()
    print(f"❌ Error: {e}")
    sys.exit(1)

