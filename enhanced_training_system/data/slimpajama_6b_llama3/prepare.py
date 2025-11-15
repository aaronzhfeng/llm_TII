"""
Prepare SlimPajama-6B dataset with LLaMA-3 tokenizer.

This script downloads and tokenizes the SlimPajama-6B dataset using the
LLaMA-3.1 tokenizer (128,256 vocab size).

Usage:
    python prepare.py

Output:
    - train.bin: Tokenized training data (~6GB)
    - val.bin: Tokenized validation data (~30MB)
    - meta.pkl: Metadata (vocab size, etc.)

Time: ~20-40 minutes depending on CPU
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import warnings
import logging
import sys
from contextlib import contextmanager

# Suppress ALL tokenizer warnings about long sequences
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

@contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output"""
    devnull = open(os.devnull, 'w')
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()

# Download and prepare tokenizer
print("Loading LLaMA-3.1 tokenizer...")
tokenizer_path = "../../llama3_tokenizer"
if not os.path.exists(tokenizer_path):
    print(f"Tokenizer not found at {tokenizer_path}")
    print("Please download it first:")
    print("  cd ../../ && python -c \"from transformers import AutoTokenizer; ")
    print("    tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B'); ")
    print("    tok.save_pretrained('./llama3_tokenizer')\"")
    exit(1)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
if tokenizer.is_fast:
    print(f"✓ Loaded FAST tokenizer (vocab_size={tokenizer.vocab_size}) ⚡")
else:
    print(f"✓ Loaded tokenizer (vocab_size={tokenizer.vocab_size}) [slow version]")

# Load dataset
print("\nDownloading SlimPajama-6B dataset...")
print("This will download ~6GB and may take 5-15 minutes...")
dataset = load_dataset("DKYoon/SlimPajama-6B", num_proc=os.cpu_count())
print(f"✓ Dataset loaded: {dataset}")

# Check dataset structure and create train/val split
print("\nCreating train/validation split...")
if 'train' in dataset and 'validation' not in dataset:
    print("Splitting train set (99% train, 1% validation)...")
    split_dataset = dataset["train"].train_test_split(test_size=0.01, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')
elif 'train' in dataset and 'validation' in dataset:
    print("Using existing splits...")
    split_dataset = DatasetDict({
        'train': dataset['train'],
        'val': dataset['validation']
    })
else:
    print("Dataset structure unknown, creating 99/1 split...")
    split_dataset = dataset["train"].train_test_split(test_size=0.01, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

print(f"✓ Split complete:")
for split_name, split_data in split_dataset.items():
    print(f"  {split_name}: {len(split_data):,} examples")

# Tokenize (batched for speed)
# Track sequences that exceed max length
long_sequences_count = 0
max_model_length = tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') else 131072

def process_batch(examples):
    """Tokenize a batch of examples (much faster!)"""
    global long_sequences_count
    all_ids = []
    all_lens = []
    
    for text in examples['text']:
        with warnings.catch_warnings(), suppress_stderr():
            warnings.simplefilter("ignore")
            ids = tokenizer(text, truncation=False, add_special_tokens=False)['input_ids']
        ids.append(tokenizer.eos_token_id)
        
        # Track long sequences silently
        if len(ids) > max_model_length:
            long_sequences_count += 1
        
        all_ids.append(ids)
        all_lens.append(len(ids))
    
    return {'ids': all_ids, 'len': all_lens}

print("\nTokenizing dataset with batching (2-5× faster)...")
print(f"(Sequences exceeding {max_model_length:,} tokens will be tracked but not truncated)")
tokenized = split_dataset.map(
    process_batch,
    batched=True,
    batch_size=1000,
    remove_columns=['text', 'meta'],
    desc="Tokenizing",
    num_proc=os.cpu_count(),
)

# Concatenate all ids into single array
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint32  # LLaMA-3 tokenizer requires uint32 for 128K vocab
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'Writing {split}.bin'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

    print(f"✓ {split}.bin: {arr_len:,} tokens")

# Save metadata
meta = {
    'vocab_size': tokenizer.vocab_size,
    'eos_token_id': tokenizer.eos_token_id,
    'tokenizer': 'llama3',
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("\n✅ Dataset preparation complete!")
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Train tokens: {tokenized['train'].num_rows:,}")
print(f"   Val tokens: {tokenized['val'].num_rows:,}")
print(f"\n📊 Long sequence summary:")
print(f"   Sequences exceeding {max_model_length:,} tokens: {long_sequences_count:,}")
total_sequences = tokenized['train'].num_rows + tokenized['val'].num_rows
print(f"   Percentage: {100 * long_sequences_count / total_sequences:.2f}%")
print(f"   Note: These sequences were kept as-is (not truncated)")

