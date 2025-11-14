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
from datasets import load_dataset
from transformers import AutoTokenizer

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

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print(f"✓ Loaded tokenizer (vocab_size={tokenizer.vocab_size})")

# Load dataset
print("\nDownloading SlimPajama-6B dataset...")
dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=False, trust_remote_code=True)

# Take first 6B tokens worth (approximate)
# Assuming ~500 tokens per sample on average, 6B tokens ≈ 12M samples
dataset = dataset.select(range(min(12_000_000, len(dataset))))

# Split into train/val (99%/1%)
print("Splitting into train/val...")
split_dataset = dataset.train_test_split(test_size=0.01, seed=2357)
split_dataset['val'] = split_dataset.pop('test')

# Tokenize
def process(example):
    ids = tokenizer(example['text'], truncation=False, add_special_tokens=False)['input_ids']
    ids.append(tokenizer.eos_token_id)
    return {'ids': ids, 'len': len(ids)}

print("\nTokenizing dataset...")
tokenized = split_dataset.map(
    process,
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

