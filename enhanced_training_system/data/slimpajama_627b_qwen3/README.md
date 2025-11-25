# SlimPajama-627B Dataset (Qwen3 Tokenizer)

## Overview

This directory contains preparation scripts for the **SlimPajama-627B** dataset, the largest extensively deduplicated, multi-corpora, open-source dataset for training large language models.

- **Dataset**: [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- **Size**: ~895GB compressed, 627B tokens
- **Files**: 59,166 jsonl files
- **Tokenizer**: Qwen3 (151,643 vocab, BBPE)
- **License**: Apache 2.0

## Dataset Details

SlimPajama is a cleaned and deduplicated version of RedPajama-1.2T:
- **Original**: RedPajama 1.2T tokens
- **After cleaning**: 627B tokens (49.6% reduction)
- **Method**: MinHashLSH deduplication + quality filtering

### Data Sources

| Source        | SlimPajama | RedPajama | Dedup Rate |
|---------------|------------|-----------|------------|
| CommonCrawl   | 52.2%      | 72.6%     | 63.76%     |
| C4            | 26.7%      | 14.4%     | 6.85%      |
| GitHub        | 5.2%       | 4.9%      | 46.16%     |
| Books         | 4.2%       | 2.1%      | 2.01%      |
| ArXiv         | 4.6%       | 2.3%      | 0.06%      |
| Wikipedia     | 3.8%       | 2.0%      | 2.24%      |
| StackExchange | 3.3%       | 1.7%      | 0.20%      |

### Splits

- **train**: ~627B tokens (main training data)
- **validation**: ~500M tokens (decontaminated)
- **test**: ~500M tokens (decontaminated)

## Setup

### Prerequisites

```bash
# Install required packages
pip install datasets transformers numpy tqdm

# Optional: Login to HuggingFace (if needed)
huggingface-cli login
```

### Tokenizer Setup

The script will automatically look for a local Qwen3 tokenizer at `../../qwen3_tokenizer`. If not found, it will download from HuggingFace.

To save bandwidth, download the tokenizer once:

```bash
# From project root
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
tokenizer.save_pretrained('qwen3_tokenizer')
"
```

## Usage

### Test Mode (Recommended First)

Test with a small subset to verify setup:

```bash
cd /raid/zhf004/llm_TII/enhanced_training_system/data/slimpajama_627b_qwen3
python3 prepare.py --max_examples 1000
```

This will process 1000 examples (~5 minutes) to verify everything works.

### High-Scale Workflow (Manifest ➜ Tokenize)

The 627B pipeline is now split into explicit stages so we can parallelize the heavy steps.

#### 1. Download / Mirror
Use `huggingface_hub.snapshot_download` or `huggingface-cli download cerebras/SlimPajama-627B` to mirror the dataset into `/raid/zhf004/huggingface_cache/hub/datasets--cerebras--SlimPajama-627B/`.

#### 2. Build Manifest
Scan the cached shards and write a manifest that downstream jobs can consume:
```bash
cd /raid/zhf004/llm_TII/enhanced_training_system/data/slimpajama_627b_qwen3
python3 build_manifest.py \
  --output manifests/slimpajama_manifest.jsonl \
  --summary manifests/slimpajama_manifest_summary.json
```
The script finds the latest snapshot automatically. Use `--snapshot <hash>` if you need a specific revision.

#### 3. Convert / Tokenize (custom tooling)
- (Optional) Convert `.jsonl.zst` shards to Arrow in parallel using the manifest assignments.
- Tokenize directly from the manifest by distributing shards to workers; each worker writes its own `.bin` file which you can concatenate later.

> The tokenizer implementation is intentionally separated so that you can run it on any machine with access to the mirrored shards (even if the download happened elsewhere).

### Tokenization Script (`tokenize_from_manifest.py`)

Single-process example (train split):

```bash
cd /raid/zhf004/llm_TII/enhanced_training_system/data/slimpajama_627b_qwen3
python3 tokenize_from_manifest.py \
  --manifest manifests/slimpajama_manifest.jsonl \
  --split train \
  --tokenizer ../../qwen3_tokenizer \
  --output tokenized/train_part0.bin
```

Manual sharding across multiple workers:

```bash
# Worker 0 of 8
python3 tokenize_from_manifest.py \
  --manifest manifests/slimpajama_manifest.jsonl \
  --split train \
  --output tokenized/train_part0.bin \
  --process-index 0 \
  --process-count 8

# Worker 1 of 8 (run elsewhere)
python3 tokenize_from_manifest.py \
  --manifest manifests/slimpajama_manifest.jsonl \
  --split train \
  --output tokenized/train_part1.bin \
  --process-index 1 \
  --process-count 8
```

Each worker writes its own `.bin` + `.meta.json`. Concatenate the bins (`cat train_part*.bin > train.bin`) and aggregate token counts from the metadata files once all workers are done.

#### Auto-launch using all local CPU cores

Use `--spawn-workers -1` to automatically launch one subprocess per available CPU core. Provide an output directory and the script will create `train_workerXXX.bin` files:

```bash
python3 tokenize_from_manifest.py \
  --manifest manifests/slimpajama_manifest.jsonl \
  --split train \
  --tokenizer ../../qwen3_tokenizer \
  --output-dir tokenized \
  --spawn-workers -1
```

When all workers finish, concatenate `tokenized/train_worker*.bin` into `train.bin`.

#### Validation / Test splits

Repeat the same procedure for `validation` and `test` splits so evaluation uses the exact same preprocessing:

```bash
# Validation
python3 tokenize_from_manifest.py \
  --manifest manifests/slimpajama_manifest.jsonl \
  --split validation \
  --tokenizer ../../qwen3_tokenizer \
  --output-dir tokenized_val \
  --spawn-workers 32
cat tokenized_val/validation_worker*.bin > val.bin

# Test
python3 tokenize_from_manifest.py \
  --manifest manifests/slimpajama_manifest.jsonl \
  --split test \
  --tokenizer ../../qwen3_tokenizer \
  --output-dir tokenized_test \
  --spawn-workers 32
cat tokenized_test/test_worker*.bin > test.bin
```

## Expected Output

After successful tokenization, you'll have:

```
slimpajama_627b_qwen3/
├── prepare.py          # This preparation script
├── README.md           # This file
├── train.bin           # Training data (~1.2TB as uint32)
├── val.bin             # Validation data (~1GB)
├── test.bin            # Test data (~1GB)
└── meta.pkl            # Metadata (vocab size, etc.)
```

### File Sizes (Approximate)

- `train.bin`: ~1.2TB (627B tokens × 4 bytes)
- `val.bin`: ~1GB (500M tokens × 4 bytes)
- `test.bin`: ~1GB (500M tokens × 4 bytes)
- `meta.pkl`: ~1KB

**Note**: Uses `uint32` (4 bytes per token) because Qwen3 vocab (151,643) exceeds `uint16` limit (65,536).

## Time Estimates

Depends on hardware and network:

| Setup | Mode | Time Estimate |
|-------|------|---------------|
| 8 cores, 100 Mbps | Streaming | 2-3 days |
| 32 cores, 1 Gbps | Streaming | 12-24 hours |
| 32 cores, 1 Gbps | Download first | 8-12 hours download + 4-8 hours tokenization |
| 64 cores, 10 Gbps | Download first | 2-4 hours download + 2-4 hours tokenization |

## Training Configuration

After preparation, update your training config:

```python
# In your config file (e.g., config/full_qwen3_1.8b_b200_optimal.py)

dataset = 'slimpajama_627b_qwen3'
data_dir = 'data/slimpajama_627b_qwen3'

# With 627B tokens, you can train for multiple epochs
max_iters = 157_000  # For 1 epoch at batch_size=4M tokens
# Or
max_iters = 314_000  # For 2 epochs at batch_size=4M tokens
```

## Verification

Check that files were created correctly:

```bash
# List files with sizes
ls -lh *.bin *.pkl

# Check train.bin size (should be ~1.2TB)
du -h train.bin

# Verify metadata
python -c "import pickle; print(pickle.load(open('meta.pkl', 'rb')))"
```

Expected output:
```python
{
    'vocab_size': 151643,
    'eos_token_id': 151643,
    'tokenizer': 'qwen3',
    'dataset': 'slimpajama-627b',
    'total_tokens': {
        'train': 627000000000,  # Approximate
        'validation': 500000000,
        'test': 500000000
    }
}
```

## Troubleshooting

### Out of Memory

If you run out of memory during tokenization:

```bash
# Reduce number of processes
python prepare.py --num_proc 8

# Or use streaming mode
python prepare.py --streaming
```

### Slow Download

If download is very slow:

```bash
# Use streaming mode to start immediately
python prepare.py --streaming

# Or download with aria2 (faster)
pip install huggingface_hub[cli]
huggingface-cli download cerebras/SlimPajama-627B --repo-type dataset
```

### Disk Space Issues

The dataset requires significant space:

- **Raw data**: ~895GB compressed
- **Tokenized data**: ~1.2TB (uint32)
- **Total**: ~2TB recommended

To save space:
1. Use streaming mode (no raw data storage)
2. Delete raw data after tokenization
3. Use external storage

### Network Interruption

If using streaming mode and network fails:
- The script will need to restart
- Consider downloading first for more stability

If downloading first and interrupted:
- HuggingFace will resume from where it left off
- Just run the script again

## Citation

If you use SlimPajama-627B, please cite:

```bibtex
@misc{cerebras2023slimpajama,
  author = {Soboleva, Daria and Al-Khateeb, Faisal and Myers, Robert and Steeves, Jacob R and Hestness, Joel and Dey, Nolan},
  title = {{SlimPajama: A 627B token cleaned and deduplicated version of RedPajama}},
  month = June,
  year = 2023,
  howpublished = {\url{https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama}},
  url = {https://huggingface.co/datasets/cerebras/SlimPajama-627B},
}
```

## Additional Resources

- [SlimPajama Blog Post](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)
- [Cerebras GitHub (Pre-processing Tools)](https://github.com/Cerebras/modelzoo)
- [HuggingFace Dataset Card](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
- [Cerebras Discord Community](https://discord.gg/cerebras)

## Performance Tips

1. **Use Fast Tokenizer**: The script automatically uses Rust-based fast tokenizer if available
2. **Maximize CPU Cores**: More cores = faster tokenization
3. **SSD Storage**: Much faster I/O than HDD
4. **Network**: 1Gbps+ recommended for downloading
5. **Streaming for Large Scale**: If disk space is limited, streaming mode is better

## Comparison with SlimPajama-6B

| Feature | SlimPajama-6B | SlimPajama-627B |
|---------|---------------|-----------------|
| Tokens | 6B | 627B |
| Size (compressed) | ~6GB | ~895GB |
| Size (tokenized) | ~12GB | ~1.2TB |
| Preparation time | 20-40 min | 12-48 hours |
| Use case | Quick testing, small models | Full-scale LLM training |
| Epochs needed | Many | 1-2 sufficient |

## License

Please refer to the licenses of the data subsets:
- [Common Crawl Foundation Terms of Use](https://commoncrawl.org/terms-of-use/)
- [C4 License](https://huggingface.co/datasets/c4)
- GitHub: MIT, BSD, or Apache licenses only
- [ArXiv Terms of Use](https://arxiv.org/help/api/tou)
- [Wikipedia License](https://en.wikipedia.org/wiki/Wikipedia:Copyrights)
- [StackExchange License](https://archive.org/details/stackexchange)

