#!/usr/bin/env python3
"""
Tokenize SlimPajama-627B shards using a manifest produced by build_manifest.py.

Usage example (single process):
    python tokenize_from_manifest.py \
        --manifest manifests/slimpajama_manifest.jsonl \
        --split train \
        --tokenizer ../../qwen3_tokenizer \
        --output tokens/train_part0.bin

For multi-host / multi-process runs, shard work manually:
    # Process 0 of 8
    python tokenize_from_manifest.py ... --process-index 0 --process-count 8
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np
import zstandard as zstd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize SlimPajama shards from manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest JSONL file.")
    parser.add_argument("--split", type=str, default="train", help="Split to tokenize (default: %(default)s).")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("../../qwen3_tokenizer"),
        help="Tokenizer directory or HF identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .bin file (uint32). Required for single/worker modes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to place auto-launched worker outputs.",
    )
    parser.add_argument(
        "--process-index",
        type=int,
        default=0,
        help="Index of this worker (for manual sharding).",
    )
    parser.add_argument(
        "--process-count",
        type=int,
        default=1,
        help="Total number of workers (must be >=1).",
    )
    parser.add_argument(
        "--spawn-workers",
        type=int,
        default=0,
        help=(
            "If >0, launch this many local worker subprocesses automatically. "
            "Set to -1 to use os.cpu_count(). Requires --output-dir."
        ),
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Optional limit for debugging (process only the first N assigned shards).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of JSON rows processed (per worker).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output file instead of overwriting (useful for resume).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Log progress every N examples (default: %(default)s).",
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_path: Path | str):
    print(f"Loading tokenizer from {tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have eos_token_id defined.")
    print(f"  Vocab size: {tokenizer.vocab_size:,} | eos_token_id: {tokenizer.eos_token_id}")
    return tokenizer


def iter_manifest(manifest_path: Path, split: str) -> Iterator[dict]:
    with manifest_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)
            if entry.get("split") == split:
                yield entry


def assign_shards(
    entries: Iterable[dict],
    process_index: int,
    process_count: int,
    max_shards: Optional[int],
) -> List[dict]:
    assigned: List[dict] = []
    for idx, entry in enumerate(entries):
        if idx % process_count != process_index:
            continue
        assigned.append(entry)
        if max_shards is not None and len(assigned) >= max_shards:
            break
    return assigned


def stream_jsonl(path: Path) -> Iterator[dict]:
    compressor = zstd.ZstdDecompressor()
    with path.open("rb") as fh:
        with compressor.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def tokenize_shard(
    shard_path: Path,
    tokenizer,
    bin_file,
    eos_id: int,
    max_examples: Optional[int],
    log_interval: int,
    example_offset: int,
) -> int:
    """Returns number of examples processed."""
    count = 0
    for example in stream_jsonl(shard_path):
        text = example.get("text", "")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(eos_id)
        np.asarray(ids, dtype=np.uint32).tofile(bin_file)
        count += 1
        if count % log_interval == 0:
            print(f"    [{example_offset + count:,} examples] ...")
        if max_examples is not None and (example_offset + count) >= max_examples:
            break
    return count


def main() -> None:
    args = parse_args()
    if args.process_count < 1:
        raise ValueError("--process-count must be >= 1")
    if not (0 <= args.process_index < args.process_count):
        raise ValueError("--process-index must be in [0, process_count)")

    if args.spawn_workers:
        launch_local_workers(args)
        return

    if args.output is None:
        raise ValueError("--output is required in single/worker mode.")

    tokenizer = load_tokenizer(args.tokenizer)
    manifest_entries = assign_shards(
        iter_manifest(args.manifest, args.split),
        process_index=args.process_index,
        process_count=args.process_count,
        max_shards=args.max_shards,
    )

    if not manifest_entries:
        raise RuntimeError("No shards assigned to this worker. Check split/index/count.")

    print(f"Worker {args.process_index}/{args.process_count}: {len(manifest_entries)} shard(s) to process.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if args.append and args.output.exists() else "wb"

    total_examples = 0
    total_tokens_written = 0
    with args.output.open(mode) as bin_file:
        for shard_idx, entry in enumerate(manifest_entries, start=1):
            shard_path = Path(entry["path"]).resolve()
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard not found: {shard_path}")

            print(f"[{shard_idx}/{len(manifest_entries)}] {shard_path} ({entry['size_bytes']/1e6:.2f} MB)")
            prev_examples = total_examples
            processed = tokenize_shard(
                shard_path=shard_path,
                tokenizer=tokenizer,
                bin_file=bin_file,
                eos_id=tokenizer.eos_token_id,
                max_examples=args.max_examples,
                log_interval=args.log_interval,
                example_offset=total_examples,
            )
            total_examples += processed
            if processed == 0:
                continue

            # tokens = bytes_written / 4 (uint32). We can compute by checking file size delta.
            bin_file.flush()
            if args.output.exists():
                file_size = args.output.stat().st_size
                total_tokens_written = file_size // 4

            print(
                f"    Processed {processed:,} examples "
                f"(total {total_examples:,}); tokens written so far ≈ {total_tokens_written:,}"
            )

            if args.max_examples is not None and total_examples >= args.max_examples:
                print("Reached --max-examples limit; stopping early.")
                break

    meta = {
        "split": args.split,
        "manifest": str(args.manifest),
        "tokenizer": str(args.tokenizer),
        "output": str(args.output),
        "examples": total_examples,
        "tokens_uint32": total_tokens_written,
        "process_index": args.process_index,
        "process_count": args.process_count,
    }
    meta_path = args.output.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as fout:
        json.dump(meta, fout, indent=2, sort_keys=True)
    print(f"\nDone. Metadata written to {meta_path}")


def launch_local_workers(args: argparse.Namespace) -> None:
    num_workers = args.spawn_workers if args.spawn_workers > 0 else os.cpu_count()
    if num_workers is None or num_workers < 1:
        raise ValueError("Unable to determine worker count.")
    if args.output_dir is None:
        raise ValueError("--output-dir must be provided when --spawn-workers is used.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Spawning {num_workers} worker subprocesses...")
    procs = []
    for idx in range(num_workers):
        worker_output = output_dir / f"{args.split}_worker{idx:03d}.bin"
        cmd = [
            sys.executable,
            __file__,
            "--manifest",
            str(args.manifest),
            "--split",
            args.split,
            "--tokenizer",
            str(args.tokenizer),
            "--output",
            str(worker_output),
            "--process-index",
            str(idx),
            "--process-count",
            str(num_workers),
            "--log-interval",
            str(args.log_interval),
        ]
        if args.max_shards is not None:
            cmd += ["--max-shards", str(args.max_shards)]
        if args.max_examples is not None:
            cmd += ["--max-examples", str(args.max_examples)]
        procs.append(subprocess.Popen(cmd))

    exit_codes = [p.wait() for p in procs]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"One or more workers exited with non-zero status: {exit_codes}")
    print("All workers completed. Concatenate the worker .bin files when ready.")


if __name__ == "__main__":
    main()


