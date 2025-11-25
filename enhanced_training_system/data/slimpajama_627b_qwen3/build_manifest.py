#!/usr/bin/env python3
"""
Build a manifest of SlimPajama-627B shards stored in the local HuggingFace cache.

The manifest is a JSONL file where each line describes one `.jsonl.zst` shard:

    {
        "split": "train",
        "chunk": "chunk1",
        "file_index": 0,
        "path": "/raid/.../example_train_0.jsonl.zst",
        "size_bytes": 123456789,
        "compression": "zstd"
    }

Use the manifest to fan out conversion/tokenization jobs across machines.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE = REPO_ROOT / "huggingface_cache"
DEFAULT_DATASET_DIR = (
    DEFAULT_HF_CACHE / "hub" / "datasets--cerebras--SlimPajama-627B"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate shard manifest for SlimPajama-627B."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=(
            "Path to the local HuggingFace dataset mirror "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="Specific snapshot hash to use. Defaults to the newest snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifest.jsonl"),
        help="Path to write the manifest JSONL file (default: %(default)s)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("manifest_summary.json"),
        help="Path to write a summary JSON file (default: %(default)s)",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Store paths relative to the manifest location instead of absolute.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for debugging; stop after visiting this many files.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to include (default: %(default)s)",
    )
    return parser.parse_args()


def resolve_snapshot(dataset_dir: Path, snapshot: Optional[str]) -> Path:
    snapshots_dir = dataset_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots directory at {snapshots_dir}")

    if snapshot:
        root = snapshots_dir / snapshot
        if not root.exists():
            raise FileNotFoundError(
                f"Snapshot {snapshot} not found under {snapshots_dir}"
            )
        return root

    revisions = sorted(
        p for p in snapshots_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not revisions:
        raise FileNotFoundError(f"No snapshots present under {snapshots_dir}")
    return revisions[-1]


def iter_shard_files(
    split_root: Path, max_files: Optional[int] = None
) -> List[Tuple[Path, str]]:
    """Return a sorted list of (path, chunk_name) tuples for a split."""
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory missing: {split_root}")

    collected: List[Tuple[Path, str]] = []
    chunk_dirs = sorted(
        p for p in split_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    for chunk in chunk_dirs:
        shard_paths = sorted(
            p for p in chunk.glob("*.jsonl.zst") if p.is_file()
        )
        for shard in shard_paths:
            collected.append((shard, chunk.name))
            if max_files is not None and len(collected) >= max_files:
                return collected
    return collected


def build_manifest_entries(
    split: str,
    split_root: Path,
    max_files: Optional[int],
    relative_to: Optional[Path],
) -> Tuple[List[Dict], Dict]:
    entries: List[Dict] = []
    stats = {"files": 0, "bytes": 0}

    shard_items = iter_shard_files(split_root, max_files)
    for idx, (shard_path, chunk_name) in enumerate(shard_items):
        path_to_store: Path
        if relative_to:
            path_to_store = shard_path.relative_to(relative_to)
        else:
            path_to_store = shard_path.resolve()

        size_bytes = shard_path.stat().st_size
        entry = {
            "split": split,
            "chunk": chunk_name,
            "file_index": idx,
            "path": str(path_to_store),
            "size_bytes": size_bytes,
            "compression": "zstd",
        }
        entries.append(entry)
        stats["files"] += 1
        stats["bytes"] += size_bytes
    return entries, stats


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    snapshot_root = resolve_snapshot(dataset_dir, args.snapshot)

    print(f"Dataset dir  : {dataset_dir}")
    print(f"Snapshot used: {snapshot_root.name}")

    relative_base = None
    if args.relative_paths:
        relative_base = args.output.resolve().parent
        print(f"Paths stored relative to: {relative_base}")

    manifest_entries: List[Dict] = []
    summary: Dict[str, Dict] = {}

    for split in args.splits:
        split_root = snapshot_root / split
        if not split_root.exists():
            print(f"⚠️  Split '{split}' not found at {split_root}, skipping.")
            continue

        per_split_entries, stats = build_manifest_entries(
            split=split,
            split_root=split_root,
            max_files=args.max_files,
            relative_to=relative_base,
        )

        manifest_entries.extend(per_split_entries)
        summary[split] = stats
        print(
            f"  {split:<11}: {stats['files']:>6} files | "
            f"{stats['bytes']/1e12:8.3f} TB (compressed)"
        )

    if not manifest_entries:
        raise RuntimeError("No files discovered; manifest would be empty.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for entry in manifest_entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with args.summary.open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, sort_keys=True)

    print(f"\nManifest written to {args.output.resolve()}")
    print(f"Summary  written to {args.summary.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()


