#!/usr/bin/env python3
"""
SFT Data Preparation Script

Prepares instruction-following datasets for Supervised Fine-Tuning (SFT).
Supports multiple dataset formats and applies ChatML template for Qwen3.

Usage:
    python prepare_sft.py --dataset alpaca --output_dir ./sft_alpaca
    python prepare_sft.py --dataset ultrachat --output_dir ./sft_ultrachat --max_samples 100000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "enhanced_training_system"))

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    from datasets import load_dataset, Dataset, DatasetDict
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not available. Install with: pip install datasets")


# =============================================================================
# SUPPORTED DATASETS
# =============================================================================

SUPPORTED_DATASETS = {
    "alpaca": {
        "hf_name": "tatsu-lab/alpaca",
        "description": "Stanford Alpaca (52K instruction-response pairs)",
        "format": "alpaca",  # instruction, input, output
    },
    "alpaca_cleaned": {
        "hf_name": "yahma/alpaca-cleaned",
        "description": "Cleaned Alpaca dataset (52K, higher quality)",
        "format": "alpaca",
    },
    "dolly": {
        "hf_name": "databricks/databricks-dolly-15k",
        "description": "Databricks Dolly (15K human-generated)",
        "format": "dolly",  # instruction, context, response
    },
    "oasst1": {
        "hf_name": "OpenAssistant/oasst1",
        "description": "OpenAssistant conversations (human feedback)",
        "format": "oasst",
    },
    "ultrachat": {
        "hf_name": "HuggingFaceH4/ultrachat_200k",
        "description": "UltraChat 200K (high-quality synthetic)",
        "format": "ultrachat",  # messages format
    },
    "slimorca": {
        "hf_name": "Open-Orca/SlimOrca",
        "description": "SlimOrca (517K cleaned examples)",
        "format": "sharegpt",  # conversations format
    },
}


# =============================================================================
# CHAT TEMPLATE FORMATTING
# =============================================================================

class ChatMLFormatter:
    """
    Formats conversations using ChatML template for Qwen3.
    
    ChatML Format:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>
    """
    
    # Special token IDs for Qwen3
    IM_START_ID = 151644  # <|im_start|>
    IM_END_ID = 151645    # <|im_end|>
    ENDOFTEXT_ID = 151643 # <|endoftext|>
    
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, harmless, and honest assistant. "
        "Always answer as helpfully as possible while being safe."
    )
    
    def __init__(self, tokenizer, system_prompt: Optional[str] = None):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Verify tokenizer has required special tokens
        self._verify_tokenizer()
    
    def _verify_tokenizer(self):
        """Verify tokenizer has ChatML special tokens."""
        required_tokens = ["<|im_start|>", "<|im_end|>"]
        vocab = self.tokenizer.get_vocab()
        for token in required_tokens:
            if token not in vocab:
                print(f"Warning: Token '{token}' not in tokenizer vocabulary")
    
    def format_conversation(
        self, 
        instruction: str, 
        response: str, 
        input_text: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format a single-turn conversation with ChatML template.
        
        Args:
            instruction: User instruction/question
            response: Assistant response
            input_text: Optional context/input for the instruction
            system_prompt: Optional override for system prompt
        
        Returns:
            Formatted string with ChatML template
        """
        system = system_prompt or self.system_prompt
        
        # Combine instruction with optional input
        if input_text and input_text.strip():
            user_content = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_content = instruction
        
        # Build ChatML format
        chat = f"<|im_start|>system\n{system}<|im_end|>\n"
        chat += f"<|im_start|>user\n{user_content}<|im_end|>\n"
        chat += f"<|im_start|>assistant\n{response}<|im_end|>"
        
        return chat
    
    def format_multi_turn(
        self, 
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format multi-turn conversation with ChatML template.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            system_prompt: Optional system prompt
        
        Returns:
            Formatted string with ChatML template
        """
        system = system_prompt or self.system_prompt
        chat = f"<|im_start|>system\n{system}<|im_end|>\n"
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            chat += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Remove trailing newline
        return chat.rstrip("\n")
    
    def tokenize_with_labels(
        self, 
        text: str,
        max_length: int = 2048,
        mask_instruction: bool = True
    ) -> Dict[str, List[int]]:
        """
        Tokenize text and create labels with optional instruction masking.
        
        Args:
            text: Formatted ChatML text
            max_length: Maximum sequence length
            mask_instruction: If True, mask instruction tokens (label=-100)
        
        Returns:
            Dict with 'input_ids' and 'labels'
        """
        # Tokenize full text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Truncate if necessary
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        if mask_instruction:
            # Create labels with instruction tokens masked
            labels = self._create_masked_labels(text, tokens)
        else:
            # Standard causal LM: labels = input_ids shifted
            labels = tokens.copy()
        
        return {
            "input_ids": tokens,
            "labels": labels,
            "length": len(tokens)
        }
    
    def _create_masked_labels(self, text: str, tokens: List[int]) -> List[int]:
        """
        Create labels where instruction tokens are masked with -100.
        Only compute loss on assistant response tokens.
        """
        labels = [-100] * len(tokens)  # Start with all masked
        
        # Find assistant response boundaries
        # Look for "<|im_start|>assistant\n" and mask everything before it
        assistant_marker = "<|im_start|>assistant\n"
        end_marker = "<|im_end|>"
        
        # Find all assistant response spans
        idx = 0
        while True:
            start_pos = text.find(assistant_marker, idx)
            if start_pos == -1:
                break
            
            # Find the content start (after the marker)
            content_start = start_pos + len(assistant_marker)
            
            # Find the end of this response
            end_pos = text.find(end_marker, content_start)
            if end_pos == -1:
                end_pos = len(text)
            
            # Convert character positions to token positions
            # This is approximate - we tokenize the prefix to find token boundaries
            prefix_before_content = text[:content_start]
            prefix_tokens = self.tokenizer.encode(prefix_before_content, add_special_tokens=False)
            token_start = len(prefix_tokens)
            
            content_text = text[:end_pos]
            content_tokens = self.tokenizer.encode(content_text, add_special_tokens=False)
            token_end = len(content_tokens)
            
            # Unmask the assistant response tokens (set labels = tokens)
            for i in range(token_start, min(token_end, len(tokens))):
                labels[i] = tokens[i]
            
            idx = end_pos + len(end_marker)
        
        return labels


# =============================================================================
# DATASET CONVERTERS
# =============================================================================

def convert_alpaca_format(example: Dict, formatter: ChatMLFormatter) -> Dict:
    """Convert Alpaca format (instruction, input, output) to ChatML."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if not instruction or not output:
        return None
    
    text = formatter.format_conversation(instruction, output, input_text)
    return {"text": text, "instruction": instruction, "response": output}


def convert_dolly_format(example: Dict, formatter: ChatMLFormatter) -> Dict:
    """Convert Dolly format (instruction, context, response) to ChatML."""
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")
    
    if not instruction or not response:
        return None
    
    text = formatter.format_conversation(instruction, response, context)
    return {"text": text, "instruction": instruction, "response": response}


def convert_ultrachat_format(example: Dict, formatter: ChatMLFormatter) -> Dict:
    """Convert UltraChat format (messages list) to ChatML."""
    messages = example.get("messages", [])
    
    if len(messages) < 2:
        return None
    
    # Filter to user/assistant messages only
    filtered = [m for m in messages if m.get("role") in ["user", "assistant"]]
    
    if len(filtered) < 2:
        return None
    
    text = formatter.format_multi_turn(filtered)
    
    # Extract first instruction and last response for metadata
    instruction = filtered[0].get("content", "") if filtered[0].get("role") == "user" else ""
    response = filtered[-1].get("content", "") if filtered[-1].get("role") == "assistant" else ""
    
    return {"text": text, "instruction": instruction, "response": response}


def convert_sharegpt_format(example: Dict, formatter: ChatMLFormatter) -> Dict:
    """Convert ShareGPT/SlimOrca format (conversations) to ChatML."""
    conversations = example.get("conversations", [])
    
    if len(conversations) < 2:
        return None
    
    # Map 'human'/'gpt' to 'user'/'assistant'
    messages = []
    for conv in conversations:
        role = conv.get("from", "")
        content = conv.get("value", "")
        
        if role == "human":
            messages.append({"role": "user", "content": content})
        elif role == "gpt":
            messages.append({"role": "assistant", "content": content})
        elif role == "system":
            # Handle system messages separately
            pass
    
    if len(messages) < 2:
        return None
    
    text = formatter.format_multi_turn(messages)
    
    instruction = messages[0].get("content", "") if messages else ""
    response = messages[-1].get("content", "") if messages else ""
    
    return {"text": text, "instruction": instruction, "response": response}


def convert_oasst_format(example: Dict, formatter: ChatMLFormatter) -> Dict:
    """Convert OpenAssistant format to ChatML (single turn only)."""
    # OASST has complex tree structure; for simplicity, use text directly
    text = example.get("text", "")
    role = example.get("role", "")
    
    # Skip non-assistant messages (we need pairs)
    # This is simplified - full implementation would reconstruct conversations
    if role != "assistant":
        return None
    
    # For OASST, we'd need to reconstruct conversation trees
    # This is a placeholder - consider using a preprocessed version
    return None


CONVERTERS = {
    "alpaca": convert_alpaca_format,
    "dolly": convert_dolly_format,
    "ultrachat": convert_ultrachat_format,
    "sharegpt": convert_sharegpt_format,
    "oasst": convert_oasst_format,
}


# =============================================================================
# MAIN PREPARATION FUNCTIONS
# =============================================================================

def load_and_convert_dataset(
    dataset_name: str,
    tokenizer,
    max_samples: Optional[int] = None,
    max_length: int = 2048,
    system_prompt: Optional[str] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Load dataset and convert to tokenized SFT format.
    
    Returns:
        Tuple of (examples list, statistics dict)
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(SUPPORTED_DATASETS.keys())}")
    
    config = SUPPORTED_DATASETS[dataset_name]
    formatter = ChatMLFormatter(tokenizer, system_prompt)
    converter = CONVERTERS[config["format"]]
    
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    print(f"HuggingFace name: {config['hf_name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}\n")
    
    # Load from HuggingFace
    try:
        dataset = load_dataset(config["hf_name"], split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    print(f"Loaded {len(dataset)} examples")
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        print(f"Sampled {max_samples} examples")
    
    # Convert and tokenize
    examples = []
    stats = {
        "total": len(dataset),
        "converted": 0,
        "skipped": 0,
        "truncated": 0,
        "total_tokens": 0,
        "avg_tokens": 0,
        "max_tokens": 0,
        "min_tokens": float("inf"),
    }
    
    print("\nConverting to ChatML format and tokenizing...")
    for example in tqdm(dataset, desc="Processing"):
        converted = converter(example, formatter)
        
        if converted is None:
            stats["skipped"] += 1
            continue
        
        # Tokenize with labels
        tokenized = formatter.tokenize_with_labels(
            converted["text"],
            max_length=max_length,
            mask_instruction=True
        )
        
        if tokenized["length"] == max_length:
            stats["truncated"] += 1
        
        examples.append({
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "length": tokenized["length"],
            "text": converted["text"],  # Keep for debugging
        })
        
        stats["converted"] += 1
        stats["total_tokens"] += tokenized["length"]
        stats["max_tokens"] = max(stats["max_tokens"], tokenized["length"])
        stats["min_tokens"] = min(stats["min_tokens"], tokenized["length"])
    
    if stats["converted"] > 0:
        stats["avg_tokens"] = stats["total_tokens"] / stats["converted"]
    
    if stats["min_tokens"] == float("inf"):
        stats["min_tokens"] = 0
    
    return examples, stats


def save_sft_dataset(
    examples: List[Dict],
    output_dir: str,
    val_split: float = 0.02,
    seed: int = 42
):
    """
    Save prepared SFT dataset to disk.
    
    Saves:
        - train.bin / val.bin: Memory-mapped token arrays
        - train_labels.bin / val_labels.bin: Label arrays
        - meta.json: Dataset metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(examples))
    
    val_size = int(len(examples) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    print(f"\nSplitting dataset:")
    print(f"  Train: {len(train_indices)} examples")
    print(f"  Val: {len(val_indices)} examples")
    
    # Save function
    def save_split(split_indices, prefix):
        # Concatenate all sequences with separator
        all_input_ids = []
        all_labels = []
        lengths = []
        
        for idx in tqdm(split_indices, desc=f"Saving {prefix}"):
            ex = examples[idx]
            all_input_ids.extend(ex["input_ids"])
            all_labels.extend(ex["labels"])
            lengths.append(ex["length"])
        
        # Save as numpy arrays
        input_ids_arr = np.array(all_input_ids, dtype=np.int32)
        labels_arr = np.array(all_labels, dtype=np.int32)
        
        input_ids_arr.tofile(output_path / f"{prefix}.bin")
        labels_arr.tofile(output_path / f"{prefix}_labels.bin")
        
        return {
            "num_examples": len(split_indices),
            "total_tokens": len(all_input_ids),
            "lengths": lengths,
        }
    
    train_info = save_split(train_indices, "train")
    val_info = save_split(val_indices, "val")
    
    # Save metadata
    meta = {
        "dataset_type": "sft",
        "format": "chatml",
        "train": train_info,
        "val": val_info,
        "total_examples": len(examples),
        "val_split": val_split,
    }
    
    # Don't save lengths array in JSON (too large)
    meta["train"]["avg_length"] = np.mean(train_info["lengths"])
    meta["val"]["avg_length"] = np.mean(val_info["lengths"])
    del meta["train"]["lengths"]
    del meta["val"]["lengths"]
    
    with open(output_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"  train.bin: {train_info['total_tokens']:,} tokens")
    print(f"  val.bin: {val_info['total_tokens']:,} tokens")
    
    return meta


def save_jsonl_format(
    examples: List[Dict],
    output_dir: str,
    val_split: float = 0.02,
    seed: int = 42
):
    """
    Save as JSONL format (alternative to binary).
    Easier to inspect and compatible with more tools.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(examples))
    
    val_size = int(len(examples) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    def save_jsonl(split_indices, filename):
        with open(output_path / filename, "w") as f:
            for idx in split_indices:
                ex = examples[idx]
                # Save only essential fields
                record = {
                    "input_ids": ex["input_ids"],
                    "labels": ex["labels"],
                }
                f.write(json.dumps(record) + "\n")
    
    save_jsonl(train_indices, "train.jsonl")
    save_jsonl(val_indices, "val.jsonl")
    
    print(f"\nJSONL saved to: {output_path}")
    print(f"  train.jsonl: {len(train_indices)} examples")
    print(f"  val.jsonl: {len(val_indices)} examples")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT datasets for instruction fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported datasets:
  alpaca         - Stanford Alpaca (52K examples)
  alpaca_cleaned - Cleaned Alpaca (52K, higher quality)
  dolly          - Databricks Dolly (15K human-generated)
  ultrachat      - UltraChat 200K (high-quality synthetic)
  slimorca       - SlimOrca (517K cleaned examples)

Examples:
  python prepare_sft.py --dataset alpaca --output_dir ./sft_alpaca
  python prepare_sft.py --dataset ultrachat --max_samples 50000
  python prepare_sft.py --dataset slimorca --max_length 4096
        """
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        choices=list(SUPPORTED_DATASETS.keys()),
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ./sft_{dataset})"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (default: qwen3_tokenizer from enhanced_training_system)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.02,
        help="Validation split ratio (default: 0.02)"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["binary", "jsonl", "both"],
        default="both",
        help="Output format (default: both)"
    )
    
    args = parser.parse_args()
    
    # Validate dependencies
    if not HF_AVAILABLE:
        print("Error: transformers library required. Install with: pip install transformers")
        sys.exit(1)
    
    if not DATASETS_AVAILABLE:
        print("Error: datasets library required. Install with: pip install datasets")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"./sft_{args.dataset}"
    
    # Load tokenizer
    if args.tokenizer_path is None:
        # Default to Qwen3 tokenizer
        default_path = Path(__file__).parent.parent.parent / "enhanced_training_system" / "qwen3_tokenizer"
        if default_path.exists():
            args.tokenizer_path = str(default_path)
        else:
            print("Error: Tokenizer not found. Specify with --tokenizer_path")
            sys.exit(1)
    
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Load and convert dataset
    examples, stats = load_and_convert_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        system_prompt=args.system_prompt,
    )
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total examples: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Truncated: {stats['truncated']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Avg tokens/example: {stats['avg_tokens']:.1f}")
    print(f"Max tokens: {stats['max_tokens']}")
    print(f"Min tokens: {stats['min_tokens']}")
    print(f"{'='*60}")
    
    # Save dataset
    if args.format in ["binary", "both"]:
        save_sft_dataset(
            examples=examples,
            output_dir=args.output_dir,
            val_split=args.val_split,
        )
    
    if args.format in ["jsonl", "both"]:
        save_jsonl_format(
            examples=examples,
            output_dir=args.output_dir,
            val_split=args.val_split,
        )
    
    print("\n✅ SFT data preparation complete!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()

