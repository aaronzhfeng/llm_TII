#!/usr/bin/env python3
"""
DPO Data Preparation Script

Prepares preference datasets for Direct Preference Optimization (DPO).
Each example contains: prompt, chosen response, rejected response.

Usage:
    python prepare_dpo.py --dataset ultrafeedback --output_dir ./dpo_ultrafeedback
    python prepare_dpo.py --dataset hh_rlhf --max_samples 50000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "enhanced_training_system"))

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available")

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets not available")


# =============================================================================
# SUPPORTED PREFERENCE DATASETS
# =============================================================================

SUPPORTED_DATASETS = {
    "ultrafeedback": {
        "hf_name": "argilla/ultrafeedback-binarized-preferences-cleaned",
        "description": "UltraFeedback binarized preferences (60K)",
        "format": "ultrafeedback",
    },
    "hh_rlhf": {
        "hf_name": "Anthropic/hh-rlhf",
        "description": "Anthropic HH-RLHF (human harmlessness/helpfulness)",
        "format": "hh_rlhf",
    },
    "orca_dpo": {
        "hf_name": "Intel/orca_dpo_pairs",
        "description": "Intel Orca DPO pairs (12K)",
        "format": "orca_dpo",
    },
    "distilabel_capybara": {
        "hf_name": "argilla/distilabel-capybara-dpo-7k-binarized",
        "description": "Distilabel Capybara DPO (7K)",
        "format": "capybara",
    },
}


# =============================================================================
# CHAT TEMPLATE
# =============================================================================

class DPOFormatter:
    """
    Formats preference pairs for DPO training with ChatML template.
    """
    
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, harmless, and honest assistant."
    )
    
    def __init__(self, tokenizer, system_prompt: Optional[str] = None):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
    
    def format_single_turn(
        self,
        prompt: str,
        response: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Format single turn with ChatML template."""
        system = system_prompt or self.system_prompt
        
        chat = f"<|im_start|>system\n{system}<|im_end|>\n"
        chat += f"<|im_start|>user\n{prompt}<|im_end|>\n"
        chat += f"<|im_start|>assistant\n{response}<|im_end|>"
        
        return chat
    
    def format_multi_turn(
        self,
        conversation: List[Dict[str, str]],
        final_response: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Format multi-turn conversation with ChatML template."""
        system = system_prompt or self.system_prompt
        chat = f"<|im_start|>system\n{system}<|im_end|>\n"
        
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chat += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        chat += f"<|im_start|>assistant\n{final_response}<|im_end|>"
        
        return chat
    
    def tokenize_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        max_length: int = 2048
    ) -> Dict:
        """
        Tokenize a preference pair.
        
        Returns dict with:
            - chosen_ids, chosen_labels
            - rejected_ids, rejected_labels
        """
        chosen_text = self.format_single_turn(prompt, chosen)
        rejected_text = self.format_single_turn(prompt, rejected)
        
        chosen_ids = self.tokenizer.encode(chosen_text, add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(rejected_text, add_special_tokens=False)
        
        # Create labels with prompt masked
        chosen_labels = self._create_labels(chosen_text, chosen_ids)
        rejected_labels = self._create_labels(rejected_text, rejected_ids)
        
        # Truncate
        if len(chosen_ids) > max_length:
            chosen_ids = chosen_ids[:max_length]
            chosen_labels = chosen_labels[:max_length]
        
        if len(rejected_ids) > max_length:
            rejected_ids = rejected_ids[:max_length]
            rejected_labels = rejected_labels[:max_length]
        
        return {
            "chosen_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "prompt": prompt,
        }
    
    def _create_labels(self, text: str, tokens: List[int]) -> List[int]:
        """Create labels with prompt tokens masked (-100)."""
        labels = [-100] * len(tokens)
        
        # Find assistant response start
        assistant_marker = "<|im_start|>assistant\n"
        end_marker = "<|im_end|>"
        
        idx = 0
        while True:
            start_pos = text.find(assistant_marker, idx)
            if start_pos == -1:
                break
            
            content_start = start_pos + len(assistant_marker)
            end_pos = text.find(end_marker, content_start)
            if end_pos == -1:
                end_pos = len(text)
            
            # Convert positions to tokens
            prefix_tokens = self.tokenizer.encode(text[:content_start], add_special_tokens=False)
            content_tokens = self.tokenizer.encode(text[:end_pos], add_special_tokens=False)
            
            token_start = len(prefix_tokens)
            token_end = len(content_tokens)
            
            # Unmask response tokens
            for i in range(token_start, min(token_end, len(tokens))):
                labels[i] = tokens[i]
            
            idx = end_pos + len(end_marker)
        
        return labels


# =============================================================================
# DATASET CONVERTERS
# =============================================================================

def convert_ultrafeedback(example: Dict, formatter: DPOFormatter) -> Optional[Dict]:
    """Convert UltraFeedback format."""
    prompt = example.get("prompt", "")
    chosen_list = example.get("chosen", [])
    rejected_list = example.get("rejected", [])
    
    # Extract content from message lists
    if isinstance(chosen_list, list) and len(chosen_list) > 0:
        chosen = chosen_list[-1].get("content", "") if isinstance(chosen_list[-1], dict) else str(chosen_list[-1])
    else:
        return None
    
    if isinstance(rejected_list, list) and len(rejected_list) > 0:
        rejected = rejected_list[-1].get("content", "") if isinstance(rejected_list[-1], dict) else str(rejected_list[-1])
    else:
        return None
    
    if not prompt or not chosen or not rejected:
        return None
    
    return formatter.tokenize_pair(prompt, chosen, rejected)


def convert_hh_rlhf(example: Dict, formatter: DPOFormatter) -> Optional[Dict]:
    """Convert Anthropic HH-RLHF format."""
    chosen_text = example.get("chosen", "")
    rejected_text = example.get("rejected", "")
    
    if not chosen_text or not rejected_text:
        return None
    
    # Parse the conversation format (Human: ... Assistant: ...)
    def parse_conversation(text):
        parts = text.split("Human: ")
        if len(parts) < 2:
            return None, None
        
        last_exchange = parts[-1]
        if "Assistant: " not in last_exchange:
            return None, None
        
        human_part, assistant_part = last_exchange.split("Assistant: ", 1)
        return human_part.strip(), assistant_part.strip()
    
    prompt_c, chosen = parse_conversation(chosen_text)
    prompt_r, rejected = parse_conversation(rejected_text)
    
    if not prompt_c or not chosen or not rejected:
        return None
    
    # Use prompt from chosen (they should be the same)
    return formatter.tokenize_pair(prompt_c, chosen, rejected)


def convert_orca_dpo(example: Dict, formatter: DPOFormatter) -> Optional[Dict]:
    """Convert Intel Orca DPO format."""
    system = example.get("system", "")
    question = example.get("question", "")
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")
    
    if not question or not chosen or not rejected:
        return None
    
    prompt = question
    if system:
        prompt = f"{system}\n\n{question}"
    
    return formatter.tokenize_pair(prompt, chosen, rejected)


def convert_capybara(example: Dict, formatter: DPOFormatter) -> Optional[Dict]:
    """Convert Capybara DPO format."""
    chosen_list = example.get("chosen", [])
    rejected_list = example.get("rejected", [])
    
    if not chosen_list or not rejected_list:
        return None
    
    # Extract prompt and responses
    prompt = chosen_list[0].get("content", "") if len(chosen_list) > 0 else ""
    chosen = chosen_list[-1].get("content", "") if len(chosen_list) > 1 else ""
    rejected = rejected_list[-1].get("content", "") if len(rejected_list) > 1 else ""
    
    if not prompt or not chosen or not rejected:
        return None
    
    return formatter.tokenize_pair(prompt, chosen, rejected)


CONVERTERS = {
    "ultrafeedback": convert_ultrafeedback,
    "hh_rlhf": convert_hh_rlhf,
    "orca_dpo": convert_orca_dpo,
    "capybara": convert_capybara,
}


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_and_convert_dpo_dataset(
    dataset_name: str,
    tokenizer,
    max_samples: Optional[int] = None,
    max_length: int = 2048,
    system_prompt: Optional[str] = None,
) -> Tuple[List[Dict], Dict]:
    """Load and convert DPO dataset."""
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = SUPPORTED_DATASETS[dataset_name]
    formatter = DPOFormatter(tokenizer, system_prompt)
    converter = CONVERTERS[config["format"]]
    
    print(f"\n{'='*60}")
    print(f"Loading DPO dataset: {dataset_name}")
    print(f"HuggingFace name: {config['hf_name']}")
    print(f"{'='*60}\n")
    
    # Load dataset
    try:
        dataset = load_dataset(config["hf_name"], split="train")
    except Exception as e:
        # Try with train_prefs for some datasets
        try:
            dataset = load_dataset(config["hf_name"], split="train_prefs")
        except:
            raise e
    
    print(f"Loaded {len(dataset)} examples")
    
    if max_samples and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
        print(f"Sampled {max_samples} examples")
    
    # Convert
    examples = []
    stats = {
        "total": len(dataset),
        "converted": 0,
        "skipped": 0,
        "truncated": 0,
    }
    
    print("\nConverting to DPO format...")
    for example in tqdm(dataset, desc="Processing"):
        converted = converter(example, formatter)
        
        if converted is None:
            stats["skipped"] += 1
            continue
        
        if len(converted["chosen_ids"]) == max_length or len(converted["rejected_ids"]) == max_length:
            stats["truncated"] += 1
        
        examples.append(converted)
        stats["converted"] += 1
    
    return examples, stats


def save_dpo_dataset(
    examples: List[Dict],
    output_dir: str,
    val_split: float = 0.05,
    seed: int = 42
):
    """Save DPO dataset as JSONL."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(seed)
    indices = np.random.permutation(len(examples))
    
    val_size = int(len(examples) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    print(f"\nSplitting: {len(train_indices)} train, {len(val_indices)} val")
    
    def save_jsonl(split_indices, filename):
        with open(output_path / filename, "w") as f:
            for idx in split_indices:
                f.write(json.dumps(examples[idx]) + "\n")
    
    save_jsonl(train_indices, "train.jsonl")
    save_jsonl(val_indices, "val.jsonl")
    
    # Save metadata
    meta = {
        "dataset_type": "dpo",
        "train_examples": len(train_indices),
        "val_examples": len(val_indices),
    }
    with open(output_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nDataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO datasets")
    
    parser.add_argument("--dataset", type=str, required=True, choices=list(SUPPORTED_DATASETS.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--system_prompt", type=str, default=None)
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE or not DATASETS_AVAILABLE:
        print("Error: Required libraries not available")
        sys.exit(1)
    
    if args.output_dir is None:
        args.output_dir = f"./dpo_{args.dataset}"
    
    if args.tokenizer_path is None:
        default_path = Path(__file__).parent.parent.parent / "enhanced_training_system" / "qwen3_tokenizer"
        if default_path.exists():
            args.tokenizer_path = str(default_path)
        else:
            print("Error: Tokenizer not found")
            sys.exit(1)
    
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    examples, stats = load_and_convert_dpo_dataset(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
        system_prompt=args.system_prompt,
    )
    
    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Truncated: {stats['truncated']}")
    
    save_dpo_dataset(
        examples=examples,
        output_dir=args.output_dir,
        val_split=args.val_split,
    )
    
    print("\n✅ DPO data preparation complete!")


if __name__ == "__main__":
    main()

