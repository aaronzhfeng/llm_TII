#!/usr/bin/env python3
"""
Benchmark Evaluation for Qwen3-1.8B on Multiple-Choice QA Tasks

Evaluates the model on:
- OpenBookQA: Science facts + common knowledge reasoning
- ARC-Challenge: Grade-school science (hard)
- ARC-Easy: Grade-school science (easy)

Two evaluation modes:
1. Log-probability scoring (default): Deterministic, standard for benchmarks
2. Generation-based (--mode generate): Uses sampling with temperature/top_k

Usage:
    # Log-prob scoring (default, deterministic)
    python eval_benchmarks.py --checkpoint /path/to/ckpt.pt
    
    # Generation-based with sampling
    python eval_benchmarks.py --checkpoint /path/to/ckpt.pt --mode generate \
        --temperature 0.3 --max-tokens 5 --top-k 10
    
    # Use specific GPU
    CUDA_VISIBLE_DEVICES=6 python eval_benchmarks.py --checkpoint /path/to/ckpt.pt
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Add training system to path
SCRIPTS_DIR = Path(__file__).parent.resolve()
EVAL_ROOT = SCRIPTS_DIR.parent
RESULTS_DIR = EVAL_ROOT / "results" / "benchmark"
TRAINING_SYSTEM = EVAL_ROOT.parent / "enhanced_training_system"
sys.path.insert(0, str(TRAINING_SYSTEM))

from model_builder import ConfigurableGPT
from model_config import ModelArchitectureConfig


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "/raid/zhf004/out-qwen3-1.8b-b200-50h/ckpt_160000.pt"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-1.5B"  # Use HuggingFace directly (local copy has format issues)

BENCHMARKS = {
    "openbookqa": {
        "dataset": "allenai/openbookqa",
        "subset": "main",
        "split": "test",
        "question_key": "question_stem",
        "choices_key": "choices",
        "answer_key": "answerKey",
    },
    "arc_challenge": {
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "split": "test",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answerKey",
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "split": "test",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answerKey",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    """Load model from checkpoint."""
    print(f"📦 Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct model
    model_args = ckpt["model_args"]
    config = ModelArchitectureConfig(**model_args)
    model = ConfigurableGPT(config)
    
    # Strip "_orig_mod." prefix from torch.compile checkpoint
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device, dtype=dtype)  # Cast to bf16 for Flash Attention
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ Model loaded: {param_count:.2f}B parameters")
    print(f"   Iteration: {ckpt.get('iter_num', 'unknown')}")
    print(f"   Val Loss: {ckpt.get('best_val_loss', 'unknown')}")
    
    return model, config


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer."""
    print(f"📦 Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print(f"✅ Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    return tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────
def format_mcq_prompt(question: str, choices: list, labels: list, prompt_style: str = "simple") -> list:
    """
    Format a multiple-choice question for evaluation.
    
    Args:
        question: The question text
        choices: List of answer choices
        labels: List of labels (A, B, C, D)
        prompt_style: 
            - "simple": "Question: X\nAnswer: Y"
            - "mcq": Full MCQ format with all choices shown, then "The answer is: X"
    
    Returns list of prompts for each choice.
    """
    prompts = []
    
    if prompt_style == "mcq":
        # Full MCQ format - shows all choices, then evaluates each answer
        choices_str = "\n".join([f"{l}. {c}" for l, c in zip(labels, choices)])
        for label, choice in zip(labels, choices):
            prompt = f"""Question: {question}

{choices_str}

The answer is: {label}"""
            prompts.append(prompt)
    else:
        # Simple format (default)
        for label, choice in zip(labels, choices):
            prompt = f"Question: {question}\nAnswer: {choice}"
            prompts.append(prompt)
    
    return prompts


def compute_choice_logprobs(model, tokenizer, prompts: list, device: str) -> list:
    """
    Compute log-probability for each choice completion.
    
    For multiple-choice QA, we compute P(answer | question) by:
    1. Tokenizing the full "Question: X\nAnswer: Y" string
    2. Computing the sum of log-probs for the answer tokens only
    """
    logprobs = []
    
    for prompt in prompts:
        # Tokenize
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        seq_len = tokens.shape[1]
        
        if seq_len < 2:
            logprobs.append(-float('inf'))
            continue
        
        # Get model logits - pass dummy targets to get logits for ALL positions
        # (model only returns last position logits when targets=None)
        dummy_targets = tokens.clone()  # Same as input, we just need full logits
        with torch.no_grad():
            logits, _ = model(tokens, targets=dummy_targets)
        
        # logits shape: [batch, seq_len, vocab_size]
        # Compute log-probs for each token (shifted by 1 for autoregressive)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Sum log-probs of actual tokens (teacher forcing)
        # We want P(token_i | tokens_<i)
        # logits[0, i, :] predicts token at position i+1
        token_logprobs = []
        for i in range(seq_len - 1):
            next_token_id = tokens[0, i + 1].item()
            lp = log_probs[0, i, next_token_id].item()
            token_logprobs.append(lp)
        
        # Average log-prob (length-normalized)
        avg_logprob = np.mean(token_logprobs) if token_logprobs else -float('inf')
        logprobs.append(avg_logprob)
    
    return logprobs


def evaluate_benchmark(
    model,
    tokenizer,
    benchmark_name: str,
    device: str = "cuda",
    max_samples: int = None,
    prompt_style: str = "simple",
) -> dict:
    """
    Evaluate model on a benchmark dataset using log-probability scoring.
    
    Args:
        prompt_style: "simple" or "mcq" (with format constraint)
    
    Returns accuracy and detailed results.
    """
    config = BENCHMARKS[benchmark_name]
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluating on {benchmark_name.upper()} (Log-Prob, style={prompt_style})")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"Loading dataset: {config['dataset']} ({config['subset']})...")
    dataset = load_dataset(
        config["dataset"],
        config["subset"],
        split=config["split"],
        trust_remote_code=True,
    )
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    results = []
    
    for item in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
        question = item[config["question_key"]]
        choices_data = item[config["choices_key"]]
        answer_key = item[config["answer_key"]]
        
        # Extract choice texts and labels
        if isinstance(choices_data, dict):
            # ARC/OpenBookQA format: {"text": [...], "label": [...]}
            choice_texts = choices_data["text"]
            choice_labels = choices_data["label"]
        else:
            # Fallback
            choice_texts = choices_data
            choice_labels = ["A", "B", "C", "D"][:len(choices_data)]
        
        # Format prompts for each choice
        prompts = format_mcq_prompt(question, choice_texts, choice_labels, prompt_style)
        
        # Compute log-probs for each choice
        logprobs = compute_choice_logprobs(model, tokenizer, prompts, device)
        
        # Predict: choice with highest log-prob
        pred_idx = np.argmax(logprobs)
        pred_label = choice_labels[pred_idx]
        
        # Check if correct
        is_correct = (pred_label == answer_key)
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "choices": choice_texts,
            "correct_answer": answer_key,
            "predicted_answer": pred_label,
            "is_correct": is_correct,
            "logprobs": logprobs,
        })
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n✅ {benchmark_name.upper()} Results:")
    print(f"   Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def format_generation_prompt(question: str, choices: list, labels: list) -> str:
    """
    Format a multiple-choice question for generation-based evaluation.
    Includes format constraint to output only the answer letter.
    """
    choices_str = "\n".join([f"{label}. {text}" for label, text in zip(labels, choices)])
    prompt = f"""Question: {question}

{choices_str}

Answer with only the letter (A, B, C, or D):"""
    return prompt


def evaluate_benchmark_generation(
    model,
    tokenizer,
    benchmark_name: str,
    device: str = "cuda",
    max_samples: int = None,
    temperature: float = 0.3,
    max_tokens: int = 5,
    top_k: int = 10,
    repetition_penalty: float = 1.0,
) -> dict:
    """
    Evaluate model using generation-based approach.
    
    The model generates an answer letter (A/B/C/D) and we check if it matches.
    """
    config = BENCHMARKS[benchmark_name]
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluating on {benchmark_name.upper()} (Generation Mode)")
    print(f"   Temperature: {temperature}, Max Tokens: {max_tokens}, Top K: {top_k}")
    print(f"{'='*60}")
    
    # Load dataset
    print(f"Loading dataset: {config['dataset']} ({config['subset']})...")
    dataset = load_dataset(
        config["dataset"],
        config["subset"],
        split=config["split"],
        trust_remote_code=True,
    )
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    results = []
    
    for item in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
        question = item[config["question_key"]]
        choices_data = item[config["choices_key"]]
        answer_key = item[config["answer_key"]]
        
        # Extract choice texts and labels
        if isinstance(choices_data, dict):
            choice_texts = choices_data["text"]
            choice_labels = choices_data["label"]
        else:
            choice_texts = choices_data
            choice_labels = ["A", "B", "C", "D"][:len(choices_data)]
        
        # Format prompt with format constraint
        prompt = format_generation_prompt(question, choice_texts, choice_labels)
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        
        # Decode only the generated part
        generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        generated = generated.strip().upper()
        
        # Extract the answer letter (first A/B/C/D found)
        pred_label = None
        for char in generated:
            if char in choice_labels:
                pred_label = char
                break
        
        # If no valid letter found, mark as wrong
        if pred_label is None:
            pred_label = "?"
        
        # Check if correct
        is_correct = (pred_label == answer_key)
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "choices": choice_texts,
            "correct_answer": answer_key,
            "predicted_answer": pred_label,
            "raw_generation": generated,
            "is_correct": is_correct,
        })
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n✅ {benchmark_name.upper()} Results (Generation):")
    print(f"   Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return {
        "benchmark": benchmark_name,
        "mode": "generation",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "settings": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        },
        "results": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-1.8B on QA benchmarks")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER,
                        help="Path to tokenizer")
    parser.add_argument("--benchmark", type=str, default="all",
                        choices=["all", "openbookqa", "arc_challenge", "arc_easy"],
                        help="Which benchmark to run")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per benchmark (for quick testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    # Evaluation mode
    parser.add_argument("--mode", type=str, default="logprob",
                        choices=["logprob", "generate"],
                        help="Evaluation mode: 'logprob' (default) or 'generate'")
    
    # Prompt style (for logprob mode)
    parser.add_argument("--prompt-style", type=str, default="simple",
                        choices=["simple", "mcq"],
                        help="Prompt format: 'simple' (Q&A) or 'mcq' (full choices + 'The answer is:')")
    
    # Generation parameters (only used in generate mode)
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (generate mode only)")
    parser.add_argument("--max-tokens", type=int, default=5,
                        help="Max tokens to generate (generate mode only)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k sampling (generate mode only)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (generate mode only)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load model and tokenizer
    model, model_config = load_model(args.checkpoint, args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Determine which benchmarks to run
    if args.benchmark == "all":
        benchmarks_to_run = list(BENCHMARKS.keys())
    else:
        benchmarks_to_run = [args.benchmark]
    
    # Run evaluations
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "model_params": sum(p.numel() for p in model.parameters()),
        "benchmarks": {},
    }
    
    for benchmark in benchmarks_to_run:
        if args.mode == "generate":
            result = evaluate_benchmark_generation(
                model, tokenizer, benchmark,
                device=args.device,
                max_samples=args.max_samples,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            result = evaluate_benchmark(
                model, tokenizer, benchmark,
                device=args.device,
                max_samples=args.max_samples,
            )
        
        all_results["benchmarks"][benchmark] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "mode": args.mode,
        }
        
        # Add generation settings if applicable
        if args.mode == "generate":
            all_results["benchmarks"][benchmark]["settings"] = result.get("settings", {})
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    for name, res in all_results["benchmarks"].items():
        print(f"  {name:15s}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")
    
    # Save results
    if args.output:
        output_path = RESULTS_DIR / args.output
    else:
        output_path = RESULTS_DIR / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()

