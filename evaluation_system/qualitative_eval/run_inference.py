#!/usr/bin/env python3
"""
Qualitative Evaluation: Run example conversations across checkpoints.

Compares Base, SFT, and DPO models on diverse prompts with various 
temperature and max_tokens settings.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from itertools import product

import torch
import torch.nn.functional as F

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "enhanced_training_system"))
from model_builder import ConfigurableGPT
from model_config import ModelArchitectureConfig


# ══════════════════════════════════════════════════════════════════════════════
# Default Checkpoint Paths
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CHECKPOINTS = {
    "base": "/raid/zhf004/llm_TII/enhanced_training_system/out-qwen3-1.8b-b200-50h/ckpt_160000.pt",
    "sft": "/raid/zhf004/llm_TII/post_training/out-qwen3-1.8b-sft/ckpt_002800.pt",
    "dpo": "/raid/zhf004/llm_TII/post_training/out-qwen3-1.8b-dpo/ckpt_000800.pt",
}

DEFAULT_TOKENIZER = "/raid/zhf004/llm_TII/enhanced_training_system/qwen3_tokenizer"


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint."""
    print(f"📦 Loading {checkpoint_path}...")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Reconstruct model from saved config
    model_args = ckpt['model_args']
    config = ModelArchitectureConfig(**model_args)
    model = ConfigurableGPT(config)
    
    # Load state dict (strip "_orig_mod." prefix from torch.compile)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device, dtype=torch.bfloat16)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ Model loaded: {param_count:.2f}B parameters")
    print(f"   Iteration: {ckpt.get('iter_num', 'unknown')}")
    
    return model


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    device: str = 'cuda'
):
    """Generate text from a prompt."""
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    generated = input_ids.clone()
    
    for _ in range(max_new_tokens):
        # Get logits for last position
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, _ = model(generated, targets=generated)
        
        logits = logits[:, -1, :].float()  # [1, vocab_size]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                logits[0, token_id] /= repetition_penalty
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        if temperature > 0:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        generated = torch.cat([generated, next_token], dim=1)
        
        # Stop on EOS
        if next_token.item() in [tokenizer.eos_token_id, 151645]:  # <|im_end|>
            break
    
    # Decode
    output = tokenizer.decode(generated[0], skip_special_tokens=False)
    response = output[len(prompt):]
    
    return response.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Qualitative evaluation across checkpoints')
    parser.add_argument('--prompts', type=str, default='prompts.json', help='Path to prompts JSON')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--base-ckpt', type=str, default=DEFAULT_CHECKPOINTS['base'])
    parser.add_argument('--sft-ckpt', type=str, default=DEFAULT_CHECKPOINTS['sft'])
    parser.add_argument('--dpo-ckpt', type=str, default=DEFAULT_CHECKPOINTS['dpo'])
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--temperatures', type=str, default='0.3,0.7,1.0', 
                        help='Comma-separated temperatures')
    parser.add_argument('--max-tokens-list', type=str, default='64,128,256',
                        help='Comma-separated max token values')
    parser.add_argument('--prompt-ids', type=str, default=None,
                        help='Comma-separated prompt IDs to run (default: all)')
    parser.add_argument('--models', type=str, default='base,sft,dpo',
                        help='Comma-separated model names to evaluate')
    args = parser.parse_args()
    
    # Parse hyperparameters
    temperatures = [float(t) for t in args.temperatures.split(',')]
    max_tokens_list = [int(m) for m in args.max_tokens_list.split(',')]
    model_names = [m.strip() for m in args.models.split(',')]
    
    # Load prompts
    prompts_path = Path(__file__).parent / args.prompts
    with open(prompts_path) as f:
        prompts_data = json.load(f)
    
    prompts = prompts_data['prompts']
    
    # Filter prompts if specified
    if args.prompt_ids:
        ids = [int(i) for i in args.prompt_ids.split(',')]
        prompts = [p for p in prompts if p['id'] in ids]
    
    print(f"═══════════════════════════════════════════════════════════════════")
    print(f"  Qualitative Evaluation")
    print(f"═══════════════════════════════════════════════════════════════════")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Models: {model_names}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Max tokens: {max_tokens_list}")
    print(f"  Total generations: {len(prompts) * len(model_names) * len(temperatures) * len(max_tokens_list)}")
    print(f"═══════════════════════════════════════════════════════════════════\n")
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Checkpoint mapping
    ckpt_paths = {
        'base': args.base_ckpt,
        'sft': args.sft_ckpt,
        'dpo': args.dpo_ckpt,
    }
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'temperatures': temperatures,
            'max_tokens': max_tokens_list,
            'models': model_names,
        },
        'checkpoints': {k: v for k, v in ckpt_paths.items() if k in model_names},
        'generations': []
    }
    
    # Run for each model
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"  Loading {model_name.upper()} model")
        print(f"{'='*70}")
        
        model = load_model(ckpt_paths[model_name], args.device)
        
        for prompt_data in prompts:
            prompt_id = prompt_data['id']
            category = prompt_data['category']
            prompt = prompt_data['prompt']
            
            for temp, max_tokens in product(temperatures, max_tokens_list):
                print(f"\n[{model_name}] Prompt {prompt_id} ({category}) | temp={temp}, max_tokens={max_tokens}")
                print(f"  Q: {prompt[:60]}...")
                
                start_time = time.time()
                response = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    device=args.device
                )
                gen_time = time.time() - start_time
                
                print(f"  A: {response[:100]}..." if len(response) > 100 else f"  A: {response}")
                print(f"  Time: {gen_time:.2f}s")
                
                results['generations'].append({
                    'model': model_name,
                    'prompt_id': prompt_id,
                    'category': category,
                    'prompt': prompt,
                    'temperature': temp,
                    'max_tokens': max_tokens,
                    'response': response,
                    'generation_time': gen_time,
                })
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Save results
    RESULTS_DIR = Path(__file__).parent.parent / "results" / "qualitative"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or f"qualitative_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = RESULTS_DIR / output_path
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Print summary table
    print("\n📊 Generation Summary by Model:")
    for model_name in model_names:
        model_gens = [g for g in results['generations'] if g['model'] == model_name]
        avg_time = sum(g['generation_time'] for g in model_gens) / len(model_gens)
        avg_len = sum(len(g['response']) for g in model_gens) / len(model_gens)
        print(f"  {model_name:5s}: {len(model_gens)} generations, avg {avg_time:.2f}s, avg {avg_len:.0f} chars")


if __name__ == '__main__':
    main()

