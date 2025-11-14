================================================================================
DATASET SETUP GUIDE - LLaMA 3 Optimal Configurations
================================================================================

IMPORTANT: The optimal configs require LLaMA-3 tokenized datasets!

================================================================================
QUICK START (Testing)
================================================================================

For immediate testing, prepare the 6B token dataset:

1. Navigate to the dataset directory:
   cd data/slimpajama_6b_llama3

2. Run preparation script:
   python prepare.py
   
   Time: 20-40 minutes
   Storage: ~12GB for tokenized data
   Requires: HuggingFace access (LLaMA-3.1 license)

3. Test your model:
   cd ../..
   python train.py config/full_llama3_1.5b_optimal.py --max_iters=100

================================================================================
DATASET OPTIONS
================================================================================

OPTION 1: slimpajama_6b_llama3 (RECOMMENDED FOR TESTING)
  ✓ Size: 6 billion tokens
  ✓ Preparation: 20-40 minutes
  ✓ Storage: ~12GB
  ✓ Use for: Quick testing, prototyping, validation
  ✓ Status: prepare.py ready to use
  
  Commands:
    cd data/slimpajama_6b_llama3
    python prepare.py

OPTION 2: slimpajama_627b_llama3 (FOR OPTIMAL TRAINING)
  ✓ Size: 627 billion tokens
  ✓ Preparation: 10-20 hours
  ✓ Storage: ~1.2TB
  ✓ Use for: Full optimal training (62B-102B tokens)
  ✓ Status: README created, prepare.py TODO
  
  Note: Only prepare this after validating configs with 6B dataset!

================================================================================
CURRENT DATASET STATUS
================================================================================

READY TO USE:
  ✓ data/slimpajama_6b_gpt2/       (GPT-2 tokenizer, 32K vocab)
  ✓ data/slimpajama_6b_llama/      (LLaMA-2 tokenizer, 32K vocab)
  ✓ data/shakespeare/              (Character-level, testing)
  ✓ data/openwebtext/              (GPT-2 tokenizer)

NEEDS PREPARATION (NEW):
  ⏳ data/slimpajama_6b_llama3/    (LLaMA-3 tokenizer, 128K vocab)
     → Script ready: run prepare.py
  
  ⏳ data/slimpajama_627b_llama3/  (LLaMA-3 tokenizer, 128K vocab)
     → README created, prepare.py TODO

================================================================================
WHY SEPARATE LLaMA-3 DATASETS?
================================================================================

LLaMA-3 uses a DIFFERENT tokenizer than LLaMA-2:
  - LLaMA-2: 32,000 vocabulary (SentencePiece BPE)
  - LLaMA-3: 128,256 vocabulary (tiktoken-based BPE)

Key differences:
  1. Vocab size: 4× larger (128K vs 32K)
  2. Storage: 2× larger files (uint32 vs uint16)
  3. Tokenization: Different BPE algorithm
  4. Better multilingual support in LLaMA-3

You CANNOT use LLaMA-2 datasets with LLaMA-3 configs!

================================================================================
RECOMMENDED WORKFLOW
================================================================================

Step 1: Prepare 6B dataset for testing
  cd data/slimpajama_6b_llama3
  python prepare.py
  # Wait 20-40 minutes

Step 2: Run smoke test (10 iterations)
  cd ../..
  python train.py config/full_llama3_1.5b_optimal.py --max_iters=10

Step 3: Run 1000-iteration test
  torchrun --standalone --nproc_per_node=8 train.py \
    config/full_llama3_1.5b_optimal.py --max_iters=1000

Step 4: Validate loss is decreasing
  tail -f out-llama3-1.5b-optimal/run_*.json

Step 5: (Optional) Prepare full 627B dataset
  cd data/slimpajama_627b_llama3
  # TODO: Create prepare.py (similar to 6B version)
  python prepare.py
  # Wait 10-20 hours

Step 6: Update config for full training
  # Edit config file: dataset = 'slimpajama_627b_llama3'
  # Or override: --dataset=slimpajama_627b_llama3

================================================================================
PREREQUISITES
================================================================================

Python packages:
  pip install transformers>=4.40 datasets tqdm numpy

HuggingFace access:
  1. Accept LLaMA-3.1 license:
     https://huggingface.co/meta-llama/Meta-Llama-3.1-8B
  
  2. Login:
     huggingface-cli login

Storage requirements:
  - 6B dataset: ~18GB total (6GB raw + 12GB tokenized)
  - 627B dataset: ~1.5TB total (300GB raw + 1.2TB tokenized)

================================================================================
TROUBLESHOOTING
================================================================================

Q: "Can I use slimpajama_6b_llama with LLaMA-3 configs?"
A: NO! Vocab mismatch (32K vs 128K). Prepare slimpajama_6b_llama3.

Q: "prepare.py fails with 'tokenizer not found'"
A: Accept LLaMA-3.1 license and run: huggingface-cli login

Q: "I don't have 1.2TB for full dataset"
A: Use 6B dataset for testing. Loss won't be optimal, but architecture works.

Q: "Can I download tokenizer locally?"
A: Yes! See data/slimpajama_6b_llama3/prepare.py comments

Q: "SSH blocks HuggingFace"
A: Download dataset+tokenizer locally, upload to server

================================================================================
FILES CREATED
================================================================================

✓ data/slimpajama_6b_llama3/prepare.py   (Ready to use)
✓ data/slimpajama_6b_llama3/README.md
✓ data/slimpajama_627b_llama3/README.md  (Placeholder)

Updated configs to use slimpajama_6b_llama3 by default:
✓ config/full_llama3_1.5b_optimal.py
✓ config/full_llama3_2.2b_chinchilla.py

================================================================================
NEXT STEPS
================================================================================

1. cd data/slimpajama_6b_llama3
2. python prepare.py
3. Wait 20-40 minutes
4. cd ../..
5. python train.py config/full_llama3_1.5b_optimal.py --max_iters=100

Ready to prepare? Run the prepare.py script! 🚀

================================================================================
