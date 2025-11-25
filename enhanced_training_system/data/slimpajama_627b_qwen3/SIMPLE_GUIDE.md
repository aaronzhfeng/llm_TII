# SlimPajama-627B: Simple 2-Step Guide

## ✅ New Simplified Workflow (No Rate Limits!)

### Step 1: Download (8-12 hours)

```bash
cd /raid/zhf004/llm_TII/enhanced_training_system/data/slimpajama_627b_qwen3

# Run in tmux (survives disconnection)
tmux new -s download
python3 download.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t download
```

**What it does:**
- Downloads 895GB sequentially (no parallel workers)
- No rate limit issues! ✅
- Saves to `/raid/zhf004/huggingface_cache/`
- Can stop/resume anytime (progress cached)

**Monitor progress:**
```bash
watch -n 30 'du -sh /raid/zhf004/huggingface_cache/datasets'
```

---

### Step 2: Tokenize (2-4 hours)

After download completes:

```bash
cd /raid/zhf004/llm_TII/enhanced_training_system/data/slimpajama_627b_qwen3
python3 prepare.py
```

**What it does:**
- Tokenizes with Qwen3 tokenizer
- Uses all 224 CPU cores ⚡
- Creates `train.bin` (~1.2TB), `val.bin`, `test.bin`
- No network access needed

---

## 📊 Expected Output

After both steps complete:

```
slimpajama_627b_qwen3/
├── download.py          # Step 1 script
├── prepare.py           # Step 2 script
├── train.bin            # 627B tokens (~1.2TB)
├── val.bin              # 500M tokens (~1GB)
├── test.bin             # 500M tokens (~1GB)
└── meta.pkl             # Metadata
```

---

## 🎯 Why This Works Better

| Old Approach | New Approach |
|--------------|--------------|
| Parallel downloads → Rate limits 💥 | Sequential → No rate limits ✅ |
| Complex flags (--max_workers, etc) | Simple: just run 2 scripts |
| One script does both | Clean separation |

---

## ⏱️ Total Time

```
Step 1 (download.py):  8-12 hours
Step 2 (prepare.py):   2-4 hours
─────────────────────────────────
Total:                 10-16 hours
```

Much more reliable than fighting rate limits! 🎉

