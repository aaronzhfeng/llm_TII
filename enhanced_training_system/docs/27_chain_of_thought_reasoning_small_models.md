# Chain-of-Thought Reasoning in Small Language Models

**Date:** November 13, 2025  
**Phase:** Research Context - Reasoning Capabilities  
**Status:** Background Research

---

## Overview

This document examines whether small language models (1-2B parameters) can effectively perform chain-of-thought (CoT) reasoning, and what training approaches enable this capability.

**TL;DR:** Small models *can* do CoT reasoning, but it doesn't emerge naturally - it requires explicit training with rationales (distillation/fine-tuning).

---

## 1. Natural CoT Emergence: Size Matters

### The Original CoT Findings (Wei et al., 2022)

The seminal "Chain-of-Thought Prompting Elicits Reasoning" paper showed:

**Key Result:** CoT gains only "emerge" in sufficiently large models
- **PaLM 62B-540B:** Clear improvements with "let's think step by step"
- **GPT-3 175B:** Similar emergent behavior
- **Smaller models (<10B):** CoT prompting barely helps or even hurts

### What Happens with Small Models + CoT Prompting?

**Vanilla 1-2B base models prompted with "think step by step":**

❌ **Failure modes:**
- Longer, rambling outputs
- No improvement in accuracy
- Sometimes *worse* performance than direct prompting
- Hallucinated or nonsensical reasoning steps

✅ **What they can do:**
- Follow the format (produce text after "let's think:")
- Generate syntactically correct chain-like text
- Mimic the *appearance* of reasoning

❌ **What they can't do naturally:**
- Actually *use* the reasoning to improve answers
- Maintain logical consistency across steps
- Self-correct through reasoning

**Conclusion:** CoT is not a "free" capability at small scale - it requires training.

---

## 2. Training Small Models for CoT: Distillation Approaches

### Core Insight

**Chain-of-thought reasoning is a learnable skill**, not just an emergent property. Small models can acquire CoT capabilities through:

1. **Supervised fine-tuning** on reasoning traces
2. **Distillation** from larger teacher models
3. **Reinforcement learning** on task performance

### Key Research Demonstrations

#### a) Distilling Step-by-Step (Google Research, 2023)

**Approach:**
- Train small task-specific models on teacher CoT traces
- Use rationales as intermediate supervision
- Much less data than traditional fine-tuning

**Results:**
- Small models outperform few-shot prompted large LLMs
- 700× fewer training examples needed vs standard fine-tuning
- Works on models as small as T5-Base (~220M params)

**Key takeaway:** With the right supervision, tiny models can reason.

#### b) Symbolic CoT Distillation (SCoTD)

**Approach:**
- Formalize CoT as symbolic reasoning patterns
- Distill these patterns from large teachers to small students
- Structured supervision on rationalizations

**Results:**
- Small Language Models (SLMs) learn systematic reasoning
- Generalizes better than training on answers alone
- Improves out-of-distribution performance

#### c) LFM-1.3B-Math (Large Foundation Model for Math)

**Approach:**
- ~1.3B parameter model
- Large-scale supervised CoT training
- Reinforcement learning on final answer correctness
- Test-time verification

**Results:**
- **Strong math reasoning at 1.3B scale**
- Competitive with much larger models on math benchmarks
- Proof that 1B+ can be serious reasoners with proper training

#### d) "s1" Model (Qwen2.5-based, 2024)

**Approach:**
- Distilled from Qwen2.5
- Trained on ~1,000 high-quality CoT examples
- Test-time scaling (generate multiple reasoning paths)
- Best-of-N selection

**Results:**
- **Rivals OpenAI's o1 on math competitions**
- Uses <1K training examples (extreme data efficiency)
- Shows that quality > quantity for CoT distillation

### Common Success Patterns

Across all successful small-model CoT work:

✅ **What works:**
- High-quality teacher rationales (from GPT-4, Claude, etc.)
- Short, focused reasoning chains (not rambling)
- Task-specific fine-tuning (not trying to be general reasoner)
- Verification/RL on final answers
- Test-time scaling (multiple attempts + selection)

❌ **What doesn't work:**
- Just prompting with "think step by step"
- Training on low-quality synthetic data
- Trying to replicate o1's generality at 1B scale
- Very long reasoning chains (100+ tokens)

---

## 3. Requirements for CoT in Small Models

### Minimum Training Pipeline

To enable CoT reasoning in a 1-2B model:

**1. Teacher Model**
- Access to a larger model with good reasoning (GPT-4, Claude, Qwen-72B, etc.)
- Can generate high-quality rationales for your target tasks

**2. Dataset Construction**
```
Structure: (input, reasoning_trace, final_answer)

Example:
Input: "If 3 apples cost $2, how much for 7 apples?"
Reasoning: "First, find cost per apple: $2/3 = $0.67. 
           Then multiply by 7: $0.67 × 7 = $4.67"
Answer: "$4.67"
```

**3. Distillation Training**
- Supervised fine-tuning on (input → reasoning → answer)
- Loss on both reasoning tokens AND final answer
- ~1K-10K high-quality examples for focused tasks
- ~100K+ examples for broader reasoning

**4. (Optional) Reinforcement Learning**
- RL on final answer correctness
- Encourages model to use reasoning to improve accuracy
- Not just generate plausible-sounding steps

**5. Test-Time Scaling**
- Generate multiple reasoning paths
- Use majority voting or verification
- Improves reliability significantly

### Data Requirements

**Minimal (focused task):**
- 1,000-5,000 examples with rationales
- Single task domain (e.g., grade-school math)
- Can achieve strong task-specific performance

**Substantial (broader reasoning):**
- 50,000-500,000 examples across multiple domains
- Mix of math, logic, common sense, coding
- Approaches "general" small reasoning model

**Quality > Quantity:**
- 1K GPT-4 rationales > 100K synthetic junk
- Diverse reasoning patterns matter more than volume
- Test-time scaling compensates for limited training data

---

## 4. Implications for Qwen3 1-2B Training

### What CoT Means for Our Models

**Base training (pre-training on tokens):**
- ❌ Will NOT produce CoT reasoning naturally
- ✅ Will have language understanding foundations
- ✅ Can *format* reasoning-like text
- ❌ Won't reliably *use* reasoning to improve answers

**To add CoT capability:**
- Need post-training phase with reasoning traces
- Either distillation from larger model OR
- Supervised fine-tuning on curated reasoning datasets

### Recommended Approach for Qwen3-1.5B

**Phase 1: Base Training (60 hours on DGX B200)**
- Train 1.5B Qwen3-style model on 80-100B tokens
- Standard next-token prediction
- No CoT expectations - just solid base model

**Phase 2: (Optional) CoT Fine-Tuning**
- **If desired:** Add reasoning capability post-hoc
- Use Qwen-72B or GPT-4 to generate reasoning traces
- ~10K high-quality examples in target domains
- Fine-tune for 1-2 hours
- Test-time scaling for reliability

**Realistic Expectations:**
- ✅ Base model: Strong language understanding, comparable to LLaMA 3 1.5B
- ⚠️ Base model: No special reasoning beyond typical 1-2B capabilities
- ✅ After CoT fine-tuning: Task-specific reasoning improvements
- ❌ After CoT fine-tuning: NOT competing with o1/R1 on generality

### Cost-Benefit Analysis

**Skip CoT entirely:**
- ✅ Focus compute on base model quality
- ✅ Simpler training pipeline
- ✅ Still get strong general-purpose LLM
- ❌ No reasoning superpowers

**Add CoT post-training:**
- ✅ Enables reasoning on specific tasks
- ✅ Differentiates from base Qwen3
- ⚠️ Requires teacher model access + curation effort
- ⚠️ Only improves target domains
- ⚠️ Not a "general reasoner" at 1-2B scale

---

## 5. Practical Recommendations

### For Research/Academic Projects

**If goal is architecture/efficiency research:**
- ❌ Skip CoT entirely
- ✅ Focus on: base model quality, MFU optimization, scaling laws
- ✅ Easier to compare with other base models

**If goal is demonstrating reasoning:**
- ✅ Add CoT distillation phase
- ✅ Pick 1-2 focused tasks (e.g., grade-school math, code reasoning)
- ✅ Use test-time scaling for impressive demos
- ⚠️ Be honest about limitations

### For Production/Applied Projects

**If deploying for general use:**
- ❌ Don't promise "reasoning" without CoT training
- ✅ Train solid base model
- ✅ (Optional) Add task-specific reasoning for high-value domains

**If building reasoning assistant:**
- ✅ Invest in CoT distillation (10-50K examples)
- ✅ Use larger teacher model for traces
- ✅ Implement test-time scaling
- ✅ Verify reasoning quality before deployment

---

## 6. Comparison: Small Model CoT vs Large Model CoT

| Aspect | Large Models (70B+) | Small Models (1-2B) with CoT Training |
|--------|---------------------|--------------------------------------|
| **Natural CoT** | ✅ Emerges from scale | ❌ Requires explicit training |
| **Reasoning Quality** | ✅ Sophisticated, general | ⚠️ Good on trained tasks, weak elsewhere |
| **Training Cost** | 💰💰💰 Extremely expensive | 💰 Relatively cheap post-training |
| **Inference Cost** | 💰💰💰 High (slow, large) | ✅ Fast and cheap |
| **Generalization** | ✅ Broad reasoning ability | ⚠️ Narrow to training domains |
| **Test-Time Scaling** | ✅ Effective | ✅ Highly effective (compensates for size) |
| **Data Efficiency** | ❌ Needs massive pre-training | ✅ 1K-10K examples can work |

---

## 7. State-of-the-Art Examples (2024-2025)

### Small Models with Strong Reasoning

**1. LFM-1.3B-Math**
- 1.3B parameters
- Math-specific CoT training
- Competitive with Llama-2-13B on math benchmarks

**2. "s1" (Qwen2.5-based)**
- ~7B parameters (still "small" vs 70B+)
- 1K high-quality examples
- Rivals o1-preview on AIME math competition

**3. Phi-3-Mini (3.8B)**
- Strong reasoning on benchmarks
- Heavily curated training data
- Shows quality training data > model size

### Key Lessons

1. **Small can compete** - on specific tasks with right training
2. **Test-time scaling is crucial** - multiple attempts + verification
3. **Quality training data matters more than scale** - 1K good examples > 100K mediocre
4. **Don't expect generality** - focused reasoning > trying to be o1-lite

---

## 8. Conclusion

### Can 1-2B Models Do CoT Reasoning?

**Yes, but:**
- ❌ Not naturally/emergently
- ✅ With proper distillation training
- ⚠️ On specific task domains
- ✅ With test-time scaling for reliability
- ❌ Not as general as large models

### For Our Qwen3-1.5B Implementation

**Recommended path:**
1. **Base training:** Focus on solid foundation (60 hours, 80-100B tokens)
2. **Evaluate base model:** Compare with LLaMA 3, Qwen3-0.6B
3. **CoT decision point:** 
   - If research focused: Skip CoT, publish base model results
   - If application focused: Add targeted CoT for specific domains
4. **Post-training (if desired):** 1-2 hours, 10K examples, test-time scaling

**Don't expect:** A 1.5B model to rival DeepSeek-R1 or o1 on general reasoning.

**Do expect:** Task-specific reasoning improvements where you invest training effort.

---

## References

- Wei et al., 2022: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- Ho et al., 2023: "Large Language Models Are Reasoning Teachers" (Distilling step-by-step)
- Zhao et al., 2024: "Symbolic Chain-of-Thought Distillation" (SCoTD)
- Anonymous, 2024: "s1: Test-Time Scaling for Small Reasoning Models"
- Lecture notes: "When Chain of Thought May Not Be Effective" (Stanford CS324)

---

**Next Steps:** See `28_qwen3_implementation_plan.md` for how we'll integrate these insights into our Qwen3 training approach.

